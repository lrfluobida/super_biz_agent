"""向量存储管理器 - 封装 Milvus VectorStore 操作"""

from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.keyword_search_service import keyword_search_service
from app.services.vector_embedding_service import vector_embedding_service


# 统一使用 biz collection
COLLECTION_NAME = "biz"


def _build_dense_texts(documents: List[Document]) -> List[str]:
    """为 dense embedding 构建输入文本，有上下文时拼接到 chunk 前面。

    BM25 和 content 字段仍使用原始 page_content，不受影响。
    """
    result: List[str] = []
    for doc in documents:
        ctx = doc.metadata.get("_doc_context", "")
        if ctx:
            result.append(f"{ctx}\n\n{doc.page_content}")
        else:
            result.append(doc.page_content)
    return result


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self):
        """初始化向量存储管理器"""
        self.vector_store = None
        self.collection_name = COLLECTION_NAME
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """初始化 Milvus VectorStore"""
        try:
            # 必须在 PyMilvus / langchain_milvus 访问 Collection 之前建立连接，
            # 否则会出现 ConnectionNotExistException: should create connection first.
            # （模块导入时就会执行此处，早于 FastAPI lifespan 中的 milvus_manager.connect）
            _ = milvus_manager.connect()

            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }

            # 创建 LangChain Milvus VectorStore
            # 使用 biz collection，字段映射：text_field -> content, vector_field -> vector
            self.vector_store = Milvus(
                embedding_function=vector_embedding_service,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,  # 使用自定义 id
                drop_old=False,
                text_field="content",  # 文本内容存储到 content 字段
                vector_field="vector",  # 向量存储到 vector 字段
                primary_field="id",  # 主键字段
                metadata_field="metadata",  # 元数据字段
            )

            logger.info(
                f"VectorStore 初始化成功: {config.milvus_host}:{config.milvus_port}, "
                f"collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"VectorStore 初始化失败: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        批量添加文档到向量存储（同时写入 dense 和 sparse 向量）

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        try:
            import time
            import uuid
            start_time = time.time()

            ids = [str(uuid.uuid4()) for _ in documents]
            # 用于 sparse 嵌入和 content 字段（原始文本）
            texts = [doc.page_content for doc in documents]
            # 用于 dense 嵌入（可拼接文档上下文摘要）
            dense_texts = _build_dense_texts(documents)

            # 如果 BM25 已训练，使用 PyMilvus 直接插入 dense + sparse
            if keyword_search_service.is_fitted:
                return self._add_documents_with_sparse(documents, ids, texts, dense_texts, start_time)

            # 回退：仅 dense 向量，使用 PyMilvus 直接插入（带空 sparse_vector）
            dense_vectors = vector_embedding_service.embed_documents(dense_texts)
            insert_data = []
            for i, doc in enumerate(documents):
                row = {
                    "id": ids[i],
                    "vector": dense_vectors[i],
                    "sparse_vector": {},
                    "content": doc.page_content[:8000],
                    "metadata": doc.metadata,
                }
                insert_data.append(row)
            collection = milvus_manager.get_collection()
            collection.insert(insert_data)
            collection.flush()
            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {len(documents)} 个文档到 VectorStore 完成 (仅 dense), "
                f"耗时: {elapsed:.2f}秒"
            )
            return ids

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def _add_documents_with_sparse(
        self,
        documents: List[Document],
        ids: List[str],
        texts: List[str],
        dense_texts: List[str],
        start_time: float,
    ) -> List[str]:
        """使用 PyMilvus 直接插入 dense + sparse 向量

        texts: 用于 sparse 嵌入和 content 字段（原始文本）
        dense_texts: 用于 dense 嵌入（可包含文档上下文摘要）
        """
        import time

        # 生成 dense 向量（使用可能拼接了上下文的文本）
        dense_vectors = vector_embedding_service.embed_documents(dense_texts)
        # 生成 sparse 向量（始终使用原始文本，不受上下文影响）
        sparse_vectors = keyword_search_service.encode_documents(texts)

        # 构建插入数据
        insert_data = []
        for i, doc in enumerate(documents):
            row = {
                "id": ids[i],
                "vector": dense_vectors[i],
                "sparse_vector": sparse_vectors[i] if sparse_vectors[i] else {},
                "content": doc.page_content[:8000],  # 截断到字段限制
                "metadata": doc.metadata,
            }
            insert_data.append(row)

        collection = milvus_manager.get_collection()
        collection.insert(insert_data)
        collection.flush()

        elapsed = time.time() - start_time
        logger.info(
            f"批量添加 {len(documents)} 个文档到 VectorStore 完成 (dense + sparse), "
            f"耗时: {elapsed:.2f}秒, 平均: {elapsed/len(documents):.2f}秒/个"
        )
        return ids

    def delete_by_source(self, file_path: str) -> int:
        """
        删除指定文件的所有文档

        Args:
            file_path: 文件路径

        Returns:
            int: 删除的文档数量
        """
        try:
            # 使用 milvus_manager 获取已连接的 collection
            collection = milvus_manager.get_collection()
            
            # metadata 是 JSON 字段，使用 JSON 路径查询语法
            # _source 是文档的来源文件路径
            expr = f'metadata["_source"] == "{file_path}"'
            
            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0
            
            logger.info(f"删除文件旧数据: {file_path}, 删除数量: {deleted_count}")
            return deleted_count
            
        except Exception as e:
            logger.warning(f"删除旧数据失败 (可能是首次索引): {e}")
            return 0

    def get_vector_store(self) -> Milvus:
        """
        获取 VectorStore 实例

        Returns:
            Milvus: VectorStore 实例
        """
        return self.vector_store

    def rebuild_bm25_from_collection(self) -> None:
        """从 Milvus 中读取全部文档内容，重建 BM25 模型并持久化"""
        try:
            import time
            start_time = time.time()

            collection = milvus_manager.get_collection()
            # 分批读取所有文档内容
            all_texts: list[str] = []
            offset = 0
            batch_size = 500

            while True:
                results = collection.query(
                    expr="id != ''",
                    output_fields=["content"],
                    offset=offset,
                    limit=batch_size,
                )
                if not results:
                    break
                all_texts.extend(r["content"] for r in results)
                offset += batch_size

            if not all_texts:
                logger.warning("Collection 中没有文档，跳过 BM25 重建")
                return

            keyword_search_service.fit(all_texts)
            keyword_search_service.save(config.bm25_model_path)

            elapsed = time.time() - start_time
            logger.info(
                f"BM25 模型重建完成: 文档数={len(all_texts)}, 耗时={elapsed:.2f}秒"
            )
        except Exception as e:
            logger.error(f"BM25 模型重建失败: {e}")
            raise

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"相似度搜索完成: query='{query}', 结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []


# 全局单例
vector_store_manager = VectorStoreManager()

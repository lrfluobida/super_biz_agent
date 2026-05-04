import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class HybridSearchServiceTestCase(unittest.TestCase):
    """测试混合检索的回退逻辑和 RRF 合并"""

    def setUp(self):
        import app.config

        self._orig_enabled = app.config.config.hybrid_search_enabled

    def tearDown(self):
        import app.config

        app.config.config.hybrid_search_enabled = self._orig_enabled

    def test_dense_only_fallback_when_disabled(self):
        """use_hybrid=False 时走纯向量检索"""
        from app.services.hybrid_search_service import hybrid_search

        with patch(
            "app.services.hybrid_search_service._dense_search",
            return_value=[Document(page_content="test", metadata={"_milvus_id": "1"})],
        ) as mock_dense:
            docs, meta = hybrid_search("test", top_k=3, use_hybrid=False)
            mock_dense.assert_called_once()
            self.assertEqual(meta["mode"], "dense_only")
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].metadata["_recall_path"], "dense")

    def test_dense_only_fallback_when_bm25_not_fitted(self):
        """BM25 未训练时回退纯向量检索"""
        from app.services.hybrid_search_service import hybrid_search
        from app.services import keyword_search_service as kss

        kss.keyword_search_service._bm25 = None

        with patch(
            "app.services.hybrid_search_service._dense_search",
            return_value=[Document(page_content="fallback", metadata={"_milvus_id": "2"})],
        ) as mock_dense:
            docs, meta = hybrid_search("test", top_k=3)
            mock_dense.assert_called_once()
            self.assertEqual(docs[0].page_content, "fallback")
            self.assertEqual(docs[0].metadata["_recall_path"], "dense")

    def test_dense_search_returns_documents(self):
        """纯向量检索返回 LangChain Document"""
        from app.services.hybrid_search_service import _dense_search

        mock_collection = MagicMock()
        mock_hit = MagicMock()
        mock_hit.entity.get.side_effect = lambda key, default=None: {
            "id": "test-1",
            "content": "some content",
            "metadata": {"_file_name": "test.md"},
        }.get(key, default)
        mock_hit.distance = 0.95
        mock_collection.search.return_value = [[mock_hit]]

        with patch(
            "app.services.hybrid_search_service.milvus_manager.get_collection",
            return_value=mock_collection,
        ), patch(
            "app.services.hybrid_search_service.vector_embedding_service.embed_query",
            return_value=[0.1] * 1024,
        ):
            docs = _dense_search("test query", 3)
            self.assertEqual(len(docs), 1)
            self.assertIsInstance(docs[0], Document)
            self.assertEqual(docs[0].page_content, "some content")
            self.assertEqual(docs[0].metadata["_file_name"], "test.md")
            self.assertEqual(docs[0].metadata["_milvus_id"], "test-1")

    def test_hybrid_search_fallback_on_error(self):
        """hybrid_search 稀疏检索异常时回退纯向量"""
        import app.config
        from app.services.hybrid_search_service import hybrid_search
        from app.services import keyword_search_service as kss

        app.config.config.hybrid_search_enabled = True
        kss.keyword_search_service._bm25 = MagicMock()

        fallback_doc = Document(page_content="fb", metadata={"_milvus_id": "fb"})
        with patch(
            "app.services.hybrid_search_service._dense_search",
            return_value=[fallback_doc],
        ), patch(
            "app.services.hybrid_search_service._sparse_search",
            side_effect=RuntimeError("sparse search failed"),
        ):
            docs, meta = hybrid_search("test", top_k=3)
            self.assertEqual(meta["mode"], "dense_fallback")
            self.assertEqual(len(docs), 1)

    def test_rrf_merge_tags_recall_path(self):
        """RRF 合并后正确标记每个文档的召回路径"""
        from app.services.hybrid_search_service import _rrf_merge

        dense_docs = [
            Document(page_content="A", metadata={"_milvus_id": "1"}),
            Document(page_content="B", metadata={"_milvus_id": "2"}),
            Document(page_content="C", metadata={"_milvus_id": "3"}),
        ]
        sparse_docs = [
            Document(page_content="B", metadata={"_milvus_id": "2"}),
            Document(page_content="D", metadata={"_milvus_id": "4"}),
        ]

        merged = _rrf_merge(dense_docs, sparse_docs, k=5, top_k=3)

        self.assertEqual(len(merged), 3)
        # Doc B is in both
        b = [d for d in merged if d.metadata["_milvus_id"] == "2"][0]
        self.assertEqual(b.metadata["_recall_path"], "both")
        # Doc A is only in dense
        a = [d for d in merged if d.metadata["_milvus_id"] == "1"][0]
        self.assertEqual(a.metadata["_recall_path"], "dense")
        # Doc D is only in sparse
        d = [d for d in merged if d.metadata["_milvus_id"] == "4"][0]
        self.assertEqual(d.metadata["_recall_path"], "keyword")

    def test_rrf_respects_top_k(self):
        """RRF 合并后数量不超过 top_k"""
        from app.services.hybrid_search_service import _rrf_merge

        dense_docs = [
            Document(page_content=str(i), metadata={"_milvus_id": str(i)})
            for i in range(10)
        ]
        sparse_docs: list[Document] = []

        merged = _rrf_merge(dense_docs, sparse_docs, k=5, top_k=3)
        self.assertEqual(len(merged), 3)

    def test_hybrid_search_returns_recall_meta(self):
        """混合检索返回 recall_meta 包含两路信息"""
        import app.config
        from app.services.hybrid_search_service import hybrid_search
        from app.services import keyword_search_service as kss

        app.config.config.hybrid_search_enabled = True

        # 模拟 BM25 已训练
        kss.keyword_search_service._bm25 = MagicMock()

        dense_docs = [
            Document(page_content="X", metadata={"_milvus_id": "a", "_file_name": "a.md", "_source": "a.md"}),
        ]
        sparse_docs = [
            Document(page_content="Y", metadata={"_milvus_id": "b", "_file_name": "b.md", "_source": "b.md"}),
        ]

        with patch(
            "app.services.hybrid_search_service._dense_search",
            return_value=dense_docs,
        ), patch(
            "app.services.hybrid_search_service._sparse_search",
            return_value=sparse_docs,
        ):
            docs, meta = hybrid_search("test", top_k=3)
            self.assertEqual(meta["mode"], "hybrid")
            self.assertEqual(meta["dense_count"], 1)
            self.assertEqual(meta["keyword_count"], 1)
            self.assertIn("a", meta["dense_ids"])
            self.assertIn("b", meta["keyword_ids"])

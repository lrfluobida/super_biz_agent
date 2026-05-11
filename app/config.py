"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Dict, Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "SuperBizAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "qwen3.6-flash"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "qwen3.6-flash"  # 使用快速响应模型，不带扩展思考

    # Hybrid Search 配置（二路召回 + RRF）
    hybrid_search_enabled: bool = True
    hybrid_rrf_k: int = 5
    hybrid_per_ranker_limit: int = 10
    bm25_model_path: str = "volumes/bm25_model.pkl"

    # 对话记忆配置
    memory_window_turns: int = 5
    memory_prompt_budget_tokens: int = 24000
    memory_summary_soft_ratio: float = 0.6
    memory_summary_hard_ratio: float = 0.8
    memory_reserved_output_tokens: int = 4096
    memory_sqlite_path: str = "volumes/memory/session_memory.sqlite3"

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # Contextual Chunking — 索引时为每个文档生成上下文摘要，嵌入时拼到 chunk 前面
    contextual_chunking_enabled: bool = True

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"
    prometheus_enabled: bool = False
    prometheus_base_url: str = "http://localhost:9090"
    prometheus_timeout_seconds: float = 5.0

    # Query Rewrite 配置
    rewrite_enabled: bool = True

    # 本地改写模型（Ollama）
    rewrite_local_model_url: str = "http://localhost:11434/v1"
    rewrite_local_model_name: str = "qwen2.5:1.5b"
    rewrite_local_model_temperature: float = 0.1
    rewrite_local_model_timeout: int = 10  # 秒

    # Router
    rewrite_router_enabled: bool = True

    # Drift Guard
    rewrite_drift_threshold: float = 0.65
    rewrite_drift_moderate_threshold: float = 0.40

    # 并行检索
    rewrite_parallel_retrieval_enabled: bool = True
    rewrite_parallel_max_workers: int = 4

    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug(cls, value: Any) -> Any:
        """兼容 DEBUG=release 等非标准布尔值"""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on", "debug"}:
                return True
            if normalized in {"false", "0", "no", "off", "release", "prod", "production"}:
                return False
        return value

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()

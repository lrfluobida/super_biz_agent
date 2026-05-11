"""Query Rewrite 数据模型

RouterResult / DecomposeResult / StepBackResult / ContextualizeResult /
DriftCheckResult / RewritePipelineResult
"""

from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    DIRECT = "direct"
    DECOMPOSE = "decompose"
    STEP_BACK = "step_back"
    CONTEXTUALIZE = "contextualize"


class RouterSource(str, Enum):
    RULE = "rule"
    MODEL = "model"
    FALLBACK = "fallback"


class RouterResult(BaseModel):
    intent: Intent = Field(description="路由分类结果")
    reason: str = Field(default="", description="分类理由")
    source: RouterSource = Field(default=RouterSource.FALLBACK, description="决策来源：规则/模型/降级")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度")


class DecomposeResult(BaseModel):
    sub_queries: list[str] = Field(description="拆分后的子查询列表（2-4个）")
    original_topic: str = Field(default="", description="原始问题主题概括")


class StepBackResult(BaseModel):
    step_back_query: str = Field(description="泛化后的通用检索查询")
    retained_specifics: str = Field(default="", description="保留的关键具体信息")


class ContextualizeResult(BaseModel):
    standalone_query: str = Field(description="补全上下文后的独立完整检索语句")


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class DriftCheckResult(BaseModel):
    similarity: float = Field(description="改写 query 与原始 query 的余弦相似度")
    severity: DriftSeverity = Field(description="漂移严重程度")
    rewritten_usable: bool = Field(description="改写结果是否可用")
    action: str = Field(default="", description="建议动作：use_rewritten / use_both / use_original")


class RewritePipelineResult(BaseModel):
    # 最终喂给 Agent 的问题文本
    agent_question: str = Field(description="改写后的最终问题文本，喂给 Agent")
    # 用户原始输入（用于 memory 存储，不被改写覆盖）
    original_query: str = Field(description="用户原始查询文本")
    # 路由结果
    intent: Intent = Field(default=Intent.DIRECT)
    router_result: RouterResult = Field(default_factory=lambda: RouterResult(intent=Intent.DIRECT))
    # 是否实际执行了改写
    rewritten: bool = Field(default=False)
    # 改写详情（trace 用）
    rewrite_meta: dict = Field(default_factory=dict, description="改写详情")
    # 漂移检测结果（未启用 drift guard 时为 None）
    drift_check: Optional[DriftCheckResult] = Field(default=None, description="漂移检测结果")
    # 改写后需要追加到 System Prompt 的指令（decompose/step_back 场景）
    extra_system_prompt: str = Field(default="", description="额外系统提示词")

"""查询意图路由 — Stage 1 规则 + Stage 2 模型分类

规则优先（<1ms），命中直接返回；规则未命中时调用本地小模型分类（~1-2s）。
模型不可用时 fallback 到 direct，宁可不改写也不错改写。
"""

import re
from typing import Optional

from loguru import logger

from app.config import config
from app.models.rewrite import Intent, RouterResult, RouterSource
from app.services.rewrite_model_service import rewrite_model_service

ROUTING_PROMPT = """你是一个查询意图分类器。将查询归为以下4类之一：
- direct: 简单明确，可直接检索
- decompose: 复合多主题，需拆分子查询
- step_back: 过于具体，需泛化抽象
- contextualize: 包含指代词，需补全上下文

输出严格JSON：{"intent": "...", "reason": "..."}

示例：
"CPU使用率过高怎么排查" → {"intent": "direct", "reason": "单一明确问题"}
"CPU和内存的排查流程有什么区别" → {"intent": "decompose", "reason": "两个独立主题对比"}
"针对level=70的阈值设置建议" → {"intent": "step_back", "reason": "过于具体，应泛化为告警阈值设置"}
"它为什么会触发" → {"intent": "contextualize", "reason": "代词指代需解析历史"}"""

CONTEXTUALIZE_PRONOUNS = re.compile(
    r"(?:它(?:们)?|这个|那个|这些|那些|上面|之前|刚才|他|她|其)"
)

CONTEXTUALIZE_SHORT_MAX_LEN = 5

DECOMPOSE_MARKERS = [
    "区别", "分别", "相比", "比较", "不同", "vs",
    "以及", "哪个更", "差异",
]

# 简单中文实体识别：匹配中文/英文名词短语（不依赖 jieba）
_ENTITY_PATTERN = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9_.-]*)"  # 英文技术名词
    r"|(?:[一-鿿]{2,})"  # 中文词语（2字以上）
)

STEP_BACK_NUMERIC = re.compile(r"\d+")
STEP_BACK_SPECIFIC_WORDS = re.compile(
    r"(?:阈值|参数|设置|配置|数值|级别|等级|level|timeout|threshold|scrape|interval|max_|min_)"
)


class QueryRouter:
    """查询意图路由"""

    def __init__(self) -> None:
        self._enabled = config.rewrite_enabled and config.rewrite_router_enabled

    async def route(self, query: str, session_round: int = 1) -> RouterResult:
        """路由查询意图

        Args:
            query: 用户原始查询
            session_round: 当前会话轮次（1-based，user 消息计数）

        Returns:
            RouterResult: 意图分类结果
        """
        if not self._enabled:
            return RouterResult(
                intent=Intent.DIRECT,
                reason="rewrite disabled",
                source=RouterSource.FALLBACK,
                confidence=1.0,
            )

        # Stage 1: 规则快速路径
        rule_result = self._rule_route(query, session_round)
        if rule_result:
            logger.info(f"[Router] 规则命中: intent={rule_result.intent.value}, reason={rule_result.reason}")
            return rule_result

        # Stage 2: 模型分类
        logger.info(f"[Router] 规则未命中，进入 Stage 2 模型分类")
        return await self._model_route(query)

    def _rule_route(self, query: str, session_round: int) -> Optional[RouterResult]:
        """Stage 1: 规则路由

        返回 RouterResult 表示规则命中，返回 None 表示需进入 Stage 2。
        """

        # contextualize: 代词检测 + 非首轮
        if session_round > 1:
            if CONTEXTUALIZE_PRONOUNS.search(query):
                return RouterResult(
                    intent=Intent.CONTEXTUALIZE,
                    reason="包含指代词且非首轮对话",
                    source=RouterSource.RULE,
                    confidence=0.95,
                )
            if len(query) <= CONTEXTUALIZE_SHORT_MAX_LEN and not _has_question_word(query):
                return RouterResult(
                    intent=Intent.CONTEXTUALIZE,
                    reason="短查询且非首轮，省略主语",
                    source=RouterSource.RULE,
                    confidence=0.85,
                )

        # decompose: 对比/并列标记
        if _has_decompose_marker(query) and _entity_count(query) >= 2:
            return RouterResult(
                intent=Intent.DECOMPOSE,
                reason="包含对比标记且含多个独立实体",
                source=RouterSource.RULE,
                confidence=0.90,
            )

        # decompose: 多问号拼接
        if query.count("?") >= 2 or query.count("？") >= 2:
            return RouterResult(
                intent=Intent.DECOMPOSE,
                reason="多个问句拼接",
                source=RouterSource.RULE,
                confidence=0.95,
            )

        # step_back: 过度具体
        if _is_overly_specific(query):
            return RouterResult(
                intent=Intent.STEP_BACK,
                reason="包含具体数值/参数且缺少抽象概念",
                source=RouterSource.RULE,
                confidence=0.70,
            )

        # direct: 简单明确
        if len(query) <= 20 and _has_question_word(query) and session_round == 1:
            return RouterResult(
                intent=Intent.DIRECT,
                reason="首轮简短问句",
                source=RouterSource.RULE,
                confidence=0.95,
            )

        return None

    async def _model_route(self, query: str) -> RouterResult:
        """Stage 2: 调用本地模型分类"""
        if not rewrite_model_service:
            return RouterResult(
                intent=Intent.DIRECT,
                reason="rewrite model not available",
                source=RouterSource.FALLBACK,
                confidence=0.5,
            )

        try:
            result = await rewrite_model_service.classify(ROUTING_PROMPT, query)
            intent_raw = result.get("intent", "")
            reason = result.get("reason", "")

            try:
                intent = Intent(intent_raw)
            except ValueError:
                logger.warning(f"[Router] 模型返回未知意图 '{intent_raw}'，fallback direct")
                return RouterResult(
                    intent=Intent.DIRECT,
                    reason=f"unknown intent '{intent_raw}', fallback",
                    source=RouterSource.FALLBACK,
                    confidence=0.3,
                )

            # 不解析 contextualize（首轮场景不可能）
            if intent == Intent.CONTEXTUALIZE:
                logger.info("[Router] 模型判定 contextualize，但 Stage 2 不采纳（改用 direct）")
                intent = Intent.DIRECT
                reason = "模型判定 contextualize，降级为 direct"

            return RouterResult(
                intent=intent,
                reason=reason,
                source=RouterSource.MODEL,
                confidence=0.75,
            )
        except Exception as e:
            logger.warning(f"[Router] 模型分类失败: {e}，fallback direct")
            return RouterResult(
                intent=Intent.DIRECT,
                reason=f"model error: {e}",
                source=RouterSource.FALLBACK,
                confidence=0.5,
            )


def _has_question_word(text: str) -> bool:
    return bool(re.search(r"(?:怎么|如何|什么|为什么|怎样|能否|可否|是不是|有没有|能不能)", text))


def _has_decompose_marker(text: str) -> bool:
    return any(marker in text for marker in DECOMPOSE_MARKERS)


def _entity_count(text: str) -> int:
    return len(_ENTITY_PATTERN.findall(text))


def _is_overly_specific(query: str) -> bool:
    """判断查询是否过度具体（含数值/参数，缺少抽象概念）"""
    if len(query) <= 15:
        return False
    has_numeric = bool(STEP_BACK_NUMERIC.search(query))
    has_specific = bool(STEP_BACK_SPECIFIC_WORDS.search(query))
    if not (has_numeric or has_specific):
        return False
    # 不含泛化疑问词（"怎么排查"这类算有抽象方向）
    has_generic = bool(re.search(r"(?:怎么|如何|什么|为什么|方法|流程|原理|架构|策略)", query))
    return not has_generic


query_router = QueryRouter()

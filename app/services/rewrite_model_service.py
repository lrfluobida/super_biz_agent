"""本地改写模型服务 - 封装 Ollama ChatOpenAI 兼容接口"""

import json
import re
import time
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from loguru import logger

from app.config import config


class RewriteModelService:
    """本地改写模型服务，仅用于改写/分类，不做答案生成"""

    def __init__(self) -> None:
        self._model: BaseChatModel | None = None
        self._base_url = config.rewrite_local_model_url.rstrip("/")
        self._model_name = config.rewrite_local_model_name
        self._temperature = config.rewrite_local_model_temperature
        self._timeout = config.rewrite_local_model_timeout

    @property
    def model(self) -> BaseChatModel:
        if self._model is None:
            self._model = ChatOpenAI(
                model=self._model_name,
                base_url=self._base_url,
                api_key="ollama",  # Ollama 不需要真实的 key，但不能为空
                temperature=self._temperature,
                timeout=self._timeout,
                max_tokens=256,
            )
            logger.info(f"改写模型已初始化: {self._model_name} @ {self._base_url}")
        return self._model

    async def classify(self, prompt: str, query: str) -> Dict[str, Any]:
        """调用路由分类模型，返回解析后的 dict"""
        raw = await self._call_model(prompt, query)
        return self._parse_json(raw)

    async def rewrite(self, prompt: str, query: str, **extra) -> Dict[str, Any]:
        """调用改写模型（decompose/step_back/contextualize），返回解析后的 dict"""
        filled_prompt = prompt
        for key, value in extra.items():
            filled_prompt = filled_prompt.replace(f"<<{key.upper()}>>", str(value))
        raw = await self._call_model(filled_prompt, query)
        return self._parse_json(raw)

    async def _call_model(self, system_prompt: str, user_query: str) -> str:
        """调用 Ollama 模型，返回原始文本"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]
        response = await self.model.ainvoke(messages)
        content = getattr(response, "content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            content = "".join(parts)
        if not content:
            content = str(response)
        return content.strip()

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """三层 JSON 提取 + 重试

        策略：json.loads → regex {.*} → 一律失败 fallback 到空 dict
        注意：不在这里重试，重试逻辑由调用方（query_router / query_rewriter）处理
        """
        # 1. 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. 正则提取第一个 {...}
        match = re.search(r"\{[^{}]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # 3. ```json ... ``` 代码块
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        logger.warning(f"无法解析模型输出为 JSON: {text[:200]}")
        return {}

    async def health_check(self) -> bool:
        """检查 Ollama 模型是否可用"""
        try:
            t0 = time.perf_counter()
            result = await self._call_model("你是一个JSON工具", '{"test": true}')
            elapsed = time.perf_counter() - t0
            parsed = self._parse_json(result)
            ok = len(parsed) > 0
            logger.info(f"改写模型健康检查: {'OK' if ok else 'FAIL'} ({elapsed:.1f}s)")
            return ok
        except Exception as e:
            logger.warning(f"改写模型健康检查失败: {e}")
            return False


rewrite_model_service = RewriteModelService()

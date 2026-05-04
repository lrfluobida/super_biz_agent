"""RAG Agent 服务 - 基于 LangGraph 的智能代理"""

import time
from textwrap import dedent
from typing import Any, AsyncGenerator, Dict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from langchain_qwq import ChatQwen

from app.config import config
from app.services.rag_trace import begin_rag_trace, end_rag_trace, get_rag_trace_payload
from app.tools import get_current_time, retrieve_knowledge
from app.tools.knowledge_tool import format_docs, retrieve_knowledge_documents
from app.agent.mcp_client import get_mcp_client_with_retry
from app.services.memory_manager import MemoryManager


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(self, streaming: bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()
        self.eval_temperature = 0.1


        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            temperature=0.7,
            streaming=streaming,
        )

        # 定义基础工具
        self.tools = [retrieve_knowledge, get_current_time]

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []

        # 会话记忆管理器
        self.memory_manager = MemoryManager()

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(f"RAG Agent 服务初始化完成 (ChatQwen), model={self.model_name}, streaming={streaming}")

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具）"""
        if self._agent_initialized:
            return

        # 使用全局 MCP 客户端管理器（带重试拦截器）
        mcp_client = await get_mcp_client_with_retry()

        # 获取 MCP 工具
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")

        # 将 MCP 工具添加到实例变量中
        self.mcp_tools = mcp_tools

        # 合并所有工具
        all_tools = self.tools + self.mcp_tools

        self.agent = create_agent(
            self.model,
            tools=all_tools,
        )

        self._agent_initialized = True


        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """
        return dedent("""
            你是一个专业的AI助手。你必须遵守以下强制工作流程：

            【强制规则 - 必须严格遵守】
            1. 回答任何问题前，你**必须**先调用 retrieve_knowledge 工具检索知识库
            2. 即使你觉得已经知道答案，也必须先检索，因为知识库可能包含最新或更准确的信息
            3. 你**严禁**在未检索的情况下直接回答任何专业知识问题
            4. 只有检索完成后，才能基于检索结果组织回答

            【回答要求】
            - 以检索到的参考资料为核心依据
            - 如果检索结果为空或不相关，明确告知用户"未在知识库中找到相关信息"
            - 基于事实，不编造信息
            - 回答简洁明了，重点突出
        """).strip()

    def _detect_eval_question_style(self, question: str) -> str:
        """识别评测问题类型，便于使用更稳定的输出骨架。"""
        compare_keywords = ("区别", "分别", "相比", "关系", "不同")
        scenario_keywords = ("如果", "某次", "同时", "更像", "应该怎么处理", "建议怎么应对")

        if any(keyword in question for keyword in compare_keywords):
            return "compare"
        if any(keyword in question for keyword in scenario_keywords):
            return "scenario"
        return "default"

    def _build_eval_system_prompt(self, question_style: str) -> str:
        """构建评测模式专用系统提示词。"""
        base_prompt = dedent("""
            你是一个严格的 RAG 评测回答助手。

            回答要求：
            1. 只能基于提供的参考资料回答，不要补充资料之外的泛化建议
            2. 优先覆盖用户问题里的关键判断、关键对比和关键动作
            3. 尽量使用 3-6 条简洁要点回答
            4. 如果参考资料不足，明确说明“根据当前参考资料无法确定”
        """).strip()

        if question_style == "compare":
            return (
                base_prompt
                + "\n"
                + dedent("""
                    这是对照题。
                    请使用清晰的对照结构回答，必须体现 A vs B 的差异，不要只给笼统总结。
                """).strip()
            )

        if question_style == "scenario":
            return (
                base_prompt
                + "\n"
                + dedent("""
                    这是场景题。
                    请先给出判断，再说明依据，最后给出优先动作，避免泛泛而谈。
                """).strip()
            )

        return base_prompt

    def _build_eval_question(self, question: str, context: str, question_style: str) -> str:
        """将参考资料和题型约束一起打包到评测问题中。"""
        if question_style == "compare":
            answer_format = dedent("""
                请按下面结构作答：
                A vs B 对照
                - A: ...
                - B: ...
                - 处理重点: ...
            """).strip()
        elif question_style == "scenario":
            answer_format = dedent("""
                请按下面结构作答：
                - 判断：更像什么问题
                - 依据：为什么这么判断
                - 优先动作：应该先做什么
            """).strip()
        else:
            answer_format = "请用 3-6 条要点直接回答。"

        return dedent(f"""
            用户问题：
            {question}

            参考资料：
            {context if context.strip() else "没有找到相关参考资料。"}

            输出要求：
            {answer_format}
        """).strip()

    async def _ainvoke_eval_model(self, messages: list[Any]) -> Any:
        """使用更低温度执行评测回答，减少发散。"""
        original_temperature = getattr(self.model, "temperature", None)
        if original_temperature is not None:
            self.model.temperature = self.eval_temperature
        try:
            return await self.model.ainvoke(messages)
        finally:
            if original_temperature is not None:
                self.model.temperature = original_temperature

    async def _query_with_evaluation_mode(
        self,
        question: str,
        session_id: str,
        eval_top_k: int | None,
        use_hybrid: bool | None = None,
    ) -> str:
        """评测模式走确定性的先检索后生成链路。"""
        question_style = self._detect_eval_question_style(question)
        docs = retrieve_knowledge_documents(question, top_k=eval_top_k, use_hybrid=use_hybrid)
        context = format_docs(docs) if docs else "没有找到相关参考资料。"
        eval_system_prompt = self._build_eval_system_prompt(question_style)
        eval_question = self._build_eval_question(question, context, question_style)

        messages = await self.memory_manager.build_messages(
            session_id=session_id,
            system_prompt=eval_system_prompt,
            question=eval_question,
        )
        response = await self._ainvoke_eval_model(messages)
        answer = self._message_to_text(response).strip()
        await self.memory_manager.complete_turn(session_id, question, answer)
        return answer

    async def query(
        self,
        question: str,
        session_id: str,
    ) -> str:
        result = await self.query_with_evaluation(question, session_id=session_id, eval_mode=False)
        return result["answer"]

    async def query_with_evaluation(
        self,
        question: str,
        session_id: str,
        eval_mode: bool = False,
        eval_top_k: int | None = None,
        use_hybrid: bool | None = None,
    ) -> Dict[str, Any]:
        """
        非流式处理用户问题，并在评测模式开启时返回检索轨迹。

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）
            eval_mode: 是否开启评测模式
            eval_top_k: 评测模式下检索 TopK 覆盖
            use_hybrid: 评测模式下是否启用混合检索，None 使用配置值

        Returns:
            Dict[str, Any]: 包含答案和评测信息的响应
        """
        trace_token = begin_rag_trace(eval_top_k)
        started_at = time.perf_counter()
        try:
            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            if eval_mode:
                answer = await self._query_with_evaluation_mode(
                    question=question,
                    session_id=session_id,
                    eval_top_k=eval_top_k,
                    use_hybrid=use_hybrid,
                )
                logger.info(f"[会话 {session_id}] RAG Eval 查询完成（非流式）")
            else:
                await self._initialize_agent()

                messages = await self.memory_manager.build_messages(
                    session_id=session_id,
                    system_prompt=self.system_prompt,
                    question=question,
                )

                agent_input = {"messages": messages}
                config_dict = {
                    "configurable": {
                        "thread_id": session_id
                    }
                }

                result = await self.agent.ainvoke(
                    input=agent_input,
                    config=config_dict,
                )

                messages_result = result.get("messages", [])
                if messages_result:
                    last_message = messages_result[-1]
                    answer = self._message_to_text(last_message)

                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                        logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")

                    await self.memory_manager.complete_turn(session_id, question, answer)
                    logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
                else:
                    logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
                    answer = ""

            evaluation = get_rag_trace_payload()
            evaluation["latency_seconds"] = round(time.perf_counter() - started_at, 3)
            return {
                "answer": answer,
                "evaluation": evaluation,
            }

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（非流式）: {e}")
            raise
        finally:
            end_rag_trace(trace_token)

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: "content" | "tool_call" | "complete" | "error"
                - data: 具体内容
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}")

            messages = await self.memory_manager.build_messages(
                session_id=session_id,
                system_prompt=self.system_prompt,
                question=question,
            )

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            answer_chunks: list[str] = []
            async for token, metadata in self.agent.astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                message_type = type(token).__name__

                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, 'content_blocks', None)

                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_content = block.get('text', '')
                                if text_content:
                                    answer_chunks.append(text_content)
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name
                                    }

            answer = "".join(answer_chunks).strip()
            if answer:
                await self.memory_manager.complete_turn(session_id, question, answer)
            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            yield {"type": "complete"}

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（流式）: {e}")
            yield {
                "type": "error",
                "data": str(e)
            }
            raise

    def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史（从 SQLite 持久化存储中读取）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        try:
            history = self.memory_manager.get_session_history(session_id)
            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
            return history
            
        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """
        清空会话历史（从 SQLite 和内存缓存中删除）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            bool: 是否成功
        """
        try:
            success = self.memory_manager.clear_session(session_id)
            logger.info(f"已清除会话历史: {session_id}, 结果: {success}")
            return success
            
        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")

    def get_session_message_count(self, session_id: str) -> int:
        """获取完整历史消息数"""
        return self.memory_manager.get_message_count(session_id)

    def _message_to_text(self, message: Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)


# 全局单例 - 启用流式输出
rag_agent_service = RagAgentService(streaming=True)

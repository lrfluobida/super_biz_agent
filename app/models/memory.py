"""对话记忆相关数据模型"""

from pydantic import BaseModel, Field


class MemorySummary(BaseModel):
    """结构化摘要"""

    current_goal: str = ""
    important_facts: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    resolved_items: list[str] = Field(default_factory=list)
    open_items: list[str] = Field(default_factory=list)
    user_preferences: list[str] = Field(default_factory=list)

    def has_content(self) -> bool:
        """判断摘要是否为空"""
        return any(
            [
                self.current_goal.strip(),
                self.important_facts,
                self.constraints,
                self.resolved_items,
                self.open_items,
                self.user_preferences,
            ]
        )

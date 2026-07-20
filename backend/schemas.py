from langgraph.graph import add_messages
from typing import Annotated, Optional, TypedDict, Literal

from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    next_node: Optional[str]
    loop_counter: int
    active_agent: Optional[Literal["data_node", "stats_node"]]
    schema_loaded: bool
    tool_trace: list[str]
    thinking_trace: list[str] = Field(
        default_factory=list, description="Accumulated reasoning from agent steps")


class AgentResponse(BaseModel):
    thinking: str = Field(
        description="Your internal reasoning and thought process before calling any tools. Think step by step.")
    summary: str = Field(
        default=None, description="One sentence for a non-technical user.")
    interpretation: str = Field(
        default=None, description="2-4 sentences for an analyst. Reference specific features/columns used, and numeric outputs. If a tool fallback occurred, describe the attempted path vs the corrected path here.")
    statistics: dict = Field(
        default={}, description="A dictionary of relevant statistics, with keys as the statistic name and values as the statistic value.")
    insight: str = Field(
        default=None, description="One actionable insight or pattern worth noting.")
    error_guidance: str | None = Field(
        default=None, description="If there was an error in the user's request, provide guidance on how to correct it. If there was no error, this should be null.")


class RouterSchema(BaseModel):
    next_agent: Literal['data_node', 'stats_node', 'FINISH'] = Field(
        description="The next node to call. Choose FINISH ONLY if there is nothing else to do."
    )
    reasoning: str = Field(
        description="Brief justification for why this agent or action was selected next."
    )

from langgraph.graph import add_messages
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


class AgentResponse(BaseModel):
    thinking: str = Field(
        description="Your internal reasoning and thought process before calling any tools. Think step by step.")
    summary: str = Field(
        default="", description="One sentence for a non-technical user.")
    interpretation: str = Field(
        default="", description="2-4 sentences for an analyst. Reference specific features/columns used, and numeric outputs. If a tool fallback occurred, describe the attempted path vs the corrected path here.")
    statistics: dict = Field(
        default={}, description="A dictionary of relevant statistics, with keys as the statistic name and values as the statistic value.")
    insight: str = Field(
        default="", description="One actionable insight or pattern worth noting.")
    error_guidance: str | None = Field(
        default=None, description="If there was an error in the user's request, provide guidance on how to correct it. If there was no error, this should be null.")

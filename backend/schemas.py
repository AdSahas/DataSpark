from langgraph.graph import add_messages
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

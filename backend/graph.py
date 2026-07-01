import json
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from backend.schemas import AgentState
from langchain.messages import ToolMessage


from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from backend.tools import (
    get_schema,
    get_sample,
    get_column_stats,
    get_value_counts,
    detect_outliers,
    sql_query,
    classify_dataset,
    linear_regression,
    logistic_regression,
    run_pearson_correlation,
    run_spearman_correlation,
    run_chi_squared,
    fit_curves,
    detect_trends
)
from backend.guardrails import block_invalid_tool_calls, validate_output

# agent tools
TOOLS = [
    get_schema,
    get_sample,
    get_column_stats,
    get_value_counts,
    detect_outliers,
    sql_query,
    classify_dataset,
    linear_regression,
    logistic_regression,
    run_pearson_correlation,
    run_spearman_correlation,
    run_chi_squared,
    fit_curves,
    detect_trends
]

# build graph and routes


def build_graph(api_key: str, model: str = "gpt-4o"):
    base_model = ChatOpenAI(api_key=api_key, model=model, temperature=0.3,
                            streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

    llm_with_tools = base_model.bind_tools(TOOLS)

    # nodes

    def agent_node(state: AgentState):
        system = SystemMessage(content="""
                               
You are a data analyst assistant specialized in analyzing the CSV dataset the user has loaded.
Your job is to answer questions about THIS dataset specifically — not about data science in general.

Rules:
    1. ALWAYS call get_schema() first before answering anything, even if the question seems general.
    2. Every answer must reference actual column names, values, or statistics from the dataset.
    3. If the user asks something that has nothing to do with the dataset, say: "I can only answer questions about the loaded dataset."
    4. Never answer from general knowledge alone. If you need a number, use a tool to get it.
    5. Always specify which features you use during your analysis.
    6. If there is no tool that will help you answer the question, say: "I cannot answer that question with the available tools."
    
    After all tool calls are complete, you MUST respond with a JSON object in this exact format:
    {
    "summary": "One sentence for a non-technical user.",
    "interpretation": "2-3 sentences for an analyst. Reference specific numbers.",
    "statistics": { "key": value, ... },  // raw numbers from tool results
    "insight": "One actionable insight or pattern worth noting.",
    "error_guidance": null  // or a string if something went wrong
    }

Do not wrap in markdown. Return raw JSON only.
""")

        response = llm_with_tools.invoke([system] + state["messages"])

        if hasattr(response, "tool_calls") and response.tool_calls:
            for call in response.tool_calls:
                args = ", ".join(f"{k}={repr(v)}" for k,
                                 v in call["args"].items())
                print(f"  [CALL]  {call['name']}({args})")

        return {"messages": [response]}

    def tool_node_with_trace(state: AgentState):
        result = ToolNode(TOOLS).invoke(state)
        for msg in result["messages"]:
            if hasattr(msg, "content"):
                try:
                    pretty = json.dumps(json.loads(msg.content), indent=2)
                except Exception:
                    pretty = msg.content
                print(f"  [RESULT] {pretty}\n")
        return result

    # routing

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # graph

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("guardrail_tools", block_invalid_tool_calls)
    graph.add_node("tools", tool_node_with_trace)
    graph.add_node("guardrail_output", validate_output)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "guardrail_tools", END: "guardrail_output"}
    )

    graph.add_edge("guardrail_tools", "tools")
    graph.add_edge("tools", "agent")
    graph.add_edge("guardrail_output", END)

    return graph.compile()

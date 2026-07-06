from backend.tools import (
    classify_dataset,
    detect_outliers,
    detect_trends,
    fit_curves,
    get_column_stats,
    get_sample,
    get_schema,
    get_value_counts,
    linear_regression,
    logistic_regression,
    run_anova,
    run_chi_squared,
    run_chi_squared_gof,
    run_correlation,
    run_ttest_independent,
    run_ttest_onesample,
    run_ttest_paired,
    sql_query,
)
from backend.schemas import AgentResponse, AgentState
from backend.guardrails import block_invalid_tool_calls, validate_output
import json

from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()


TOOLS = [
    get_schema,
    get_sample,
    get_column_stats,
    get_value_counts,
    detect_outliers,
    classify_dataset,
    sql_query,
    run_correlation,
    run_ttest_independent,
    run_ttest_paired,
    run_ttest_onesample,
    run_anova,
    run_chi_squared,
    run_chi_squared_gof,
    linear_regression,
    logistic_regression,
    detect_trends,
    fit_curves,
]

SYSTEM_PROMPT = """
You are a data analyst assistant specialized in analyzing the CSV dataset the user has loaded.
Your job is to answer questions about THIS dataset specifically — not about data science in general.

Rules:
1. Clarification: If the user's instructions or intent are ambiguous or unclear, ask them for clarification before calling any tools.
2. Initialization: Unless you already have the schema data in the current context, ALWAYS call get_schema() first before answering anything.
3. Thinking: Before calling any tool, think through the user's request step by step and determine which tool is most appropriate. Populate the 'thinking' field with this reasoning.
4. Analysis: You will have access to column names and types from get_schema(). Consider this information carefully while choosing tools.
5. Tool Fallback: If a tool rejects your request (e.g., assumption violations like non-normality), pivot to a more compatible tool automatically. Log this in the 'interpretation' field: "Attempted [Tool A], but encountered incompatibility: [Reason]. Redirecting to [Tool B]."
6. Grounding: Every answer must reference actual column names, values, or statistics. Never answer from general knowledge alone.
7. Scope: If the user asks something unrelated to the dataset, say: "I can only answer questions about the loaded dataset."
8. Constraints: If there is no tool that will help, say: "I cannot answer that question with the available tools."

Output:
After all tool calls are complete, you MUST respond by populating the structured output schema with these fields:
- thinking: Your internal reasoning and thought process before calling any tools.
- summary: One sentence for a non-technical user.
- interpretation: 2-4 sentences for an analyst referencing specific columns, features, and numeric outputs.
- statistics: A dictionary of relevant statistics (key-value pairs).
- insight: One actionable insight or pattern worth noting.
- error_guidance: Guidance if something went wrong, otherwise null.
"""


def build_graph(model: str = "gpt-4o"):
    base_model = ChatOpenAI(
        model=model,
        temperature=0.3,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    # Bind tools for tool-calling turns
    llm_with_tools = base_model.bind_tools(TOOLS)

    # Structured output LLM for the final response only
    llm_structured = base_model.with_structured_output(
        AgentResponse, method="function_calling"
    )

    # ── Nodes ────────────────────────────────────────────────────────────────

    def agent_node(state: AgentState):
        system = SystemMessage(content=SYSTEM_PROMPT)
        messages = [system] + state["messages"]

        # Check if the last message was a tool result — if so, produce final answer
        last = state["messages"][-1] if state["messages"] else None
        last_is_tool_result = hasattr(last, "type") and last.type == "tool"

        if last_is_tool_result:
            # All tools have been called — produce the final structured response
            response = llm_structured.invoke(messages)

            if isinstance(response, AgentResponse):
                print(f"\n  [FINAL ANSWER]\n  Summary: {response.summary}\n")
                return {"messages": [AIMessage(content=response.model_dump_json())]}

            # Fallback: structured output returned something unexpected
            return {"messages": [AIMessage(content=str(response))]}

        else:
            # No tool results yet — let the model decide which tools to call
            response = llm_with_tools.invoke(messages)

            if hasattr(response, "tool_calls") and response.tool_calls:
                for call in response.tool_calls:
                    args = ", ".join(
                        f"{k}={repr(v)}" for k, v in call["args"].items()
                    )
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

    # ── Routing ───────────────────────────────────────────────────────────────

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    # ── Graph ─────────────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("guardrail_tools", block_invalid_tool_calls)
    graph.add_node("tools", tool_node_with_trace)
    graph.add_node("guardrail_output", validate_output)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "guardrail_tools", END: "guardrail_output"},
    )

    graph.add_edge("guardrail_tools", "tools")
    graph.add_edge("tools", "agent")
    graph.add_edge("guardrail_output", END)

    return graph.compile()

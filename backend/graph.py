from backend.tools import (
    get_schema,
    get_sample,
    get_column_stats,
    get_value_counts,
    detect_outliers,
    classify_dataset,
    sql_query,
    compare_two_groups,
    compare_multi_groups,
    compare_categorical_association,
    run_regression,
    analyze_trend_and_curve
)

from backend.schemas import AgentResponse, AgentState, RouterSchema
from backend.guardrails import block_invalid_tool_calls, validate_output
import json

from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from backend.prompts.supervisor_prompt_v1 import SUPERVISOR_PROMPT

load_dotenv()

STAT_TOOLS = [
    compare_two_groups,
    compare_multi_groups,
    compare_categorical_association,
    run_regression,
    analyze_trend_and_curve
]

DATA_TOOLS = [
    get_schema,
    get_sample,
    get_column_stats,
    get_value_counts,
    sql_query,
    detect_outliers,
    classify_dataset
]

ALL_TOOLS = DATA_TOOLS + STAT_TOOLS

ALLOWED_TOOLS = {
    "data_node": {tool.name for tool in DATA_TOOLS},
    "stats_node": {tool.name for tool in STAT_TOOLS},
}

DATA_NODE_PROMPT = """You are a Data Retrieval Assistant specialized in exploring datasets and pulling records.
Your only job is to use your tools to provide raw data, sample structures, schemas, or SQL results.
Do not perform advanced statistical analysis or run regressions yourself.
IMPORTANT: You MUST call at least one tool. Do not respond with plain text.
Call only the tools you need — do not call multiple tools speculatively.
"""

STATS_NODE_PROMPT = """You are a Statistics Expert specialized in mathematical calculations, hypotheses testing, and analytics.
Your only job is to run analytical models (regressions, t-tests, correlations) on the columns provided.
If a tool rejects your request due to an assumption violation, adapt by picking an alternative compatible tool.
IMPORTANT: You MUST call at least one tool. Do not respond with plain text.
Call only the tools you need — do not call multiple tools speculatively.
"""

FINAL_OUTPUT_PROMPT = """You are a data analyst summarizing findings for the user.
Format your final analysis exactly according to the structured output parameters required.
IF THE USER'S QUESTION WAS IRRELEVANT TO STATISTICAL ANALYSIS OR DATA ANALYSIS,
PROVIDE A NOTE IN "Summary" STATING THAT YOU CANNOT HELP WITH REQUESTS OUTSIDE THE SCOPE OF DATA AND STATISTICAL ANALYSIS, AND LEAVE ALL OTHER FIELDS EMPTY.

Use the thinking_trace provided to construct your response. Do not generate new thinking — use what was actually done."""

MAX_LOOPS = 10


def has_unresolved_tool_calls(messages: list) -> bool:
    """Check if any AIMessage tool_calls lack a matching ToolMessage."""
    resolved_ids = {
        msg.tool_call_id
        for msg in messages
        if isinstance(msg, ToolMessage)
    }
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", []):
                if tc["id"] not in resolved_ids:
                    return True
    return False


def sanitize_messages_for_llm(messages: list) -> list:
    """Drop AIMessages with unresolved tool_calls so OpenAI doesn't reject the request."""
    resolved_ids = {
        msg.tool_call_id
        for msg in messages
        if isinstance(msg, ToolMessage)
    }
    clean = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", []):
            unresolved = [tc for tc in msg.tool_calls if tc["id"]
                          not in resolved_ids]
            if unresolved:
                continue  # drop message entirely if any tool call is unresolved
        clean.append(msg)
    return clean


def extract_thinking_from_response(response):
    """Extract thinking/reasoning from LLM response if present."""
    if hasattr(response, "content") and response.content:
        # If the response has reasoning in content, extract it
        return str(response.content)[:200]  # First 200 chars of reasoning
    return None


def build_graph(model: str = "gpt-4o"):

    base_model = ChatOpenAI(
        model=model,
        temperature=0.2,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    # instantiate once, not on every tool call
    data_tool_node = ToolNode(DATA_TOOLS)
    stats_tool_node = ToolNode(STAT_TOOLS)

    # --- nodes ---

    def supervisor_node(state: AgentState):
        if state.get("loop_counter", 0) >= MAX_LOOPS:
            print("\n[SUPERVISOR] Loop limit reached. Routing to final output.\n")
            return {"next_node": "FINISH"}

        if has_unresolved_tool_calls(state["messages"]):
            print(
                "\n[SUPERVISOR] Unresolved tool calls detected — re-routing to active agent.\n")
            active = state.get("active_agent", "data_node")
            return {"next_node": active}

        supervisor_llm = ChatOpenAI(
            model=model, temperature=0.0
        ).with_structured_output(RouterSchema)

        clean_messages = sanitize_messages_for_llm(state["messages"])
        messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + clean_messages
        response = supervisor_llm.invoke(messages)
        print(
            f"\n[SUPERVISOR] Next: {response.next_agent} | Reason: {response.reasoning}\n")

        return {"next_node": response.next_agent}

    def data_node(state: AgentState) -> AgentState:
        current_loops = state.get("loop_counter", 0) + 1
        llm_with_tools = base_model.bind_tools(
            DATA_TOOLS)
        messages = [SystemMessage(
            content=DATA_NODE_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)

        # Extract thinking from response
        thinking = extract_thinking_from_response(response)
        thinking_trace = state.get("thinking_trace", [])
        if thinking:
            thinking_trace.append(f"[DATA_NODE] {thinking}")

        return {
            "messages": [response],
            "loop_counter": current_loops,
            "active_agent": "data_node",
            "thinking_trace": thinking_trace,
        }

    def stats_node(state: AgentState) -> AgentState:
        current_loops = state.get("loop_counter", 0) + 1
        llm_with_tools = base_model.bind_tools(
            STAT_TOOLS)
        messages = [SystemMessage(
            content=STATS_NODE_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)

        # Extract thinking from response
        thinking = extract_thinking_from_response(response)
        thinking_trace = state.get("thinking_trace", [])
        if thinking:
            thinking_trace.append(f"[STATS_NODE] {thinking}")

        return {
            "messages": [response],
            "loop_counter": current_loops,
            "active_agent": "stats_node",
            "thinking_trace": thinking_trace,
        }

    def tool_node_with_trace(state: AgentState):
        active = state.get("active_agent")
        try:
            if active == "data_node":
                result = data_tool_node.invoke(state)
            elif active == "stats_node":
                result = stats_tool_node.invoke(state)
            else:
                result = ToolNode(ALL_TOOLS).invoke(state)
        except Exception as e:
            print(f"  [TOOL ERROR] {e}")
            last_msg = state["messages"][-1]
            executed_tools = [tc["name"] for tc in getattr(
                last_msg, "tool_calls", []) if "name" in tc]
            error_msgs = [
                ToolMessage(
                    content=f"Tool execution failed: {str(e)}",
                    tool_call_id=tc["id"]
                )
                for tc in getattr(last_msg, "tool_calls", [])
            ]
            return {
                "messages": error_msgs,
                "tool_trace": state.get("tool_trace", []) + executed_tools,
                "thinking_trace": state.get("thinking_trace", []),
            }

        messages = result["messages"] if isinstance(result, dict) else result

        last_msg = state["messages"][-1]
        calls_map = {
            tc["id"]: {"name": tc["name"], "args": tc["args"]}
            for tc in getattr(last_msg, "tool_calls", []) if "id" in tc
        }

        for msg in messages:
            if msg.type == "tool":
                tool_id = getattr(msg, "tool_call_id", None)
                meta = calls_map.get(
                    tool_id, {"name": "Unknown Tool", "args": {}})
                params_str = json.dumps(
                    meta["args"], indent=2).replace("\n", "\n  ")
                print(
                    f"  [TOOL CALL]   Running '{meta['name']}' with params:\n  {params_str}")
                if hasattr(msg, "content"):
                    try:
                        pretty = json.dumps(json.loads(msg.content), indent=2)
                    except Exception:
                        pretty = msg.content
                    print(f"  [TOOL RESULT] {pretty}\n")

        executed_tools = list(calls_map[tid]["name"] for tid in calls_map)

        return {
            "messages": messages,
            "tool_trace": state.get("tool_trace", []) + executed_tools,
            "thinking_trace": state.get("thinking_trace", []),
        }

    def final_output_node(state: AgentState):
        llm_structured = base_model.with_structured_output(
            AgentResponse, method="function_calling"
        )
        # Construct thinking from accumulated trace
        accumulated_thinking = "\n".join(state.get("thinking_trace", []))

        # sanitize before passing to LLM
        clean_messages = sanitize_messages_for_llm(state["messages"])

        # Add thinking context to prompt
        prompt_with_thinking = FINAL_OUTPUT_PROMPT + \
            f"\n\nAccumulated reasoning trace:\n{accumulated_thinking}"
        messages = [SystemMessage(
            content=prompt_with_thinking)] + clean_messages
        response = llm_structured.invoke(messages)

        if isinstance(response, AgentResponse):
            # Override thinking with accumulated trace
            response.thinking = accumulated_thinking[:
                                                     500] if accumulated_thinking else "Analysis complete."

            print(f"\n[FINAL OUTPUT]")
            print(f"  Thinking:       {response.thinking}")
            print(f"  Summary:        {response.summary}")
            print(f"  Interpretation: {response.interpretation}")
            print(f"  Statistics:     {response.statistics}")
            print(f"  Insight:        {response.insight}")

        return {"messages": [AIMessage(content=response.model_dump_json())]}

    # --- routing ---

    def route_from_supervisor(state: AgentState) -> str:
        next_node = state.get("next_node")
        if next_node == "FINISH":
            return "final_output"
        elif next_node == "data_node":
            return "data_node"
        elif next_node == "stats_node":
            return "stats_node"
        else:
            return "final_output"

    def route_from_specialist(state: AgentState) -> str:
        if state.get("loop_counter", 0) >= MAX_LOOPS:
            return "supervisor"
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "supervisor"

    def route_after_tools(state: AgentState) -> str:
        active = state.get("active_agent")
        if active in {"data_node", "stats_node"}:
            return active
        return "supervisor"

    # --- graph assembly ---

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("data_node", data_node)
    graph.add_node("stats_node", stats_node)
    graph.add_node("tools", tool_node_with_trace)
    graph.add_node("final_output", final_output_node)
    graph.add_node("guardrail_output", validate_output)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "data_node": "data_node",
            "stats_node": "stats_node",
            "final_output": "final_output",
        }
    )
    graph.add_conditional_edges(
        "data_node",
        route_from_specialist,
        {"tools": "tools", "supervisor": "supervisor"}
    )
    graph.add_conditional_edges(
        "stats_node",
        route_from_specialist,
        {"tools": "tools", "supervisor": "supervisor"}
    )

    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "data_node": "data_node",
            "stats_node": "stats_node",
            "supervisor": "supervisor",
        }
    )

    graph.add_edge("final_output", "guardrail_output")
    graph.add_edge("guardrail_output", END)

    return graph.compile()

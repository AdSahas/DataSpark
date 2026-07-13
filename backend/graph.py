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
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv


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

# prompts
SUPERVISOR_PROMPT = """You are the Lead Data Scientist Supervisor managing two specialized units:
1. data_fetcher: Best for querying databases, checking schemas, exploring basic column stats, and pulling raw data samples.
2. statistical_analyst: Best for advanced mathematical tests (ANOVA, T-Tests, Regressions, Outlier/Trend detection).

Your job is to evaluate the conversation history and dynamically assign the next step:
- ALWAYS ensure the data_fetcher calls get_schema() first if schema context is missing.
- If data needs to be queried/fetched before advanced analysis can occur, route to data_fetcher.
- CRITICAL: If the user's request is purely informational (e.g., "describe the data", "show me the schema", "what columns do we have?") and the data_fetcher has already pulled the schema/samples, do NOT route to statistical_analyst. Route directly to FINISH so the final summary can be built.
- Once enough data, queries, and statistical calculations have run to fully answer the user, select FINISH.
- IF THE USER'S QUESTION IS IRRELEVANT TO STATISTICAL ANALYSIS OR DATA ANALYSIS, ROUTE DIRECTLY TO FINISH AND STATE THAT THE REQUEST WAS OUTSIDE THE SCOPE OF DATA AND STATISTICAL ANALYSIS.
"""

DATA_NODE_PROMPT = """You are a Data Retrieval Assistant specialized in exploring datasets and pulling records.
Your only job is to use your tools to provide raw data, sample structures, schemas, or SQL results.
Do not perform advanced statistical analysis or run regressions yourself.
"""

STATS_NODE_PROMPT = """You are a Statistics Expert specialized in mathematical calculations, hypotheses testing, and analytics.
Your only job is to run analytical models(regressions, t-tests, correlations) on the columns provided.
If a tool rejects your request due to an assumption violation, adapt by picking an alternative compatible tool.
"""

FINAL_OUTPUT_PROMPT = """You are a data analyst summarizing findings for the user.
Format your final analysis exactly according to the structured output parameters required.
IF THE USER'S QUESTION WAS IRRELEVANT TO STATISTICAL ANALYSIS OR DATA ANALYSIS, 
PROVIDE A NOTE IN "Summary" STATING THAT YOU CANNOT HELP WITH REQUESTS OUTSIDE THE SCOPE OF DATA AND STATISTICAL ANALYSIS, AND LEAVE ALL OTHER FIELDS EMPTY.

"""

# graph builder


def build_graph(model: str = "gpt-4o"):

    # base configuration for standard LLM actions
    base_model = ChatOpenAI(
        model=model,
        temperature=0.2,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    # nodes

    def supervisor_node(state: AgentState):
        if state.get("loop_counter", 0) >= 6:
            print(
                "\n[SUPERVISOR] Loop limit reached. Routing to final output.\n")
            return {"next": "FINISH"}

        # enforce a low temperature structured output to prevent routing hallucination
        supervisor_llm = ChatOpenAI(
            model=model, temperature=0.0).with_structured_output(RouterSchema)
        messages = [SystemMessage(
            content=SUPERVISOR_PROMPT)] + state["messages"]

        response = supervisor_llm.invoke(messages)
        print(
            f"\n[SUPERVISOR] Next: {response.next_agent} | Reason: {response.reasoning}\n")

        return {"next": response.next_agent}

    def data_node(state: AgentState) -> AgentState:
        current_loops = state.get("loop_counter", 0) + 1

        llm_with_tools = base_model.bind_tools(DATA_TOOLS)
        messages = [SystemMessage(
            content=DATA_NODE_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "loop_counter": current_loops}

    def stats_node(state: AgentState) -> AgentState:
        current_loops = state.get("loop_counter", 0) + 1

        llm_with_tools = base_model.bind_tools(STAT_TOOLS)
        messages = [SystemMessage(
            content=STATS_NODE_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "loop_counter": current_loops}

    def tool_node_with_trace(state: AgentState):

        result = ToolNode(ALL_TOOLS).invoke(state)
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", [])

        # create a lookup map of tool_call_id -> tool details for matching
        calls_map = {
            tc["id"]: {"name": tc["name"], "args": tc["args"]}
            for tc in tool_calls if "id" in tc
        }

        for msg in result["messages"]:
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

        return result

    def final_output_node(state: AgentState):
        llm_structured = base_model.with_structured_output(
            AgentResponse, method="function_calling"
        )
        messages = [SystemMessage(
            content=FINAL_OUTPUT_PROMPT)] + state["messages"]
        response = llm_structured.invoke(messages)

        if isinstance(response, AgentResponse):
            print(f"\n[FINAL OUTPUT]")
            print(f"  Thinking: {response.thinking}")
            print(f"  Summary: {response.summary}")
            print(f"  Interpretation: {response.interpretation}")
            print(f"  Statistics: {response.statistics}")
            print(f"  Insight: {response.insight}")

        return {"messages": [AIMessage(content=str(response))]}

    # assembly of graph
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("data_node", data_node)
    graph.add_node("stats_node", stats_node)
    graph.add_node("guardrail_tools", block_invalid_tool_calls)
    graph.add_node("tools", tool_node_with_trace)
    graph.add_node("final_output", final_output_node)
    graph.add_node("guardrail_output", validate_output)

    def route_from_supervisor(state: AgentState) -> str:
        if state["next"] == "FINISH":
            return "final_output"

    def route_from_specialist(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "supervisor"

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "data_node": "data_node",
            "stats_node": "stats_node",
            "final_output": "final_output"
        }
    )
    graph.add_conditional_edges(
        "data_node",
        route_from_specialist,
        {"tools": "guardrail_tools", "supervisor": "supervisor"}
    )
    graph.add_conditional_edges(
        "stats_node",
        route_from_specialist,
        {"tools": "guardrail_tools", "supervisor": "supervisor"}
    )
    graph.add_edge("guardrail_tools", "tools")
    graph.add_edge("tools", "supervisor")
    graph.add_edge("final_output", "guardrail_output")
    graph.add_edge("guardrail_output", END)

    return graph.compile()

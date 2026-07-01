# guardrails.py
import re
import json
from langchain_core.messages import ToolMessage
from backend.tools import _df

# ── Tool guardrails ───────────────────────────────────────────────────

COLUMN_TOOLS = {
    "get_column_stats",
    "get_value_counts",
    "detect_outliers",
    "suggest_ml_models",
    "get_ml_profile",
}

NUMERIC_ONLY_TOOLS = {
    "detect_outliers",
    "compute_correlation",
}

SQL_DETECTOR_REGEX = r"(?i)\b(select|insert|update|delete|drop|alter|create)\b"


def validate_tool_calls(tool_calls: list) -> list[str | None]:
    """
    Validates each tool call before execution.
    Returns a list of error strings (or None if valid) — one per tool call.
    """
    errors = []
    for call in tool_calls:
        name = call["name"]
        args = call["args"]
        error = None

        # Check column exists
        if name in COLUMN_TOOLS:
            col = args.get("column") or args.get("target_column")
            if col and _df is not None and col not in _df.columns:
                error = (
                    f"Column '{col}' does not exist. "
                    f"Available columns: {list(_df.columns)}"
                )

        # Check column is numeric for numeric-only tools
        if name in NUMERIC_ONLY_TOOLS and error is None:
            col = args.get("column") or args.get("col_a")
            if col and _df is not None and col in _df.columns:
                import pandas as pd
                if not pd.api.types.is_numeric_dtype(_df[col]):
                    error = (
                        f"Column '{col}' is not numeric. "
                        f"'{name}' only works on numeric columns."
                    )

        # Check filter operator is valid
        if name == "filter_rows":
            op = args.get("operator")
            valid_ops = {">", "<", ">=", "<=", "=="}
            if op and op not in valid_ops:
                error = f"Invalid operator '{op}'. Must be one of: {valid_ops}"

        errors.append(error)
    return errors


def block_invalid_tool_calls(state: dict) -> dict:
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return state

    errors = validate_tool_calls(last_message.tool_calls)
    injected = []
    valid_tool_calls = []

    for call, error in zip(last_message.tool_calls, errors):
        if error:
            print(f"  [GUARDRAIL] Blocked {call['name']}: {error}")
            injected.append(
                ToolMessage(
                    content=json.dumps({"error": error}),
                    tool_call_id=call["id"],
                )
            )
        else:
            valid_tool_calls.append(call)

    if injected:
        # Rebuild the last message with only valid tool calls
        # so OpenAI doesn't see orphaned ToolMessages
        from langchain_core.messages import AIMessage
        cleaned_message = AIMessage(
            content=last_message.content,
            tool_calls=valid_tool_calls,
        )
        return {"messages": [cleaned_message] + injected}

    return state


# ── Output guardrails ─────────────────────────────────────────────────
VAGUE_PATTERNS = [
    r"i (don't|do not|cannot|can't) (access|see|find|read) (the )?(data|dataset|file|csv)",
    r"as an ai",
    r"i don't have (access to|information about) (the )?actual",
    r"without (access to |seeing )?(the )?data",
]


def validate_output(state: dict) -> dict:
    """
    LangGraph node that runs after the agent produces a final answer.
    Flags responses that are vague, hallucinatory, or not grounded in the data.
    """
    last_message = state["messages"][-1]
    content = last_message.content or ""

    for pattern in VAGUE_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            print(
                f"  [GUARDRAIL] Output flagged — agent claimed it can't access data")
            # Inject a corrective message back into the loop
            from langchain_core.messages import HumanMessage
            return {
                "messages": [
                    HumanMessage(
                        content=(
                            "You do have access to the dataset via your tools. "
                            "Please call get_schema() and try again."
                        )
                    )
                ]
            }

    return state

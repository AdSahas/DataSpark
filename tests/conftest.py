import json
import pytest

from langchain_core.messages import HumanMessage
from backend.graph import build_graph
from backend.tools import (
    get_schema,
    choose_analysis,
    run_correlation,
)


@pytest.fixture(scope="module")
def graph():
    return build_graph(
        model="gpt-4o-mini"
    )


def run_agent(graph, question):
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=question
                )
            ],
            "dataset_context": {},
            "tool_history": [],
        }
    )

    return result


def get_tool_history(result):
    return result.get(
        "tool_history",
        []
    )


def test_schema_request_routes_to_data_agent(graph):
    result = run_agent(
        graph,
        "Show me the columns in this dataset"
    )

    history = get_tool_history(result)
    assert (
        "get_schema" in history
    )


def test_statistical_question_routes_to_stat_agent(graph):
    result = run_agent(
        graph,
        """
Is mean_area related to diagnosis?
Determine the correct statistical test.
"""
    )

    history = get_tool_history(result)

    assert (
        "choose_analysis"
        in history
    )

    assert (
        "run_correlation"
        in history
        or
        "run_chi_squared"
        in history
    )


def test_analysis_recommendation_before_test(graph):

    result = run_agent(
        graph,
        """
Check whether mean_area predicts diagnosis.
"""
    )

    history = get_tool_history(result)

    choose_index = history.index(
        "choose_analysis"
    )

    test_index = min(
        [
            history.index(x)
            for x in [
                "run_correlation",
                "run_ttest_independent",
                "run_anova",
                "run_chi_squared"
            ]
            if x in history
        ]
    )

    assert (
        choose_index < test_index
    )


def test_no_invalid_tool_calls(graph):

    result = run_agent(
        graph,
        """
Analyze the relationship between mean_area and diagnosis.
"""
    )

    history = get_tool_history(result)

    allowed = {
        "choose_analysis",
        "run_correlation",
        "run_chi_squared",
        "run_ttest_independent"
    }

    invalid = (
        set(history)
        -
        allowed
    )

    assert (
        not invalid
    )


def test_schema_contains_expected_columns(graph):

    result = run_agent(
        graph,
        """
What columns are available?
"""
    )

    messages = result["messages"]

    tool_messages = [
        m
        for m in messages
        if getattr(m, "type", None)
        == "tool"
    ]

    assert len(
        tool_messages
    ) > 0

    schema = json.loads(
        tool_messages[0].content
    )

    columns = schema["columns"]

    assert (
        "mean_area"
        in columns
    )

    assert (
        "diagnosis"
        in columns
    )


def test_final_response_exists(graph):

    result = run_agent(
        graph,
        """
Analyze mean_area versus diagnosis.
"""
    )

    last = result["messages"][-1]

    assert (
        last.type == "ai"
    )

    assert (
        len(last.content)
        > 0
    )

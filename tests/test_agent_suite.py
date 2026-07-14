import time
import json
import pytest
from langchain_core.messages import HumanMessage, AIMessage


class TestDataAnalystGraph:

    def test_supervisor_routes_to_data_node_for_exploration(self, graph, initial_state):
        """Check if supervisor correctly routes exploration intent to data_node."""
        initial_state["messages"] = [HumanMessage(
            content="What are the columns in this dataset?")]

        # Invoke the full graph and check the routing decision in state
        result = graph.invoke(initial_state)

        # next_node is set by supervisor before handing off — check it was data_node at some point
        # We verify indirectly: data tools should have been used
        tools_used = result.get("tool_trace", [])
        data_tools = {"get_schema", "classify_dataset",
                      "sql_query", "get_column_stats"}
        assert any(t in data_tools for t in tools_used), (
            f"Expected a data tool to be called, got: {tools_used}"
        )

    def test_data_node_prefers_schema_over_sql(self, graph, initial_state):
        """Verify the model calls get_schema when asked about column names/types."""
        initial_state["messages"] = [HumanMessage(
            content="What are the column names and data types?")]

        result = graph.invoke(initial_state)

        tools_used = result.get("tool_trace", [])
        # get_schema OR classify_dataset are both valid first-step schema tools
        schema_tools = {"get_schema", "classify_dataset"}
        assert any(t in schema_tools for t in tools_used), (
            f"Expected a schema tool, got: {tools_used}"
        )
        assert "sql_query" not in tools_used, (
            f"sql_query should not be called for a simple schema question, got: {tools_used}"
        )

    def test_stats_node_handles_categorical_association(self, graph, initial_state):
        """Verify relationship questions trigger the correct statistics tool."""
        initial_state["messages"] = [HumanMessage(
            content="Is survival rate associated with Sex?")]
        initial_state["active_agent"] = "stats_node"

        result = graph.invoke(initial_state)

        tools_used = result.get("tool_trace", [])
        assert "compare_categorical_association" in tools_used

    def test_full_chain_regression(self, graph, initial_state):
        """Integration test: From user queston to final structured output."""
        question = "Predict Fare using Age as a predictor. Give me the regression results."
        initial_state["messages"] = [HumanMessage(content=question)]

        result = graph.invoke(initial_state)

        # Verify result contains the Final Output structure
        last_msg = result["messages"][-1]
        import json
        data = json.loads(last_msg.content)

        assert "thinking" in data
        assert "summary" in data
        assert "statistics" in data
        assert "run_regression" in result.get("tool_trace", [])

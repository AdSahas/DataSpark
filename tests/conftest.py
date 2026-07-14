import pytest
import os
from backend.csv_ingestion import load_csv
from backend.tools import set_dataframe
from backend.graph import build_graph
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    # Make sure you have a small titanic.csv in a tests/fixtures folder
    csv_path = "tests/fixtures/titanic.csv"
    if not os.path.exists(csv_path):
        pytest.fail(f"Test data missing at {csv_path}")
    df = load_csv(csv_path)
    set_dataframe(df)


@pytest.fixture
def graph():
    return build_graph(model="gpt-4o")


@pytest.fixture
def initial_state():
    return {
        "messages": [],
        "next_node": None,
        "loop_counter": 0,
        "active_agent": None,
        "schema_loaded": False,
        "tool_trace": []  # Requires the tool_trace update in graph.py
    }

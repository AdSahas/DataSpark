import os
from langchain_core.messages import HumanMessage
from backend.csv_ingestion import load_csv
from backend.tools import set_dataframe
from backend.graph import build_graph
from dotenv import load_dotenv

load_dotenv()


def run():
    csv_path = input("CSV file path: ").strip()

    print("Loading CSV...")
    df = load_csv(csv_path)
    set_dataframe(df)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.\n")

    agent = build_graph()
    conversation_history = []
    print("Data analyst ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        conversation_history.append(HumanMessage(content=user_input))
        print("\n--- agent trace ---")

        result = agent.invoke({"messages": conversation_history})

        conversation_history = result["messages"]
        print("\n")


if __name__ == "__main__":
    run()

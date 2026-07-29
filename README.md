# DataSpark

An AI-powered statistical analysis engine. Ask questions about your data in plain English — DataSpark routes your query through a multi-agent system, runs the right statistical tools, and returns a structured, interpretable result.

**Live demo:** [dataspark-demo.onrender.com/app](https://dataspark-demo.onrender.com/app)

---

## How It Works

DataSpark uses a supervisor-agent architecture built on LangGraph. When you submit a query:

1. A **Supervisor** reads your question and decides whether it requires data retrieval, statistical analysis, or both.
2. It routes to a **Data Agent** (schema exploration, sampling, SQL queries, outlier detection) or a **Statistics Agent** (regression, hypothesis testing, group comparisons, trend analysis).
3. Each agent calls only the tools it needs. If a tool fails or returns an incompatible result, the agent self-heals and retries with an alternative.
4. A **Final Output Node** assembles the accumulated reasoning trace into a structured response: summary, interpretation, statistics, and insight.
5. A **Guardrail Layer** validates the output before it reaches the user.

```
User Query
    │
    ▼
Supervisor
    ├──► Data Agent ──► Tools ──┐
    │                           │
    └──► Stats Agent ──► Tools ─┤
                                │
                          Supervisor (loop)
                                │
                          Final Output
                                │
                          Guardrail
                                │
                          Structured Response
```

---

## Capabilities

**Data tools**
- Schema inspection and column statistics
- Random sampling and value counts
- SQL querying over uploaded datasets
- Outlier detection and dataset classification

**Statistical tools**
- Two-group comparison (t-test / Mann-Whitney)
- Multi-group comparison (ANOVA / Kruskal-Wallis)
- Categorical association (chi-square)
- Regression analysis (linear / logistic)
- Trend and curve analysis

---

## Tech Stack

- **Orchestration:** LangGraph
- **LLM:** OpenAI GPT-4o
- **Backend:** FastAPI, Python
- **Statistical computation:** SciPy, scikit-learn
- **Frontend:** Static HTML served by FastAPI

---

## Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
git clone https://github.com/AdSahas/DataSpark.git
cd DataSpark
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_key_here
```

### Run

```bash
uvicorn main:app --reload
```

Then open [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app) in your browser.

---

## Example Queries

- "Is there a significant difference in sales between regions A and B?"
- "Show me the distribution of age in this dataset and flag outliers."
- "Run a linear regression predicting revenue from ad spend."
- "Is there an association between product category and return rate?"

---

## Project Structure

```
DataSpark/
├── backend/
│   ├── graph.py          # LangGraph agent graph
│   ├── tools.py          # Statistical and data tools
│   ├── schemas.py        # Pydantic models (AgentState, AgentResponse, RouterSchema)
│   ├── guardrails.py     # Output validation
│   └── prompts/          # System prompt for supervisor
├── frontend/
│   ├── app.html          # Main entrypoint
│   └── home.html         # Guide / landing page
├── README.md             # This document
├── tests/                # Tests for debugging
└── main.py               # FastAPI server
```

---

## License

MIT

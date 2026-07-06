# DataSpark: Statistical Analysis & ML Toolkit

A production-grade Python framework for **statistical analysis and data profiling** built on Pandas, Statsmodels, SciPy, and DuckDB.

It provides a unified, validated execution layer for running statistical tests, regression models, SQL queries, and dataset intelligence operations — designed for structured data and AI-driven analytics systems.

---

## ✨ Key Capabilities

### 📈 Statistical Modeling
- Linear Regression (OLS)
- Logistic Regression (Logit)
- Pearson & Spearman Correlation
- Chi-Squared Test of Independence
- Mann-Kendall Trend Detection
- Polynomial Curve Fitting with R² comparison

### 🧠 Data Intelligence
- Automatic schema inference
- Column profiling (numeric, categorical, datetime detection)
- Value distribution analysis
- Dataset classification and structure reporting
- Outlier detection using statistical thresholds

### ⚙️ Data Processing Pipeline
Input → Validate → Transform → Execute → Normalize → Return

- Input validation ensures type safety and schema correctness  
- Transformation layer handles encoding and feature preparation  
- Execution layer runs statistical or SQL operations  
- Normalization ensures consistent numeric scaling  
- Output is standardized into a unified response format  

---

## 🏗️ Architecture Overview

```txt
┌──────────────────────────────┐
│        User Input            │
└─────────────┬────────────────┘
              ↓
┌──────────────────────────────┐
│     Validation Layer         │
└─────────────┬────────────────┘
              ↓
┌──────────────────────────────┐
│   Transformation Layer       │
└─────────────┬────────────────┘
              ↓
┌──────────────────────────────┐
│    Execution Engine          │
└─────────────┬────────────────┘
              ↓
┌──────────────────────────────┐
│   Response Normalization     │
└──────────────────────────────┘
```
---

## 🔧 Core Features

### 🧮 Regression Analysis
- Multiple linear regression with diagnostics
- Logistic regression with class balance checks
- Coefficient extraction and model evaluation

### 🔬 Hypothesis Testing
- Correlation significance testing
- Chi-squared independence testing
- Trend detection using non-parametric methods

### 📊 Data Utilities
- Column-wise statistics computation
- Frequency distributions (value counts)
- Outlier detection using mean ± 2σ rule
- Dataset sampling and profiling

### 💾 SQL Integration
- DuckDB-powered in-memory SQL execution
- Query datasets using standard SQL syntax
- Seamless bridge between relational and analytical workflows

---

## 🧩 Design Principles

- Deterministic execution pipeline for reproducible outputs  
- Strict validation before computation  
- Unified result schema for downstream ML/AI consumption  
- Hybrid analytics engine combining SQL + Python stack  
- Safe preprocessing layer with encoding and normalization  

---

## 📦 Stack

Python, Pandas, NumPy, SciPy, Statsmodels, DuckDB, PyMannKendall, LangChain Tools, FastAPI

---

## 📤 Output Schema

{
  "status": "ok | error",
  "data": {},
  "message": "",
  "diagnostics": {}
}

---

## ⚠️ Limitations

- Structured tabular data only  
- Not distributed / big data optimized  
- No causal inference guarantees  
- Requires preloaded dataset in memory  

---

## 🚀 Use Cases

- Exploratory Data Analysis (EDA)
- Statistical modeling
- Feature relationship discovery
- Time-series trend detection
- SQL-based analytics

---

## 🔧 How to use
- Create a .env and add an OpenAI API key: OPENAI_API_KEY={your_key_here}
- Run: ```txt
  uvicorn main:app```
- Open localhost: http://127.0.0.1:8000

## 🚀🚀 Future Work

- Expand safe kernel to incorporate Machine Learning workflows

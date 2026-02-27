# Agentic AI — LangGraph & Azure OpenAI

Agentic workflow that autonomously selects the best machine learning model using LangGraph and GPT-4o-mini.

## Tech Stack

Python, LangGraph, LangChain, Azure OpenAI (GPT-4o-mini), scikit-learn, pandas, NumPy

## What's in this project

**Data Cleaning & Feature Engineering** — dropping rows with missing critical columns, outlier filtering, trip duration derived from timestamps, pickup hour and day of week extracted from datetime.

**Model Training** — two models trained and evaluated on validation set: Linear Regression and Neural Network, both wrapped in a sklearn Pipeline with StandardScaler.

**LangGraph Workflow** — four-node graph: train Linear Regression → train Neural Network → LLM agent decides → final test. Each node updates a shared typed state.

**LLM Agent Decision** — GPT-4o-mini receives validation RMSE results for both models and autonomously decides which model to use for final testing, returning a structured JSON decision with a technical explanation.

**Final Training & Testing** — winning model is retrained on the full train set and evaluated on the held-out test set.

## Results

- LR RMSE (val): 4.01
- NN RMSE (val): 3.25
- Agent decision: Neural Network
- Test RMSE: 3.37

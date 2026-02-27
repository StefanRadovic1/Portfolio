This notebook implements an Agentic AI workflow using LangGraph and Azure OpenAI (GPT-4o-mini) to automate machine learning model selection.

Two models are trained and evaluated on NYC taxi fare prediction data:
- Linear Regression
- Neural Network (MLPRegressor)

An LLM agent analyzes validation RMSE results and autonomously decides which model to use for final testing, providing a technical explanation for its decision.

Technologies: Python, LangGraph, LangChain, Azure OpenAI, scikit-learn, pandas, NumPy

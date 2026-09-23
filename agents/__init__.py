"""
InsightForge Agentic AI Package
Contains all specialized agents:
- DataIngestionAgent (Agent 1)
- DataCleaningAgent (Agent 2)
- AnalyticsAgent (Agent 3)
- ForecastingAgent (Agent 4)
- RecommendationAgent (Agent 5)
"""

from agents.base_agent import BaseAgent, AgentResult
from agents.ingestion import DataIngestionAgent
from agents.cleaning import DataCleaningAgent
from agents.analytics import AnalyticsAgent
from agents.forecasting import ForecastingAgent
from agents.recommendation import RecommendationAgent

__all__ = [
    "BaseAgent",
    "AgentResult",
    "DataIngestionAgent",
    "DataCleaningAgent",
    "AnalyticsAgent",
    "ForecastingAgent",
    "RecommendationAgent"
]

"""
Explainable AI (XAI) and Natural Language Explanation Modules.
"""
from .explainability import XAIEngine, StockExplanation
from .report_generator import generate_investor_report

__all__ = ["XAIEngine", "StockExplanation", "generate_investor_report"]

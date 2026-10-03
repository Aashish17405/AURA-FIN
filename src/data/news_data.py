"""
Financial News and Sentiment Data Ingestion.
Provides structured financial documents with sentiment tags, SEC filing summaries,
and contextual market catalysts for the RAG pipeline.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import datetime
import random
import re

@dataclass
class FinancialDocument:
    doc_id: str
    ticker: str
    date: str
    headline: str
    content: str
    source: str
    sentiment_label: str  # "BULLISH", "BEARISH", "NEUTRAL"
    sentiment_score: float # between -1.0 and 1.0


SAMPLE_FINANCIAL_HEADLINES = {
    "AAPL": [
        ("Apple Unveils New AI Features in Latest Hardware Refresh", "Strong consumer demand and services growth fuel quarterly guidance.", "BULLISH", 0.75),
        ("Supply Chain Bottlenecks Impact Global Smartphone Shipments", "Component lead times increase in Asian manufacturing centers.", "BEARISH", -0.45),
        ("Apple Services Revenue Surges to All-Time High", "App Store, iCloud, and payment monetization accelerate double-digit growth.", "BULLISH", 0.82),
        ("Regulatory Antitrust Scrutiny Mounts in European Markets", "App store commission structures challenged by European Commission.", "BEARISH", -0.55),
    ],
    "MSFT": [
        ("Microsoft Cloud and Azure Revenue Surpass Wall Street Expectations", "Massive enterprise AI workload adoption drives 29% cloud revenue surge.", "BULLISH", 0.88),
        ("Microsoft Expands Enterprise AI Copilot Across Global Fortune 500", "Productivity suite software monetization showing early commercial acceleration.", "BULLISH", 0.70),
        ("Capital Expenditure Increases as Data Center Expansion Continues", "Higher cloud infrastructure depreciation may pressure near-term margins.", "NEUTRAL", -0.10),
    ],
    "NVDA": [
        ("Nvidia Announces Next-Generation Blackwell Ultra AI Superchips", "Data center demand remains insatiable with multi-quarter order backlogs.", "BULLISH", 0.95),
        ("Export Restrictions on Advanced AI Accelerators Expanded", "US regulators tighten computational density rules for international shipments.", "BEARISH", -0.60),
        ("Gross Margins Expand to Record 78% on Strong H100 GPU Shipments", "Pricing power in generative AI hardware provides historic cash flow generation.", "BULLISH", 0.91),
    ],
    "GOOGL": [
        ("Alphabet Cloud Achieves Sustained Operating Profitability", "Google Cloud customer growth and Gemini enterprise integrations gain market share.", "BULLISH", 0.78),
        ("DOJ Antitrust Trial Findings Scrutinize Search Distribution Agreements", "Potential structural remedies introduce legal overhang on default browser contracts.", "BEARISH", -0.65),
    ],
    "AMZN": [
        ("AWS Accelerates Growth Driven by Enterprise Generative AI Workloads", "Cloud operating income jumps 34% with cost optimization cycle ending.", "BULLISH", 0.84),
        ("Retail Fulfillment Efficiencies Lower Cost-to-Serve Across Regional Hubs", "Automation and robotics investments in distribution network improve free cash flow.", "BULLISH", 0.62),
    ],
    "JPM": [
        ("JPMorgan Reports Record Net Interest Income Amid High Rate Environment", "Strong consumer banking deposits and trading desks outpace consensus estimates.", "BULLISH", 0.72),
        ("Credit Loss Provisions Increased in Commercial Real Estate Portfolio", "Management adopts conservative reserves for office property loan exposures.", "BEARISH", -0.38),
    ],
    "BAC": [
        ("Bank of America Digital Banking Engagement Hits New Milestones", "Mobile active users cross 40 million with robust investment banking fee rebound.", "BULLISH", 0.58),
        ("Unrealized Bond Portfolio Losses Continue to Narrow as Yields Stabilize", "Hold-to-maturity securities portfolio pressure recedes.", "NEUTRAL", 0.15),
    ],
    "GS": [
        ("Goldman Sachs M&A Advisory Fees Surge on Global Dealmaking Rebound", "Global financing markets reopen, driving strong investment banking performance.", "BULLISH", 0.76),
        ("Asset & Wealth Management Inflows Reach Record High", "Alternative asset fundraising milestones reached ahead of multi-year targets.", "BULLISH", 0.65),
    ],
    "META": [
        ("Meta Ad Impressions and Pricing Rebound Sharply with Advantage+ AI", "Digital advertising monetization re-accelerates across Instagram and WhatsApp.", "BULLISH", 0.81),
        ("Reality Labs Operating Losses Remain High Amid Metaverse R&D", "Capital intensity remains elevated on next-generation spatial computing hardware.", "NEUTRAL", -0.20),
    ],
    "TSLA": [
        ("Tesla Full Self-Driving Version 12 Rollout Receives Positive Reviews", "End-to-end neural network architecture improves autonomous driving performance.", "BULLISH", 0.68),
        ("Price Cuts in Key International Markets Compress Automotive Margins", "Competitive pressures from domestic EV manufacturers create near-term price competition.", "BEARISH", -0.58),
    ]
}


class FinancialNewsLoader:
    """
    Manages loading, filtering, and preparing financial news documents
    for vector indexing and FinLLM reasoning.
    """
    def __init__(self, tickers: Optional[List[str]] = None):
        self.tickers = tickers or list(SAMPLE_FINANCIAL_HEADLINES.keys())

    def get_documents_for_ticker(self, ticker: str, date: Optional[str] = None) -> List[FinancialDocument]:
        """
        Retrieves relevant financial documents for a ticker with defensive fallback.
        """
        docs: List[FinancialDocument] = []
        raw_items = SAMPLE_FINANCIAL_HEADLINES.get(ticker, [])
        
        # If no specific headlines for ticker, create dynamic ones
        if not raw_items:
            raw_items = [
                (f"{ticker} Reports Stable Quarterly Operational Performance",
                 f"Management maintains current fiscal year revenue and earnings guidance for {ticker}.",
                 "NEUTRAL", 0.05)
            ]
            
        cur_date = date or datetime.datetime.now().strftime("%Y-%m-%d")
        for idx, (title, content, label, score) in enumerate(raw_items):
            doc = FinancialDocument(
                doc_id=f"{ticker}_{idx}",
                ticker=ticker,
                date=cur_date,
                headline=title,
                content=content,
                source="Financial News Wire / SEC Ingestion",
                sentiment_label=label,
                sentiment_score=score
            )
            docs.append(doc)
            
        return docs

    def get_all_documents(self) -> List[FinancialDocument]:
        """
        Gathers all financial documents across all tickers.
        """
        all_docs = []
        for ticker in self.tickers:
            all_docs.extend(self.get_documents_for_ticker(ticker))
        return all_docs

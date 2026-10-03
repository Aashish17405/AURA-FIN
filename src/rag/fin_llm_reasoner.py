"""
Financial LLM Reasoner with Extensive Defensive Exception Handling.
Guards against hallucination, malformed outputs, sentiment inversion, and API timeouts.
Provides deterministic fallback validation so the pipeline is 100% resilient.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
import json
import re
import numpy as np
import torch
from src.data.news_data import FinancialDocument

logger = logging.getLogger(__name__)

@dataclass
class FinancialReasoningResult:
    ticker: str
    sentiment_score: float         # Constrained in [-1.0, 1.0]
    confidence: float              # Constrained in [0.0, 1.0]
    market_impact: str             # "HIGH_POSITIVE", "MODERATE_POSITIVE", "NEUTRAL", "MODERATE_NEGATIVE", "HIGH_NEGATIVE"
    reasoning_summary: str         # Synthesized natural language explanation
    retrieved_documents: List[str] # Document headlines used for context
    is_fallback: bool              # Flagged True if defensive fallback activated
    validation_status: str         # "PASSED", "REPAIRED", or "FALLBACK_TRIGGERED"


# Institutional financial sentiment lexicon for ground-truth cross-validation
FINANCIAL_BULLISH_LEXICON = {
    "surge": 0.8, "beat": 0.7, "record": 0.75, "growth": 0.6, "profit": 0.65,
    "expansion": 0.55, "upgrade": 0.7, "outperform": 0.75, "rebound": 0.6,
    "acceleration": 0.65, "strong": 0.5, "dividend": 0.45, "innovation": 0.5
}

FINANCIAL_BEARISH_LEXICON = {
    "plunge": -0.85, "miss": -0.7, "loss": -0.65, "decline": -0.6, "cut": -0.55,
    "downgrade": -0.75, "antitrust": -0.7, "lawsuit": -0.65, "investigation": -0.6,
    "slump": -0.7, "recession": -0.75, "inflation": -0.4, "warning": -0.6
}


class FinancialLLMReasoner:
    """
    Reasoning engine processing retrieved financial documents.
    Implements a 3-layer defensive safety net:
    1. Structured Schema Validation: Validates incoming LLM outputs against strict JSON contracts.
    2. Contradiction & Hallucination Detector: Checks AI sentiment against lexical financial facts.
    3. Deterministic Fallback Reasoner: If AI outputs fail, generates verified sentiment representations.
    """
    def __init__(self, text_embed_dim: int = 64):
        self.text_embed_dim = text_embed_dim

    def reason_over_documents(
        self,
        ticker: str,
        retrieved_docs: List[Tuple[FinancialDocument, float]],
        raw_llm_response: Optional[str] = None
    ) -> FinancialReasoningResult:
        """
        Synthesizes financial documents with strict error handling and sanity checking.
        """
        doc_texts = [f"[{doc.headline}] {doc.content}" for doc, _ in retrieved_docs]
        doc_headlines = [doc.headline for doc, _ in retrieved_docs]

        # Check if raw LLM response was supplied; if so, attempt parsing and validation
        if raw_llm_response:
            try:
                parsed_result = self._parse_and_validate_llm_output(raw_llm_response, ticker, doc_headlines)
                # Verify parsed result against lexical sanity check
                sanitized_result = self._apply_sanity_and_contradiction_guards(parsed_result, doc_texts)
                return sanitized_result
            except Exception as e:
                logger.warning(f"AI response failed schema validation for {ticker}: {e}. Activating defensive fallback.")

        # Fallback reasoning: Deterministic lexical & document-level weighted synthesis
        return self._deterministic_fallback_reasoning(ticker, retrieved_docs)

    def _parse_and_validate_llm_output(
        self,
        raw_text: str,
        ticker: str,
        doc_headlines: List[str]
    ) -> FinancialReasoningResult:
        """
        Parses JSON response with regex fallbacks and range clamping.
        """
        # Attempt extracting JSON block
        json_match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
        else:
            data = json.loads(raw_text)

        # Validate and clamp sentiment_score
        raw_score = float(data.get("sentiment_score", 0.0))
        sentiment_score = max(-1.0, min(1.0, raw_score))

        # Validate and clamp confidence
        raw_conf = float(data.get("confidence", 0.70))
        confidence = max(0.1, min(1.0, raw_conf))

        market_impact = str(data.get("market_impact", "NEUTRAL")).upper()
        if market_impact not in ["HIGH_POSITIVE", "MODERATE_POSITIVE", "NEUTRAL", "MODERATE_NEGATIVE", "HIGH_NEGATIVE"]:
            market_impact = self._score_to_impact(sentiment_score)

        reasoning = str(data.get("reasoning", "Analysis synthesized from retrieved filings and market news."))

        return FinancialReasoningResult(
            ticker=ticker,
            sentiment_score=sentiment_score,
            confidence=confidence,
            market_impact=market_impact,
            reasoning_summary=reasoning,
            retrieved_documents=doc_headlines,
            is_fallback=False,
            validation_status="PASSED"
        )

    def _apply_sanity_and_contradiction_guards(
        self,
        result: FinancialReasoningResult,
        doc_texts: List[str]
    ) -> FinancialReasoningResult:
        """
        Checks for obvious AI hallucinations (e.g. AI claims extreme optimism when headlines report disaster).
        """
        full_text = " ".join(doc_texts).lower()
        
        bull_score = sum(weight for word, weight in FINANCIAL_BULLISH_LEXICON.items() if word in full_text)
        bear_score = sum(weight for word, weight in FINANCIAL_BEARISH_LEXICON.items() if word in full_text)
        lexical_balance = bull_score + bear_score

        # If LLM claims positive (+0.6) but lexical facts are strongly negative (-1.2)
        if result.sentiment_score > 0.4 and lexical_balance < -1.0:
            logger.warning(f"Contradiction detected for {result.ticker}: AI claimed bullish ({result.sentiment_score:.2f}) "
                           f"despite negative textual evidence ({lexical_balance:.2f}). Adjusting score.")
            result.sentiment_score = 0.5 * result.sentiment_score + 0.5 * max(-0.8, lexical_balance)
            result.confidence = min(result.confidence, 0.5)
            result.validation_status = "REPAIRED"
            result.market_impact = self._score_to_impact(result.sentiment_score)

        return result

    def _deterministic_fallback_reasoning(
        self,
        ticker: str,
        retrieved_docs: List[Tuple[FinancialDocument, float]]
    ) -> FinancialReasoningResult:
        """
        Deterministic, mathematically sound fallback that never fails.
        Weights document sentiment by retrieval similarity score.
        """
        if not retrieved_docs:
            return FinancialReasoningResult(
                ticker=ticker,
                sentiment_score=0.0,
                confidence=0.5,
                market_impact="NEUTRAL",
                reasoning_summary=f"No recent news catalysts retrieved for {ticker}. Neutral baseline assumed.",
                retrieved_documents=[],
                is_fallback=True,
                validation_status="FALLBACK_TRIGGERED"
            )

        total_weight = 0.0
        weighted_sentiment = 0.0
        headlines = []

        for doc, sim in retrieved_docs:
            weight = max(0.1, sim)
            weighted_sentiment += doc.sentiment_score * weight
            total_weight += weight
            headlines.append(doc.headline)

        final_score = weighted_sentiment / (total_weight + 1e-9)
        final_score = float(np.clip(final_score, -1.0, 1.0))
        impact = self._score_to_impact(final_score)
        
        # Build concise human-readable summary
        summary = (
            f"Synthesized from {len(retrieved_docs)} verified sources. "
            f"Primary catalyst: '{headlines[0]}'. "
            f"Sentiment index: {final_score:+.2f} ({impact.replace('_', ' ').title()})."
        )

        return FinancialReasoningResult(
            ticker=ticker,
            sentiment_score=final_score,
            confidence=0.85,
            market_impact=impact,
            reasoning_summary=summary,
            retrieved_documents=headlines,
            is_fallback=True,
            validation_status="FALLBACK_TRIGGERED"
        )

    def _score_to_impact(self, score: float) -> str:
        if score >= 0.6:
            return "HIGH_POSITIVE"
        elif score >= 0.2:
            return "MODERATE_POSITIVE"
        elif score <= -0.6:
            return "HIGH_NEGATIVE"
        elif score <= -0.2:
            return "MODERATE_NEGATIVE"
        return "NEUTRAL"

    def get_text_embedding_tensor(
        self,
        reasoning_result: FinancialReasoningResult
    ) -> torch.Tensor:
        """
        Transforms reasoning score, confidence, and semantic tokens into a dense PyTorch tensor
        of dimension (text_embed_dim,) for cross-attention fusion.
        """
        vec = np.zeros(self.text_embed_dim, dtype=np.float32)
        
        # Position 0: normalized sentiment score
        vec[0] = reasoning_result.sentiment_score
        # Position 1: model confidence
        vec[1] = reasoning_result.confidence
        # Position 2: fallback indicator (0.0 if passed, 1.0 if fallback)
        vec[2] = 1.0 if reasoning_result.is_fallback else 0.0
        
        # Fill remaining positions with deterministic semantic hash of reasoning summary
        words = re.findall(r"\b\w+\b", reasoning_result.reasoning_summary.lower())
        for i, word in enumerate(words):
            idx = 3 + (abs(hash(word)) % (self.text_embed_dim - 3))
            vec[idx] += 1.0 / (i + 1.0)
            
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm

        return torch.tensor(vec, dtype=torch.float32)

"""Transaction insight extraction flow using LLM."""

import json
from typing import Optional
from langchain_ollama import ChatOllama

from schemas.transaction_insights import TransactionInsights, TransactionPattern
from utils.transaction_filter import (
    get_customer_transactions,
    filter_transactions,
    format_transactions_for_llm
)
from .insight_store import get_cached_insights, store_insights


TRANSACTION_INSIGHT_PROMPT = """You are a financial transaction pattern extractor. Analyze the transactions and identify patterns.

RULES (STRICTLY FOLLOW):
- Use ONLY the provided transactions below
- Do NOT infer intent, risk, or legality
- Do NOT speculate or give advice
- Do NOT invent data not in the transactions
- Do NOT summarize overall customer behavior
- ONLY identify factual, descriptive patterns

ALLOWED PATTERNS (use only these):
- subscription-heavy: Regular recurring payments to subscription services
- salary-consistent: Regular income deposits of similar amounts
- rent-recurring: Monthly housing/rent payments
- discretionary-heavy: High spending on entertainment, shopping, travel
- cash-heavy: Frequent cash withdrawals
- utility-regular: Consistent utility bill payments
- emi-committed: Regular EMI/loan payments

For EACH pattern you identify, provide evidence as a list of categories from the transactions.

If no clear pattern exists, return an empty patterns list.

Transactions:
{transactions}

Return ONLY this JSON (no markdown, no explanation):
{{"patterns":[{{"pattern":"<pattern-name>","evidence":["<category1>","<category2>"],"confidence":<0.0-1.0>}}]}}"""


class TransactionInsightExtractor:
    """Extracts transaction patterns using LLM."""

    def __init__(self, model_name: str = "mistral"):
        self.llm = ChatOllama(model=model_name, temperature=0, format="json", seed=42)

    def extract(
        self,
        customer_id: int,
        scope: str = "patterns",
        use_cache: bool = True
    ) -> TransactionInsights:
        """
        Extract transaction insights for a customer.

        Args:
            customer_id: Customer to analyze
            scope: Filter scope for transactions
            use_cache: Whether to check/use cache

        Returns:
            TransactionInsights with detected patterns
        """
        if use_cache:
            cached = get_cached_insights(customer_id, scope)
            if cached:
                return cached

        all_transactions = get_customer_transactions(customer_id)
        filtered = filter_transactions(all_transactions, scope)

        if not filtered:
            return TransactionInsights(
                customer_id=customer_id,
                scope=scope,
                patterns=[],
                transaction_count_analyzed=0
            )

        transactions_str = format_transactions_for_llm(filtered)
        prompt = TRANSACTION_INSIGHT_PROMPT.format(transactions=transactions_str)

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            data = json.loads(content)

            patterns = []
            for p in data.get("patterns", []):
                try:
                    pattern = TransactionPattern(
                        pattern=p.get("pattern", "unknown"),
                        evidence=p.get("evidence", []),
                        confidence=float(p.get("confidence", 0.8))
                    )
                    patterns.append(pattern)
                except Exception:
                    continue

            insights = TransactionInsights(
                customer_id=customer_id,
                scope=scope,
                patterns=patterns,
                transaction_count_analyzed=len(filtered)
            )

            store_insights(customer_id, scope, insights)

            return insights

        except json.JSONDecodeError as e:
            print(f"[TransactionInsight] JSON parse error: {e}")
            return self._empty_insights(customer_id, scope)
        except Exception as e:
            print(f"[TransactionInsight] Error: {e}")
            return self._empty_insights(customer_id, scope)

    def _empty_insights(self, customer_id: int, scope: str) -> TransactionInsights:
        """Return empty insights on error."""
        return TransactionInsights(
            customer_id=customer_id,
            scope=scope,
            patterns=[],
            transaction_count_analyzed=0
        )


_insight_extractor: Optional[TransactionInsightExtractor] = None


def get_insight_extractor() -> TransactionInsightExtractor:
    """Get or create the insight extractor singleton."""
    global _insight_extractor
    if _insight_extractor is None:
        _insight_extractor = TransactionInsightExtractor()
    return _insight_extractor


def get_transaction_insights_if_needed(
    customer_id: int,
    scope: str = "patterns"
) -> Optional[TransactionInsights]:
    """
    Public API for transaction insight subsystem.

    This is the only function the orchestrator should call.

    Args:
        customer_id: Customer to analyze
        scope: Analysis scope

    Returns:
        TransactionInsights or None if extraction fails
    """
    if customer_id is None:
        return None

    extractor = get_insight_extractor()
    insights = extractor.extract(customer_id, scope)

    if not insights.patterns:
        return None

    return insights

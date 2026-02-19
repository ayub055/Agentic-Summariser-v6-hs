"""Result merger for combining transaction insights with tool results."""

from typing import Optional

from schemas.transaction_insights import TransactionInsights


def merge_transaction_insights(
    data_str: str,
    insights: Optional[TransactionInsights] = None
) -> str:
    """
    Merge transaction insights into the explainer data context.

    This function is called by the explainer to enrich context with
    transaction patterns WITHOUT exposing raw transaction data.

    Args:
        data_str: Existing formatted data string from tool results
        insights: Optional transaction insights to merge

    Returns:
        Enhanced data string with pattern information
    """
    if not insights or not insights.patterns:
        return data_str

    insights_section = insights.to_explainer_context()

    if insights_section:
        return f"{data_str}\n\n{insights_section}"

    return data_str

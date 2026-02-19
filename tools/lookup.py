"""
Lookup tools for discovering available data.
"""

from langchain_core.tools import tool
from data.loader import get_transactions_df


@tool
def list_customers() -> str:
    """
    List all customer IDs (integers) available in the transaction data.

    Use this tool when you need to know which customers exist
    or when asked about the customers in the data.
    """
    df = get_transactions_df()
    customers = df['cust_id'].unique().tolist()
    display = customers[:10]
    suffix = f"... and {len(customers) - 10} more" if len(customers) > 10 else ""
    return f"Available customers: {', '.join(str(c) for c in display)}{suffix}"


@tool
def list_categories() -> str:
    """
    List all spending categories available in the transaction data.

    Use this tool when you need to know what categories exist
    or when the user asks about available spending categories.
    """
    df = get_transactions_df()
    categories = df['category_of_txn'].unique().tolist()
    return f"Available categories: {', '.join(categories)}"

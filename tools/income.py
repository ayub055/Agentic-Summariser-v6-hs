"""
Income-related tools for analyzing credit transactions.
"""

from langchain_core.tools import tool
from data.loader import get_transactions_df


@tool
def get_total_income(customer_id: int) -> str:
    """
    Get the total income (credit transactions) for a specific customer.

    Use this tool when asked about how much money a customer received
    or their total income/credits.

    Args:
        customer_id: The customer ID (e.g., 1)
    """
    df = get_transactions_df()
    filtered = df[
        (df['cust_id'] == customer_id) &
        (df['dr_cr_indctor'] == 'C')
    ]
    total = filtered['tran_amt_in_ac'].sum()
    return f"Customer {customer_id} total income: ${total:,.2f}"

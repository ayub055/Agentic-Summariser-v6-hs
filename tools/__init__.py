"""Tools module - LangChain tools and pure analytics functions."""

from .analytics import (
    debit_total,
    get_spending_by_category,
    top_spending_categories,
    spending_in_date_range,
)

from .income import get_total_income

from .lookup import list_customers, list_categories

from .schemas import TopSpendingInput, DateRangeInput

from . import analytics

from .category_resolver import category_presence_lookup

# LangChain tools for legacy agent
ALL_TOOLS = [
    debit_total,
    get_total_income,
    get_spending_by_category,
    top_spending_categories,
    spending_in_date_range,
    list_customers,
    list_categories,
]

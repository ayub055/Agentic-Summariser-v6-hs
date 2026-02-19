"""
Pydantic schemas for tool input validation.
Define complex input structures here.
"""

from pydantic import BaseModel, Field


class TopSpendingInput(BaseModel):
    """Input schema for top_spending_categories tool."""
    customer_id: int = Field(description="The customer ID (e.g., 1)")
    top_n: int = Field(default=5, description="Number of top categories to return")


class DateRangeInput(BaseModel):
    """Input schema for date-range based queries."""
    customer_id: int = Field(description="The customer ID (e.g., 1)")
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")


class CategoryInput(BaseModel):
    """Input schema for category-specific queries."""
    customer_id: int = Field(description="The customer ID")
    category: str = Field(description="Spending category (e.g., 'Groceries', 'Rent')")

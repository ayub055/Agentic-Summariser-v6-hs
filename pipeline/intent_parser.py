"""Intent parser using LLM to extract structured intent from user query."""

import json
import re
from difflib import get_close_matches
from langchain_ollama import ChatOllama

from schemas.intent import ParsedIntent, IntentType, CONFIDENCE_THRESHOLD_RETRY
from config.settings import PARSER_MODEL


# All valid categories for normalization
VALID_CATEGORIES = [
    "MNC_Companies", "Digital_Betting_Gaming", "Food", "Liquor_Smoke",
    "Bank_Fees_Charges", "Mobile_Bills", "Wallets", "E_Commerce",
    "Courier_Logistics", "Air_Travel", "E_Entertainment", "Mobility",
    "Railway", "Govt_Tax_Challan", "Hospital", "Grocery",
    "Fashion_Beauty", "Equipment_Construction", "Pharmacy", "Engineering",
    "Kids_School", "Education", "Rent", "Jewelry_Premium_Gifts",
    "Foreign_Transaction", "Payroll", "Investment", "Salary",
    "Electronics_Appliance", "Charity_Donations", "Books_Stationery",
    "Fuel", "Govt_Companies", "Hotel", "Insurance",
    "Personal_Home_Services", "Pet_Care", "Taxi_Cab", "Real_Estate",
    "Sports_Fitness", "EMI", "Finance", "P2P"
]

# All valid intents (excluding UNKNOWN)
VALID_INTENTS = [
    "total_spending", "total_income", "spending_by_category", "all_categories_spending", "top_categories",
    "spending_in_period", "financial_overview", "compare_categories",
    "list_customers", "list_categories",
    # Report intents
    "customer_report", "lender_profile", "credit_analysis", "debit_analysis",
    "transaction_statistics", "anomaly_detection", "balance_trend",
    "income_stability", "cash_flow",
    # Category presence
    "category_presence_lookup",
    # Bureau report
    "bureau_report",
    # Bureau chat
    "bureau_credit_cards", "bureau_loan_count", "bureau_delinquency", "bureau_overview"
]

PARSER_PROMPT = """You are a JSON extractor for a transaction analysis system. Extract intent from the query below.

INTENTS (choose the most specific one):
- total_spending: Get total spending/expenses for a customer (e.g., "What is the total spending?", "How much did I spend in total?")
- total_income: Get total income/credits for a customer (e.g., "What is the total income?", "How much did I earn?")
- spending_by_category: Spending in a specific category (e.g., "How much did I spend on Groceries?", "What did I spend on Rent?")
- all_categories_spending: Get spending breakdown for all categories (e.g., "Show me spending by category", "Break down spending by category")
- top_categories: Top N spending categories
- spending_in_period: Spending within a date range
- financial_overview: General overview of finances
- compare_categories: Compare spending between multiple categories
- list_customers: List all customers
- list_categories: List all categories
- customer_report: Generate a full PDF report for a customer with all financial data, salary, EMI, rent, cashflow, categories
- lender_profile: Creditworthiness/lender assessment report
- credit_analysis: Detailed analysis of credits/income (max credit, avg, median, sources)
- debit_analysis: Detailed analysis of debits/spending patterns
- transaction_statistics: Transaction counts and statistics
- anomaly_detection: Find unusual/spike transactions
- balance_trend: Balance trends over time
- income_stability: Income consistency and stability analysis
- cash_flow: Monthly cash flow (inflows vs outflows)
- category_presence_lookup: Check if customer has transactions for a specific category/behavior (e.g., betting, salary, rent, entertainment)
- combined_report: Generate a combined report that merges both the customer banking report and the bureau tradeline report into one document
- bureau_report: Generate a bureau/credit bureau/CIBIL tradeline report for a customer
- bureau_credit_cards: Check if customer has credit cards, get count and utilization
- bureau_loan_count: How many loans of a specific type (personal loan, home loan, etc.) the customer has. Put the loan type in "category" field.
- bureau_delinquency: Check if any loan or specific loan type is delinquent or has DPD. Put the loan type in "category" field if specified.
- bureau_overview: General bureau/tradeline summary (total tradelines, exposure, outstanding) without generating a full report
- unknown: If query doesn't match any intent

IMPORTANT for customer_report:
If user asks to "generate report", "create report", "full report", "PDF report", "comprehensive report" for a customer, classify as "customer_report".

Examples for customer_report:
- "Generate report for customer 9449274898" -> intent=customer_report, customer_id=9449274898
- "Create a report for 1234567890" -> intent=customer_report, customer_id=1234567890
- "Full report for customer 123" -> intent=customer_report, customer_id=123
- "Generate PDF report for 9449274898" -> intent=customer_report, customer_id=9449274898

IMPORTANT for category_presence_lookup:
If the user asks whether the customer spends on, pays for, receives, or has transactions related to a specific category (e.g., betting, gambling, salary, rent, entertainment, gaming), classify as "category_presence_lookup" and extract the category name.
Do NOT decide if the category is present - just extract the intent and category.

Examples for category_presence_lookup:
- "Does he spend on betting?" -> intent=category_presence_lookup, category=betting_gaming
- "Does customer pay rent?" -> intent=category_presence_lookup, category=rent
- "Does he receive salary?" -> intent=category_presence_lookup, category=salary
- "Any entertainment expenses?" -> intent=category_presence_lookup, category=entertainment
- "Is there gambling activity?" -> intent=category_presence_lookup, category=betting_gaming

IMPORTANT for combined_report:
If user asks to "generate combined report", "merged report", "both reports", or "combine banking and bureau" for a customer, classify as "combined_report".

Examples for combined_report:
- "Generate combined report for 100384958" -> intent=combined_report, customer_id=100384958
- "Merged report for customer 100384958" -> intent=combined_report, customer_id=100384958
- "Generate both reports for 100384958" -> intent=combined_report, customer_id=100384958

IMPORTANT for bureau_report:
If user asks to generate a "bureau report", "CIBIL report", "tradeline report", or "credit bureau report" for a customer, classify as "bureau_report".

Examples for bureau_report:
- "Generate bureau report for 100384958" -> intent=bureau_report, customer_id=100384958
- "Bureau report for customer 100384958" -> intent=bureau_report, customer_id=100384958
- "CIBIL report for 100384958" -> intent=bureau_report, customer_id=100384958
- "Tradeline report for customer 100384958" -> intent=bureau_report, customer_id=100384958

IMPORTANT for bureau chat queries (these are quick lookups, NOT full report generation):
- "Are there any credit cards?" -> intent=bureau_credit_cards
- "Credit card utilization?" -> intent=bureau_credit_cards
- "Does he have credit cards?" -> intent=bureau_credit_cards
- "How many personal loans?" -> intent=bureau_loan_count, category=personal_loan
- "How many home loans does he have?" -> intent=bureau_loan_count, category=home_loan
- "Loan count for business loan" -> intent=bureau_loan_count, category=business_loan
- "What loans does he have?" -> intent=bureau_loan_count
- "Is any loan delinquent?" -> intent=bureau_delinquency
- "Any DPD on personal loan?" -> intent=bureau_delinquency, category=personal_loan
- "Is there any overdue?" -> intent=bureau_delinquency
- "Bureau summary" -> intent=bureau_overview
- "Tradeline overview" -> intent=bureau_overview
- "What does the bureau look like?" -> intent=bureau_overview
- "Bureau details" -> intent=bureau_overview

LOAN TYPES for bureau queries: personal_loan, credit_card, home_loan, auto_loan, business_loan, gold_loan, two_wheeler_loan, consumer_durable, lap_las_lad, other

CATEGORIES: MNC_Companies, Digital_Betting_Gaming, Food, Liquor_Smoke, Bank_Fees_Charges, Mobile_Bills, Wallets, E_Commerce, Courier_Logistics, Air_Travel, E_Entertainment, Mobility, Railway, Govt_Tax_Challan, Hospital, Grocery, Fashion_Beauty, Equipment_Construction, Pharmacy, Engineering, Kids_School, Education, Rent, Jewelry_Premium_Gifts, Foreign_Transaction, Payroll, Investment, Salary, Electronics_Appliance, Charity_Donations, Books_Stationery, Fuel, Govt_Companies, Hotel, Insurance, Personal_Home_Services, Pet_Care, Taxi_Cab, Real_Estate, Sports_Fitness, EMI, Finance, P2P

DATE FORMAT: Use YYYY-MM-DD format (e.g., 2025-07-01)

Query: {query}

Return ONLY this JSON (no markdown, no explanation):
{{"intent":"<intent>","customer_id":<int or null>,"category":"<str or null>","categories":<list or null>,"start_date":"<YYYY-MM-DD or null>","end_date":"<YYYY-MM-DD or null>","top_n":5,"threshold_std":2.0}}"""


def normalize_category_name(category: str) -> str | None:
    """Normalize category name using case-insensitive matching."""
    if not category:
        return None

    category_lower = category.lower().strip()
    category_map = {cat.lower(): cat for cat in VALID_CATEGORIES}

    if category_lower in category_map:
        return category_map[category_lower]

    # Fuzzy matching for typos
    matches = get_close_matches(category_lower, list(category_map.keys()), n=1, cutoff=0.7)
    if matches:
        return category_map[matches[0]]

    return None


def validate_intent_name(intent_str: str) -> IntentType:
    """Validate and normalize intent string to IntentType enum."""
    if not intent_str:
        return IntentType.UNKNOWN

    intent_lower = intent_str.lower().strip()

    try:
        return IntentType(intent_lower)
    except ValueError:
        # Fuzzy match for typos
        matches = get_close_matches(intent_lower, VALID_INTENTS, n=1, cutoff=0.6)
        if matches:
            try:
                return IntentType(matches[0])
            except ValueError:
                pass
        return IntentType.UNKNOWN


def calculate_confidence(parsed: dict, query: str) -> float:
    """Calculate dynamic confidence score based on extraction quality."""
    score = 0.5  # Base score

    # Intent quality
    if parsed.get("intent") and parsed["intent"] != "unknown":
        score += 0.2

    # Customer ID presence (if query mentions customer)
    query_lower = query.lower()
    if parsed.get("customer_id") is not None:
        score += 0.15
    elif "customer" in query_lower and parsed.get("customer_id") is None:
        score -= 0.1  # Penalty: query mentions customer but not extracted

    # Category extraction quality
    if parsed.get("category") or parsed.get("categories"):
        normalized = normalize_category_name(parsed.get("category", ""))
        if normalized:
            score += 0.1
        elif parsed.get("categories"):
            score += 0.1

    # Date extraction quality
    if parsed.get("start_date") and parsed.get("end_date"):
        # Check format validity
        date_pattern = r"^\d{4}-\d{2}-\d{2}$"
        if re.match(date_pattern, parsed["start_date"]) and re.match(date_pattern, parsed["end_date"]):
            score += 0.1

    return min(max(score, 0.0), 1.0)


class IntentParser:
    def __init__(self, model_name: str = PARSER_MODEL):
        self.llm = ChatOllama(model=model_name, temperature=0, format="json", seed=42)

    def parse(self, query: str) -> ParsedIntent:
        prompt = PARSER_PROMPT.format(query=query)

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # With format="json", output should be clean JSON
            data = json.loads(content)

            # Validate and normalize intent
            intent_str = data.get("intent", "unknown")
            data["intent"] = validate_intent_name(intent_str)

            # Normalize category if present
            # Skip normalization for intents that use category for non-transaction purposes
            _skip_category_norm = {
                IntentType.CATEGORY_PRESENCE_LOOKUP,
                IntentType.BUREAU_LOAN_COUNT,
                IntentType.BUREAU_DELINQUENCY,
            }
            if data.get("category"):
                if data["intent"] not in _skip_category_norm:
                    normalized = normalize_category_name(data["category"])
                    data["category"] = normalized  # May be None if invalid
                # else: keep category as-is (loan type or category resolver handles it)

            # Normalize categories list if present
            if data.get("categories") and isinstance(data["categories"], list):
                normalized_cats = []
                for cat in data["categories"]:
                    normalized = normalize_category_name(cat)
                    if normalized:
                        normalized_cats.append(normalized)
                data["categories"] = normalized_cats if normalized_cats else None

            # Calculate confidence dynamically
            data["confidence"] = calculate_confidence(data, query)
            data["raw_query"] = query

            # Post-processing corrections for common misclassifications
            query_lower = query.lower()
            if (data["intent"] == IntentType.SPENDING_BY_CATEGORY and
                not data.get("category") and
                any(kw in query_lower for kw in ["total spending", "total expense", "spend in total"])):
                # Correct misclassification: "total spending" should be TOTAL_SPENDING, not SPENDING_BY_CATEGORY
                data["intent"] = IntentType.TOTAL_SPENDING
                data["confidence"] = min(data["confidence"] + 0.1, 1.0)  # Boost confidence for correction

            # Clean up null string values
            for key in ["category", "start_date", "end_date"]:
                if data.get(key) in ["null", "None", ""]:
                    data[key] = None

            return ParsedIntent(**data)

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw: {response.content[:300]}")
            return self._fallback_parse(query)
        except Exception as e:
            print(f"Parse error: {e}")
            return ParsedIntent(intent=IntentType.UNKNOWN, raw_query=query, confidence=0.0)

    def _fallback_parse(self, query: str) -> ParsedIntent:
        """Enhanced regex fallback when LLM JSON fails."""
        query_lower = query.lower()

        # Extract customer ID (multiple patterns)
        customer_id = None
        cust_patterns = [
            r'customer\s*[#:]?\s*(\d+)',
            r'cust(?:omer)?[_\s]?id\s*[=:]?\s*(\d+)',
            r'for\s+customer\s+(\d+)',
            r'for\s+(\d{10})',  # 10-digit phone number
            r'for\s+(\d+)',
            r'^(\d+)\s',  # ID at start
            r'(\d{10})',  # 10-digit phone number anywhere
        ]
        for pattern in cust_patterns:
            match = re.search(pattern, query_lower)
            if match:
                customer_id = int(match.group(1))
                break

        # Detect intent with priority ordering (most specific first)
        intent = IntentType.UNKNOWN

        # Category presence lookup patterns (check first - high priority)
        presence_patterns = [
            (r'does\s+(?:he|she|customer|they)\s+(?:spend|pay|have)\s+(?:on|for)?\s*(.+?)(?:\?|$)', True),
            (r'(?:is|are)\s+there\s+(?:any)?\s*(.+?)\s+(?:transactions?|expenses?|spending|activity)', True),
            (r'does\s+(?:he|she|customer|they)\s+receive\s+(.+?)(?:\?|$)', True),
            (r'any\s+(.+?)\s+(?:activity|transactions?|spending|expenses?)', True),
            (r'check\s+(?:for)?\s*(.+?)\s+(?:transactions?|presence)', True),
        ]

        for pattern, _ in presence_patterns:
            match = re.search(pattern, query_lower)
            if match:
                extracted_category = match.group(1).strip()
                # Clean up extracted category
                extracted_category = re.sub(r'\s+(transactions?|expenses?|spending|activity).*$', '', extracted_category)
                # Try to resolve to known category via alias
                from config.category_loader import resolve_category_alias
                resolved = resolve_category_alias(extracted_category)
                return ParsedIntent(
                    intent=IntentType.CATEGORY_PRESENCE_LOOKUP,
                    customer_id=customer_id,
                    category=resolved or extracted_category,
                    raw_query=query,
                    confidence=0.75
                )

        # Bureau chat intents (checked before bureau_report to avoid catch-all)
        if any(kw in query_lower for kw in ["credit card util", "credit card count", "any credit card", "has credit card", "have credit card"]):
            intent = IntentType.BUREAU_CREDIT_CARDS
        elif any(kw in query_lower for kw in ["delinquen", "dpd", "overdue", "default", "is any loan"]):
            intent = IntentType.BUREAU_DELINQUENCY
        elif any(kw in query_lower for kw in ["how many"]) and any(kw in query_lower for kw in ["loan", "tradeline", "pl", "hl", "bl"]):
            intent = IntentType.BUREAU_LOAN_COUNT
        elif any(kw in query_lower for kw in ["bureau summary", "bureau overview", "tradeline summary", "tradeline overview", "bureau detail", "what does the bureau"]):
            intent = IntentType.BUREAU_OVERVIEW

        # Combined report (must check before individual report intents)
        elif any(kw in query_lower for kw in ["combined report", "merged report", "both report", "complete combined", "merge report"]):
            intent = IntentType.COMBINED_REPORT

        # Report intents (bureau report — full PDF generation)
        elif any(kw in query_lower for kw in ["bureau report", "cibil report", "tradeline report", "credit bureau"]):
            intent = IntentType.BUREAU_REPORT
        elif any(kw in query_lower for kw in ["full report", "customer report", "comprehensive report", "complete report", "generate report", "create report", "make report", "report for", "generate a report", "pdf report"]):
            intent = IntentType.CUSTOMER_REPORT
        elif any(kw in query_lower for kw in ["lender", "creditworth", "lending", "loan", "credit profile", "underwriting"]):
            intent = IntentType.LENDER_PROFILE
        elif any(kw in query_lower for kw in ["anomal", "spike", "unusual", "outlier", "irregular"]):
            intent = IntentType.ANOMALY_DETECTION
        elif any(kw in query_lower for kw in ["balance trend", "running balance", "balance over time"]):
            intent = IntentType.BALANCE_TREND
        elif any(kw in query_lower for kw in ["income stability", "salary regularity", "income consistent"]):
            intent = IntentType.INCOME_STABILITY
        elif any(kw in query_lower for kw in ["cash flow", "inflow", "outflow"]):
            intent = IntentType.CASH_FLOW
        elif any(kw in query_lower for kw in ["credit analysis", "credit stats", "income analysis", "max credit"]):
            intent = IntentType.CREDIT_ANALYSIS
        elif any(kw in query_lower for kw in ["debit analysis", "spending analysis", "expense analysis"]):
            intent = IntentType.DEBIT_ANALYSIS
        elif any(kw in query_lower for kw in ["transaction count", "how many transaction", "transaction stats"]):
            intent = IntentType.TRANSACTION_STATISTICS

        # Existing intents
        elif any(kw in query_lower for kw in ["all categories", "spending by category", "category breakdown", "spend by category"]):
            intent = IntentType.ALL_CATEGORIES_SPENDING
        elif "compare" in query_lower and "categor" in query_lower:
            intent = IntentType.COMPARE_CATEGORIES
        elif "top" in query_lower and "categor" in query_lower:
            intent = IntentType.TOP_CATEGORIES
        elif any(kw in query_lower for kw in ["total spending", "spend in total", "total expense"]):
            intent = IntentType.TOTAL_SPENDING
        elif any(kw in query_lower for kw in ["total income", "total credit", "how much earned"]):
            intent = IntentType.TOTAL_INCOME
        elif "overview" in query_lower or "summary" in query_lower:
            intent = IntentType.FINANCIAL_OVERVIEW
        elif "list customer" in query_lower or "all customer" in query_lower:
            intent = IntentType.LIST_CUSTOMERS
        elif "list categor" in query_lower or "all categor" in query_lower:
            intent = IntentType.LIST_CATEGORIES

        # Extract category (single)
        category = None
        for cat in VALID_CATEGORIES:
            if cat.lower() in query_lower:
                category = cat
                break

        # Check for category-specific spending
        if category and intent == IntentType.UNKNOWN:
            intent = IntentType.SPENDING_BY_CATEGORY

        # Extract multiple categories for comparison
        categories = None
        if intent == IntentType.COMPARE_CATEGORIES or ("vs" in query_lower or "versus" in query_lower or "compare" in query_lower):
            found_cats = []
            for cat in VALID_CATEGORIES:
                if cat.lower() in query_lower:
                    found_cats.append(cat)
            if len(found_cats) >= 2:
                categories = found_cats
                intent = IntentType.COMPARE_CATEGORIES

        # Extract dates
        start_date = None
        end_date = None
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern, query)
        if len(dates) >= 2:
            start_date = dates[0]
            end_date = dates[1]
            if intent == IntentType.UNKNOWN:
                intent = IntentType.SPENDING_IN_PERIOD

        # Calculate confidence for fallback
        confidence = 0.5
        if customer_id:
            confidence += 0.15
        if intent != IntentType.UNKNOWN:
            confidence += 0.15
        if category or categories:
            confidence += 0.1

        return ParsedIntent(
            intent=intent,
            customer_id=customer_id,
            category=category,
            categories=categories,
            start_date=start_date,
            end_date=end_date,
            raw_query=query,
            confidence=confidence
        )


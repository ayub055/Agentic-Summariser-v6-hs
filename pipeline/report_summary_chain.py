"""Report summary chain - LLM-based customer review and persona generation.

This module generates:
1. Executive summary (3-4 lines) - financial metrics focus
2. Customer persona (4-5 lines) - lifestyle/behavior focus

Uses LangChain Expression Language (LCEL) with Ollama models.
"""

from dataclasses import asdict
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from schemas.customer_report import CustomerReport
from data.loader import get_transactions_df
from utils.helpers import mask_customer_id, format_inr
from schemas.loan_type import get_loan_type_display_name
from config.settings import EXPLAINER_MODEL


# Original summary prompt (customer-focused, kept for reference)
REVIEW_PROMPT_ORIGINAL = """Based on the following financial data for customer {customer_id}, write a 3-4 line professional financial review.

IMPORTANT RULES:
- Only mention data that is provided below
- Do NOT mention or reference missing sections
- Be factual and concise
- Highlight any red flags or positive signals for lending decision
- Focus on key financial patterns and observations

Financial Data:
{data_summary}

Write a concise, professional review:"""

# Summary prompt template - LENDER POV
# REVIEW_PROMPT = """You are a credit analyst assessing customer {customer_id} for lending purposes. Based on the financial data below, write a 3-4 line executive summary for loan underwriting.

# IMPORTANT RULES:
# - Only mention data that is provided below
# - Do NOT mention or reference missing sections
# - Focus on creditworthiness indicators: income stability, repayment capacity, existing obligations
# - Highlight any red flags or positive signals for lending decision

# Financial Data:
# {data_summary}

# Write a concise credit assessment summary:"""

# Default model for summary generation (from settings)
SUMMARY_MODEL = EXPLAINER_MODEL


def create_summary_chain(model_name: str = SUMMARY_MODEL):
    """
    Create an LCEL chain for generating customer reviews.

    Args:
        model_name: Ollama model to use (default: llama3.1:8b)

    Returns:
        LCEL chain that takes {customer_id, data_summary} and returns str
    """
    prompt = ChatPromptTemplate.from_template(REVIEW_PROMPT_ORIGINAL)
    llm = ChatOllama(model=model_name, temperature=0, seed=42)

    return prompt | llm | StrOutputParser()


def generate_customer_review(
    report: CustomerReport,
    model_name: str = SUMMARY_MODEL
) -> Optional[str]:
    """
    Generate an LLM-based customer review from populated report sections.

    This function:
    1. Extracts only populated sections from the report
    2. Builds a data summary string
    3. Invokes the LLM chain
    4. Returns the generated review (or None on failure)

    Args:
        report: CustomerReport with populated sections
        model_name: Ollama model to use

    Returns:
        Generated review string, or None if generation fails
    """
    # Build data summary from populated sections only
    sections = _build_data_summary(report)

    if not sections:
        return None

    data_summary = "\n".join(sections)

    try:
        chain = create_summary_chain(model_name)
        review = chain.invoke({
            "customer_id": mask_customer_id(report.meta.customer_id),
            "data_summary": data_summary
        })
        return review.strip() if review else None
    except Exception:
        # Fail-soft: PDF will still be generated without summary
        return None


def _build_data_summary(report: CustomerReport) -> list:
    """
    Build data summary lines from populated report sections.

    Only includes sections that have data - never mentions
    missing sections.

    Args:
        report: CustomerReport to summarize

    Returns:
        List of summary strings for each populated section
    """
    sections = []

    # Category spending
    if report.category_overview:
        top_cats = sorted(
            report.category_overview.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        cats_str = ", ".join(f"{k}: {v:,.0f}" for k, v in top_cats)
        sections.append(f"Top spending categories: {cats_str}")

    # Monthly cashflow
    if report.monthly_cashflow:
        total_inflow = sum(m.get('inflow', 0) for m in report.monthly_cashflow)
        total_outflow = sum(m.get('outflow', 0) for m in report.monthly_cashflow)
        avg_net = (total_inflow - total_outflow) / max(1, len(report.monthly_cashflow))
        sections.append(
            f"Monthly cashflow: Avg net {avg_net:,.0f} INR "
            f"(Total in: {total_inflow:,.0f}, out: {total_outflow:,.0f})"
        )

    # Salary
    if report.salary:
        sections.append(
            f"Salary income: {report.salary.avg_amount:,.0f} INR average "
            f"({report.salary.frequency} transactions)"
        )

    # EMIs
    if report.emis:
        total_emi = sum(e.amount for e in report.emis)
        emi_count = sum(e.frequency for e in report.emis)
        sections.append(f"EMI commitments: {total_emi:,.0f} INR ({emi_count} payments)")

    # Rent
    if report.rent:
        sections.append(
            f"Rent payments: {report.rent.amount:,.0f} INR "
            f"({report.rent.frequency} transactions)"
        )

    # Bills
    if report.bills:
        total_bills = sum(b.avg_amount * b.frequency for b in report.bills)
        sections.append(f"Utility bills: {total_bills:,.0f} INR total")

    # Top merchants
    if report.top_merchants:
        top_merchant = report.top_merchants[0]
        sections.append(
            f"Most frequent merchant: {top_merchant.get('name', 'Unknown')} "
            f"({top_merchant.get('count', 0)} transactions, "
            f"{top_merchant.get('total', 0):,.0f} INR)"
        )

    return sections


# Original persona prompt (general, kept for reference)
PERSONA_PROMPT_ORIGINAL = """Based on the complete financial profile for customer {customer_id}, describe who this customer is in 4-5 lines.

COMPLETE FINANCIAL DATA:
{comprehensive_data}

SAMPLE TRANSACTIONS:
{transaction_sample}

Describe the customer persona focusing on:
- Who they likely are (profession, lifestyle)
- Their financial behavior and discipline
- Spending patterns and priorities
- Overall financial health assessment

Write a 4-5 line customer persona description:"""

# Persona prompt template - LENDER POV (focuses on creditworthiness assessment)
# PERSONA_PROMPT = """You are a credit analyst building a borrower profile for customer {customer_id}. Based on the complete financial data below, write a 4-5 line borrower assessment.

# COMPLETE FINANCIAL DATA:
# {comprehensive_data}

# SAMPLE TRANSACTIONS:
# {transaction_sample}

# Describe the borrower profile focusing on:
# - Income source and stability (salaried/self-employed, regularity)
# - Financial discipline (savings behavior, spending control)
# - Existing debt burden and repayment track record
# - Risk indicators (irregular income, high discretionary spending, overleveraging)
# - Overall creditworthiness assessment (low/medium/high risk)

# Write a 4-5 line borrower profile for lending decision:"""


def create_persona_chain(model_name: str = SUMMARY_MODEL):
    """Create an LCEL chain for generating customer persona."""
    prompt = ChatPromptTemplate.from_template(PERSONA_PROMPT_ORIGINAL)
    llm = ChatOllama(model=model_name, temperature=0.1, seed=42)
    return prompt | llm | StrOutputParser()


def generate_customer_persona(
    report: CustomerReport,
    model_name: str = SUMMARY_MODEL
) -> Optional[str]:
    """
    Generate an LLM-based customer persona from all available data.

    Uses comprehensive report data plus transaction samples to create
    a 4-5 line persona description of the customer.

    Args:
        report: CustomerReport with populated sections
        model_name: Ollama model to use

    Returns:
        Generated persona string, or None if generation fails
    """
    # Build comprehensive data from report
    comprehensive_data = _build_comprehensive_data(report)

    # Get transaction sample
    transaction_sample = _get_transaction_sample(report.meta.customer_id)

    if not comprehensive_data:
        return None

    try:
        chain = create_persona_chain(model_name)
        persona = chain.invoke({
            "customer_id": mask_customer_id(report.meta.customer_id),
            "comprehensive_data": comprehensive_data,
            "transaction_sample": transaction_sample
        })
        return persona.strip() if persona else None
    except Exception:
        # Fail-soft: report will still be generated without persona
        return None


def _build_comprehensive_data(report: CustomerReport) -> str:
    """
    Build comprehensive data string from all report sections.

    Includes all available data for persona generation.
    """
    lines = []

    # Customer info
    if report.meta.prty_name:
        lines.append(f"Customer Name: {report.meta.prty_name}")
    lines.append(f"Total Transactions: {report.meta.transaction_count}")
    lines.append(f"Analysis Period: {report.meta.analysis_period}")

    # Compute overall financial metrics
    if report.monthly_cashflow:
        total_inflow = sum(m.get('inflow', 0) for m in report.monthly_cashflow)
        total_outflow = sum(m.get('outflow', 0) for m in report.monthly_cashflow)
        savings_rate = (total_inflow - total_outflow) / total_inflow if total_inflow > 0 else 0
        lines.append(f"\nFINANCIAL OVERVIEW:")
        lines.append(f"Total Income: {total_inflow:,.0f} INR")
        lines.append(f"Total Expenses: {total_outflow:,.0f} INR")
        lines.append(f"Net Position: {total_inflow - total_outflow:,.0f} INR")
        lines.append(f"Savings Rate: {savings_rate:.1%}")

    # Salary info
    if report.salary:
        lines.append(f"\nINCOME:")
        lines.append(f"Salary: {report.salary.avg_amount:,.0f} INR average ({report.salary.frequency} payments)")
        if report.salary.narration:
            lines.append(f"Source: {report.salary.narration[:60]}")

    # All spending categories
    if report.category_overview:
        lines.append(f"\nSPENDING BY CATEGORY:")
        sorted_cats = sorted(report.category_overview.items(), key=lambda x: x[1], reverse=True)
        for cat, amount in sorted_cats:
            lines.append(f"  - {cat}: {amount:,.0f} INR")

    # Monthly cashflow trend
    if report.monthly_cashflow:
        lines.append(f"\nMONTHLY CASHFLOW:")
        positive_months = sum(1 for m in report.monthly_cashflow if m.get('net', 0) > 0)
        negative_months = len(report.monthly_cashflow) - positive_months
        lines.append(f"Positive months: {positive_months}, Negative months: {negative_months}")

    # EMI commitments
    if report.emis:
        total_emi = sum(e.amount for e in report.emis)
        lines.append(f"\nEMI COMMITMENTS: {total_emi:,.0f} INR total")

    # Rent
    if report.rent:
        lines.append(f"\nRENT: {report.rent.amount:,.0f} INR ({report.rent.frequency} payments)")

    # Bills
    if report.bills:
        total_bills = sum(b.avg_amount * b.frequency for b in report.bills)
        lines.append(f"\nUTILITY BILLS: {total_bills:,.0f} INR total")

    # Top merchants
    if report.top_merchants:
        lines.append(f"\nTOP MERCHANTS:")
        for m in report.top_merchants[:5]:
            lines.append(f"  - {m.get('name', 'Unknown')[:40]}: {m.get('count', 0)} txns, {m.get('total', 0):,.0f} INR")

    return "\n".join(lines)


def _get_transaction_sample(customer_id: int, limit: int = 20) -> str:
    """
    Get sample of recent transactions for persona context.

    Args:
        customer_id: Customer to get transactions for
        limit: Maximum transactions to include

    Returns:
        Formatted string of transaction samples
    """
    try:
        df = get_transactions_df()
        cust_df = df[df['cust_id'] == customer_id].copy()

        if len(cust_df) == 0:
            return "No transactions available"

        # Sort by date descending to get recent transactions
        cust_df = cust_df.sort_values('tran_date', ascending=False).head(limit)

        lines = []
        for _, row in cust_df.iterrows():
            date = str(row.get('tran_date', 'N/A'))[:10]
            direction = row.get('dr_cr_indctor', 'D')
            amount = row.get('tran_amt_in_ac', 0)
            category = row.get('category_of_txn', 'Unknown')
            narration = str(row.get('tran_partclr', ''))[:50]

            dir_symbol = '+' if direction == 'C' else '-'
            lines.append(f"{date} | {dir_symbol}{amount:,.0f} | {category} | {narration}")

        return "\n".join(lines)
    except Exception:
        return "Transaction sample unavailable"


# =============================================================================
# Bureau Report — LLM Narration
# =============================================================================

BUREAU_REVIEW_PROMPT = """You are a senior credit analyst writing an executive summary for a loan underwriting committee.

IMPORTANT RULES:
- Only reference numbers and risk annotations provided below — do NOT invent figures
- No arithmetic — just narrate the pre-computed values and their tagged interpretations
- Features tagged [HIGH RISK], [MODERATE RISK], or [CONCERN] are red flags — highlight them in the Behavioral Insights paragraph only
- Features tagged [POSITIVE], [CLEAN], or [HEALTHY] are green signals — acknowledge them in the Behavioral Insights paragraph only

STRUCTURE YOUR RESPONSE IN TWO PARAGRAPHS:

1. PORTFOLIO OVERVIEW (6-10 lines): A factual summary of the customer's tradeline portfolio so the reader does not have to look at the raw data. Start with the big picture — total tradelines (how many live, how many closed), which loan products are present, total sanctioned exposure, total outstanding, and unsecured exposure. Then weave in the key highlights that stand out from the behavioral features: credit card utilization percentage, any DPD values above zero, missed payment percentages, enquiry counts, loan acquisition velocity, and any loan product counts that are unusually high. Present these as natural facts within the narrative flow — not as a separate list. NO risk commentary, NO opinions, NO concern flags — just state the portfolio composition and the notable data points together in one cohesive summary.

2. BEHAVIORAL INSIGHTS (4-6 lines): Now provide the risk interpretation. Use the tagged annotations ([HIGH RISK], [POSITIVE], etc.) and the COMPOSITE RISK SIGNALS to narrate the customer's credit behavior — enquiry pressure, repayment discipline, utilization, loan acquisition velocity. CRITICAL: Every inference MUST cite the actual number that backs it (e.g., "utilization is elevated at 65%", "3 new PL trades in 6 months signals loan stacking", "0% missed payments but DPD of 12 days detected"). Never state a risk opinion without the supporting data point.

Bureau Portfolio Summary:
{data_summary}

Write the two-paragraph bureau portfolio review:"""


def _annotate_value(value, thresholds):
    """Annotate a value with risk tag based on thresholds.

    Args:
        value: The numeric value (or None).
        thresholds: List of (comparator, threshold, tag) tuples, checked in order.
                    comparator is one of '>', '<', '>=', '<=', '=='.

    Returns:
        Tag string like '[HIGH RISK]' or '[POSITIVE]', or '' if no threshold matched.
    """
    if value is None:
        return ""
    for comparator, threshold, tag in thresholds:
        if comparator == ">" and value > threshold:
            return tag
        elif comparator == ">=" and value >= threshold:
            return tag
        elif comparator == "<" and value < threshold:
            return tag
        elif comparator == "<=" and value <= threshold:
            return tag
        elif comparator == "==" and value == threshold:
            return tag
    return ""


def _format_tradeline_features_for_prompt(tf) -> str:
    """Format TradelineFeatures with risk annotations for the LLM prompt.

    Each feature is annotated with a risk interpretation tag based on
    deterministic thresholds. Interaction signals are appended at the end.
    """
    tf_dict = asdict(tf) if not isinstance(tf, dict) else tf

    def _val(key):
        return tf_dict.get(key)

    def _fmt(value):
        if value is None:
            return "N/A"
        return f"{value:.2f}" if isinstance(value, float) else str(value)

    lines = []

    # --- Loan Activity ---
    lines.append("  LOAN ACTIVITY:")
    v = _val("new_trades_6m_pl")
    if v is not None:
        tag = _annotate_value(v, [(">=", 3, " [HIGH RISK — rapid PL acquisition]"),
                                   (">=", 2, " [MODERATE RISK — multiple recent PLs]")])
        lines.append(f"    New PL Trades in Last 6M: {v}{tag}")
    v = _val("months_since_last_trade_pl")
    if v is not None:
        tag = _annotate_value(v, [("<", 2, " [CONCERN — very recent PL activity]")])
        lines.append(f"    Months Since Last PL Trade: {_fmt(v)}{tag}")
    v = _val("months_since_last_trade_uns")
    if v is not None:
        tag = _annotate_value(v, [("<", 2, " [CONCERN — very recent unsecured activity]")])
        lines.append(f"    Months Since Last Unsecured Trade: {_fmt(v)}{tag}")
    # total_trades omitted — already shown in Portfolio Summary from executive inputs

    # --- DPD & Delinquency ---
    lines.append("  DPD & DELINQUENCY:")
    for field, label in [("max_dpd_6m_cc", "Max DPD Last 6M (CC)"),
                          ("max_dpd_6m_pl", "Max DPD Last 6M (PL)"),
                          ("max_dpd_9m_cc", "Max DPD Last 9M (CC)")]:
        v = _val(field)
        if v is not None:
            tag = _annotate_value(v, [(">", 90, " [HIGH RISK — severe delinquency]"),
                                       (">", 30, " [MODERATE RISK — significant DPD]"),
                                       (">", 0, " [CONCERN — past due detected]"),
                                       ("==", 0, " [CLEAN]")])
            lines.append(f"    {label}: {v}{tag}")
    v = _val("months_since_last_0p_pl")
    if v is not None:
        tag = _annotate_value(v, [(">=", 24, " [POSITIVE — no PL delinquency in 2+ years]"),
                                   (">=", 12, " [POSITIVE — clean for 1+ year]"),
                                   ("<", 6, " [CONCERN — recent PL delinquency]")])
        lines.append(f"    Months Since Last 0+ DPD (PL): {_fmt(v)}{tag}")
    v = _val("months_since_last_0p_uns")
    if v is not None:
        tag = _annotate_value(v, [(">=", 24, " [POSITIVE — no unsecured delinquency in 2+ years]"),
                                   ("<", 6, " [CONCERN — recent unsecured delinquency]")])
        lines.append(f"    Months Since Last 0+ DPD (Unsecured): {_fmt(v)}{tag}")

    # --- Payment Behavior ---
    lines.append("  PAYMENT BEHAVIOR:")
    v = _val("pct_missed_payments_18m")
    if v is not None:
        if v > 10:
            tag = " [HIGH RISK — frequent missed payments]"
        elif v > 0:
            tag = " [CONCERN — some missed payments]"
        elif v == 0:
            # Check if DPD values are non-zero — if so, 0% missed payments
            # is misleading and should not be tagged as POSITIVE
            has_dpd = any(
                _val(f) is not None and _val(f) > 0
                for f in ["max_dpd_6m_cc", "max_dpd_6m_pl", "max_dpd_9m_cc"]
            )
            if has_dpd:
                tag = " [NOTE — 0% formally missed but DPD delays detected on some products; payments were late]"
            else:
                tag = " [POSITIVE — no missed payments]"
        else:
            tag = ""
        lines.append(f"    % Missed Payments Last 18M: {_fmt(v)}{tag}")
    v = _val("pct_0plus_24m_all")
    if v is not None:
        tag = _annotate_value(v, [(">", 10, " [HIGH RISK]"), (">", 0, " [CONCERN]"),
                                   ("==", 0, " [CLEAN]")])
        lines.append(f"    % Trades with 0+ DPD in 24M (All): {_fmt(v)}{tag}")
    v = _val("pct_0plus_24m_pl")
    if v is not None:
        tag = _annotate_value(v, [(">", 10, " [HIGH RISK]"), (">", 0, " [CONCERN]"),
                                   ("==", 0, " [CLEAN]")])
        lines.append(f"    % Trades with 0+ DPD in 24M (PL): {_fmt(v)}{tag}")
    v = _val("pct_trades_0plus_12m")
    if v is not None:
        tag = _annotate_value(v, [(">", 10, " [HIGH RISK]"), (">", 0, " [CONCERN]"),
                                   ("==", 0, " [CLEAN]")])
        lines.append(f"    % Trades with 0+ DPD in 12M (All): {_fmt(v)}{tag}")
    v = _val("ratio_good_closed_pl")
    if v is not None:
        tag = _annotate_value(v, [(">=", 0.8, " [POSITIVE — strong closure track record]"),
                                   ("<", 0.5, " [HIGH RISK — poor closure history]"),
                                   ("<", 0.7, " [CONCERN — below average closure quality]")])
        lines.append(f"    Ratio Good Closed PL Loans: {v * 100:.0f}%{tag}")

    # --- Utilization ---
    lines.append("  UTILIZATION:")
    v = _val("cc_balance_utilization_pct")
    if v is not None:
        tag = _annotate_value(v, [(">", 75, " [HIGH RISK — over-utilized]"),
                                   (">", 50, " [MODERATE RISK — elevated utilization]"),
                                   ("<=", 30, " [HEALTHY]")])
        lines.append(f"    CC Balance Utilization: {_fmt(v)}%{tag}")
    v = _val("pl_balance_remaining_pct")
    if v is not None:
        tag = _annotate_value(v, [(">", 80, " [HIGH RISK — most PL balance still outstanding]"),
                                   (">", 50, " [MODERATE — significant PL balance remaining]"),
                                   ("<=", 30, " [POSITIVE — largely repaid]")])
        lines.append(f"    PL Balance Remaining: {_fmt(v)}%{tag}")

    # --- Enquiry Behavior ---
    lines.append("  ENQUIRY BEHAVIOR:")
    v = _val("unsecured_enquiries_12m")
    if v is not None:
        tag = _annotate_value(v, [(">", 15, " [HIGH RISK — very high enquiry pressure]"),
                                   (">", 10, " [MODERATE RISK — elevated enquiry pressure]"),
                                   ("<=", 3, " [HEALTHY — minimal enquiry activity]")])
        lines.append(f"    Unsecured Enquiries Last 12M: {v}{tag}")
    v = _val("trade_to_enquiry_ratio_uns_24m")
    if v is not None:
        tag = _annotate_value(v, [(">", 50, " [POSITIVE — high conversion rate]"),
                                   ("<", 20, " [CONCERN — low conversion, possible rejections]")])
        lines.append(f"    Trade-to-Enquiry Ratio (Unsec 24M): {_fmt(v)}%{tag}")

    # --- Loan Acquisition Velocity ---
    lines.append("  LOAN ACQUISITION VELOCITY:")
    for field, label in [("interpurchase_time_12m_plbl", "PL/BL (12M)"),
                          ("interpurchase_time_6m_plbl", "PL/BL (6M)"),
                          ("interpurchase_time_24m_all", "All Loans (24M)"),
                          ("interpurchase_time_12m_cl", "Consumer Loans (12M)")]:
        v = _val(field)
        if v is not None:
            tag = _annotate_value(v, [("<", 1, " [HIGH RISK — rapid loan stacking]"),
                                       ("<", 2, " [CONCERN — frequent acquisitions]"),
                                       (">=", 6, " [HEALTHY — measured pace]")])
            lines.append(f"    Avg Interpurchase Time {label}: {_fmt(v)} months{tag}")
    # Include HL/LAP and TWL only if present (less common)
    for field, label in [("interpurchase_time_9m_hl_lap", "HL/LAP (9M)"),
                          ("interpurchase_time_24m_hl_lap", "HL/LAP (24M)"),
                          ("interpurchase_time_24m_twl", "TWL (24M)")]:
        v = _val(field)
        if v is not None:
            lines.append(f"    Avg Interpurchase Time {label}: {_fmt(v)} months")

    # --- Interaction Signals (deterministic, computed from feature combinations) ---
    interaction_signals = _compute_interaction_signals(tf_dict)
    if interaction_signals:
        lines.append("  COMPOSITE RISK SIGNALS:")
        for signal in interaction_signals:
            lines.append(f"    >> {signal}")

    return "\n".join(lines)


def _compute_interaction_signals(tf_dict: dict) -> list:
    """Compute interaction-based risk signals from feature combinations.

    These are deterministic interpretations that require looking at
    multiple features together — something the LLM shouldn't do.
    """
    signals = []

    enquiries = tf_dict.get("unsecured_enquiries_12m")
    ipt_plbl = tf_dict.get("interpurchase_time_12m_plbl")
    new_pl_6m = tf_dict.get("new_trades_6m_pl")

    # Credit hungry + loan stacking
    if enquiries is not None and enquiries > 10 and new_pl_6m is not None and new_pl_6m >= 2:
        signals.append("CREDIT HUNGRY + LOAN STACKING: High enquiry activity ({}x in 12M) "
                        "combined with {} new PL trades in 6M".format(enquiries, new_pl_6m))

    # Rapid loan stacking with low interpurchase time
    if ipt_plbl is not None and ipt_plbl < 2 and new_pl_6m is not None and new_pl_6m >= 2:
        signals.append("RAPID PL STACKING: Avg {:.1f} months between PL/BL acquisitions "
                        "with {} new trades in 6M".format(ipt_plbl, new_pl_6m))

    # Clean repayment profile
    dpd_6m_cc = tf_dict.get("max_dpd_6m_cc")
    dpd_6m_pl = tf_dict.get("max_dpd_6m_pl")
    dpd_9m_cc = tf_dict.get("max_dpd_9m_cc")
    missed = tf_dict.get("pct_missed_payments_18m")
    good_ratio = tf_dict.get("ratio_good_closed_pl")
    pct_0p_24m = tf_dict.get("pct_0plus_24m_all")

    all_dpd_clean = all(v is not None and v == 0 for v in [dpd_6m_cc, dpd_6m_pl, dpd_9m_cc])
    missed_clean = missed is not None and missed == 0
    pct_clean = pct_0p_24m is not None and pct_0p_24m == 0

    if all_dpd_clean and missed_clean and pct_clean:
        msg = "CLEAN REPAYMENT PROFILE: Zero DPD across all products and windows, no missed payments"
        if good_ratio is not None and good_ratio >= 0.8:
            msg += f", {good_ratio:.0%} good PL closure ratio"
        signals.append(msg)

    # Missed payments = 0 but DPD detected — apparent contradiction
    if missed_clean and not all_dpd_clean:
        dpd_details = []
        if dpd_6m_cc is not None and dpd_6m_cc > 0:
            dpd_details.append(f"CC 6M: {dpd_6m_cc} days")
        if dpd_6m_pl is not None and dpd_6m_pl > 0:
            dpd_details.append(f"PL 6M: {dpd_6m_pl} days")
        if dpd_9m_cc is not None and dpd_9m_cc > 0:
            dpd_details.append(f"CC 9M: {dpd_9m_cc} days")
        if dpd_details:
            signals.append(
                "PAYMENT TIMING NUANCE: 0% missed payments in 18M but DPD detected ({}) — "
                "payments were eventually made but past due date; do NOT describe payment "
                "record as clean or positive".format(", ".join(dpd_details))
            )

    # High utilization + high outstanding
    cc_util = tf_dict.get("cc_balance_utilization_pct")
    pl_bal = tf_dict.get("pl_balance_remaining_pct")
    if cc_util is not None and cc_util > 50 and pl_bal is not None and pl_bal > 50:
        signals.append("ELEVATED LEVERAGE: CC utilization at {:.1f}% and {:.1f}% "
                        "PL balance still outstanding".format(cc_util, pl_bal))

    # High enquiries but low conversion (possible repeated rejections)
    trade_ratio = tf_dict.get("trade_to_enquiry_ratio_uns_24m")
    if enquiries is not None and enquiries > 10 and trade_ratio is not None and trade_ratio < 30:
        signals.append("LOW CONVERSION: High enquiry volume ({}) but only {:.1f}% "
                        "trade-to-enquiry conversion — suggests possible rejections".format(
                            enquiries, trade_ratio))

    return signals


def _build_bureau_data_summary(executive_inputs, tradeline_features=None) -> str:
    """Format BureauExecutiveSummaryInputs into a text block for the LLM prompt.

    Args:
        executive_inputs: BureauExecutiveSummaryInputs dataclass instance.
        tradeline_features: Optional TradelineFeatures dataclass instance.

    Returns:
        Formatted text summary string.
    """
    data = asdict(executive_inputs) if not isinstance(executive_inputs, dict) else executive_inputs
    product_breakdown = data.pop("product_breakdown", {})

    # Max DPD with timing info
    max_dpd = data.get('max_dpd', 'N/A')
    max_dpd_str = str(max_dpd) if max_dpd is not None else "N/A"
    dpd_months = data.get('max_dpd_months_ago')
    dpd_lt = data.get('max_dpd_loan_type')
    if max_dpd is not None and max_dpd != 'N/A':
        details = []
        if dpd_months is not None:
            details.append(f"{dpd_months} months ago")
        if dpd_lt:
            details.append(dpd_lt)
        if details:
            max_dpd_str += f" ({', '.join(details)})"

    # Unsecured outstanding as % of total outstanding
    total_os = data.get('total_outstanding', 0)
    unsec_os = data.get('unsecured_outstanding', 0)
    unsec_os_pct = f"{(unsec_os / total_os * 100):.0f}%" if total_os > 0 else "N/A"

    lines = [
        f"Total Tradelines: {data.get('total_tradelines', 0)}",
        f"Live Tradelines: {data.get('live_tradelines', 0)}",
        f"Total Sanction Amount: INR {format_inr(data.get('total_sanctioned', 0))}",
        f"Total Outstanding: INR {format_inr(data.get('total_outstanding', 0))}",
        f"Unsecured Sanction Amount: INR {format_inr(data.get('unsecured_sanctioned', 0))}",
        f"Unsecured Outstanding: {unsec_os_pct} of total outstanding",
        f"Max DPD (Days Past Due): {max_dpd_str}",
    ]

    # Add CC utilization if available in product breakdown
    for loan_type_key, vec in product_breakdown.items():
        vec_data = asdict(vec) if not isinstance(vec, dict) else vec
        util = vec_data.get("utilization_ratio")
        if util is not None:
            lt_display = get_loan_type_display_name(loan_type_key)
            lines.append(f"{lt_display} Utilization: {util * 100:.1f}%")

    # Product breakdown
    if product_breakdown:
        lines.append("\nProduct-wise Breakdown:")
        for loan_type_key, vec in product_breakdown.items():
            vec_data = asdict(vec) if not isinstance(vec, dict) else vec
            lt_display = get_loan_type_display_name(loan_type_key)
            lines.append(
                f"  - {lt_display}: {vec_data.get('loan_count', 0)} accounts "
                f"(Live: {vec_data.get('live_count', 0)}, Closed: {vec_data.get('closed_count', 0)}), "
                f"Sanctioned: INR {format_inr(vec_data.get('total_sanctioned_amount', 0))}, "
                f"Outstanding: INR {format_inr(vec_data.get('total_outstanding_amount', 0))}"
            )

    # Tradeline behavioral features
    if tradeline_features is not None:
        lines.append("\nBehavioral & Risk Features:")
        lines.append(_format_tradeline_features_for_prompt(tradeline_features))

    return "\n".join(lines)


def generate_bureau_review(
    executive_inputs,
    tradeline_features=None,
    model_name: str = SUMMARY_MODEL,
) -> Optional[str]:
    """Generate an LLM-based bureau portfolio review from executive summary inputs.

    The LLM receives ONLY pre-computed numbers — no raw tradeline data.

    Args:
        executive_inputs: BureauExecutiveSummaryInputs (dataclass or dict).
        tradeline_features: Optional TradelineFeatures (dataclass or dict).
        model_name: Ollama model to use.

    Returns:
        Generated narrative string, or None if generation fails.
    """
    data_summary = _build_bureau_data_summary(executive_inputs, tradeline_features)

    if not data_summary:
        return None

    try:
        prompt = ChatPromptTemplate.from_template(BUREAU_REVIEW_PROMPT)
        llm = ChatOllama(model=model_name, temperature=0, seed=42)
        chain = prompt | llm | StrOutputParser()

        review = chain.invoke({"data_summary": data_summary})
        return review.strip() if review else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Combined Executive Summary (banking + bureau synthesised)
# ---------------------------------------------------------------------------

COMBINED_EXECUTIVE_PROMPT = """Prepare a synthesised executive summary for customer {customer_id} \
by merging the banking transaction analysis and credit-bureau tradeline analysis below into \
ONE cohesive paragraph (6-8 lines).

STRICT RULES:
- Write in formal third-person tone throughout (e.g. "The customer exhibits…", never "we" or "I")
- Do NOT repeat the source summaries verbatim — distil and merge the key points
- Cover: income & cash-flow health, spending discipline, credit-portfolio exposure, \
payment behaviour / DPD, and an overall creditworthiness assessment
- If either summary is empty or missing, work with whatever is available
- Be factual — do not invent numbers that are not present in the inputs
- End with a clear one-line creditworthiness assessment (positive, cautious, or negative)
- Do NOT add any meta-commentary, personal notes, disclaimers, or remarks about the writing \
process — output ONLY the summary paragraph followed by the standard note below

After the summary paragraph, add exactly this note on a new line:
Note: This is a synthesised summary based on automated banking and bureau analyses. \
Independent verification is recommended before final credit decisions.

BANKING SUMMARY:
{banking_summary}

BUREAU SUMMARY:
{bureau_summary}

Write the combined executive summary:"""


def generate_combined_executive_summary(
    banking_summary: str,
    bureau_summary: str,
    customer_id: str,
    model_name: str = SUMMARY_MODEL,
) -> Optional[str]:
    """Generate a unified executive summary from both banking and bureau narratives.

    Args:
        banking_summary: The customer_review text from the banking report.
        bureau_summary: The narrative text from the bureau report.
        customer_id: Masked customer identifier.
        model_name: Ollama model to use.

    Returns:
        Synthesised summary string, or None if generation fails.
    """
    if not banking_summary and not bureau_summary:
        return None

    try:
        prompt = ChatPromptTemplate.from_template(COMBINED_EXECUTIVE_PROMPT)
        llm = ChatOllama(model=model_name, temperature=0, seed=42)
        chain = prompt | llm | StrOutputParser()

        result = chain.invoke({
            "customer_id": customer_id,
            "banking_summary": banking_summary or "(not available)",
            "bureau_summary": bureau_summary or "(not available)",
        })
        return result.strip() if result else None
    except Exception:
        return None

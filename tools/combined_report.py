"""Combined report tool - merges banking + bureau into one report.

Generates both individual reports (reusing caches), then renders
a unified combined PDF + HTML document.  If one data source is
unavailable the report is still generated with the available source.
"""

import logging
from typing import Optional, Tuple

from schemas.customer_report import CustomerReport
from schemas.bureau_report import BureauReport
from pipeline.report_orchestrator import generate_customer_report_pdf
from tools.bureau import generate_bureau_report_pdf

logger = logging.getLogger(__name__)


def generate_combined_report_pdf(
    customer_id: int,
) -> Tuple[Optional[CustomerReport], Optional[BureauReport], str]:
    """Generate a combined banking + bureau report as one PDF.

    Steps:
        1. Generate customer report (reuses cache if available)
        2. Generate bureau report (reuses cache if available)
        3. Render combined PDF + HTML

    If one data source is missing the report is still produced with a
    note about the absent source.

    Args:
        customer_id: The customer identifier (CRN).

    Returns:
        Tuple of (CustomerReport | None, BureauReport | None, combined_pdf_path).
    """
    # 1. Customer report (cached by report_orchestrator)
    customer_report = None
    try:
        customer_report, _ = generate_customer_report_pdf(customer_id)
    except Exception as e:
        logger.warning(f"Banking report unavailable for {customer_id}: {e}")

    # 2. Bureau report
    bureau_report = None
    try:
        bureau_report, _ = generate_bureau_report_pdf(customer_id)
    except Exception as e:
        logger.warning(f"Bureau report unavailable for {customer_id}: {e}")

    # 2.5 Generate combined executive summary (fail-soft)
    combined_summary = None
    banking_text = (customer_report.customer_review or "") if customer_report else ""
    bureau_text = (bureau_report.narrative or "") if bureau_report else ""
    try:
        from pipeline.report_summary_chain import generate_combined_executive_summary
        from utils.helpers import mask_customer_id
        combined_summary = generate_combined_executive_summary(
            banking_summary=banking_text,
            bureau_summary=bureau_text,
            customer_id=mask_customer_id(customer_id),
        )
    except Exception as e:
        logger.warning(f"Combined executive summary generation failed: {e}")

    # 3. Combined rendering
    from pipeline.combined_report_renderer import render_combined_report
    pdf_path = render_combined_report(
        customer_report, bureau_report, combined_summary=combined_summary,
    )

    return customer_report, bureau_report, pdf_path

"""
agent/tools.py — LangChain tool definitions for the SpendAssist agent.

Exposes two tools that the LangGraph agent can invoke:

  query_spend          — Read-only, role-scoped lookup against transactions.csv.
  draft_expense_claim  — Write-guard: assembles a claim draft and marks it
                         "awaiting_manager_approval" without touching the CSV.

Both tools use Pydantic schemas from schemas.py as their ``args_schema`` so
LangGraph validates every argument before execution.
"""

import csv
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from schemas import DraftExpenseClaimInput, QuerySpendInput

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_TRANSACTIONS_CSV = _ROOT / "data" / "transactions.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Valid categories per Company Z policy
_VALID_CATEGORIES = {"travel", "meals", "software", "supplies"}

# Mapping of natural-language time-period strings to a lookback in days.
# The agent is expected to pass one of these normalised strings.
_TIME_PERIOD_DAYS: dict[str, int] = {
    "today":         0,
    "this week":     7,
    "last week":     14,
    "this month":    30,
    "last month":    60,
    "last quarter":  90,
    "this quarter":  90,
    "this year":     365,
    "last year":     730,
}


def _parse_time_period(time_period: Optional[str]) -> Optional[date]:
    """Convert a natural-language ``time_period`` string to a cutoff date.

    Returns the earliest date that qualifies (i.e. ``today - lookback_days``),
    or ``None`` if ``time_period`` is ``None`` or unrecognised (meaning no
    date filter is applied).
    """
    if time_period is None:
        return None
    normalised = time_period.strip().lower()
    for key, days in _TIME_PERIOD_DAYS.items():
        if key in normalised:
            return (datetime.today() - timedelta(days=days)).date()
    logger.warning("Unrecognised time_period '%s' — no date filter applied.", time_period)
    return None


def _read_transactions() -> list[dict]:
    """Read all rows from data/transactions.csv and return them as dicts.

    Raises:
        FileNotFoundError: If transactions.csv is absent.
    """
    if not _TRANSACTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Transactions file not found at '{_TRANSACTIONS_CSV}'. "
            "Run scripts/generate_mock_data.py to generate it."
        )
    with _TRANSACTIONS_CSV.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _format_transactions(rows: list[dict]) -> str:
    """Render a list of transaction dicts as a human-readable table string."""
    if not rows:
        return "No matching transactions found for the given filters."

    header = (
        f"{'ID':<12} {'Employee':<10} {'Department':<32} "
        f"{'Category':<10} {'Merchant':<28} {'Amount':>8}  {'Date':<12} {'Status'}"
    )
    divider = "-" * len(header)
    lines = [header, divider]

    total = 0.0
    for row in rows:
        amount = float(row["amount"])
        total += amount
        lines.append(
            f"{row['transaction_id']:<12} {row['employee_id']:<10} "
            f"{row['department']:<32} {row['category']:<10} "
            f"{row['merchant']:<28} ${amount:>7.2f}  "
            f"{row['date']:<12} {row['status']}"
        )

    lines.append(divider)
    lines.append(f"Total records: {len(rows)}  |  Total amount: ${total:,.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: query_spend
# ---------------------------------------------------------------------------

@tool(args_schema=QuerySpendInput)
def query_spend(
    employee_id: Optional[str] = None,
    department: Optional[str] = None,
    category: Optional[str] = None,
    time_period: Optional[str] = None,
) -> str:
    """Perform a pre-approved, read-only lookup of Company Z expense transactions.

    Use this tool whenever the user asks about existing spend data — for example:
      - "Show me all my travel expenses"
      - "What did the Backend Engineering team spend on software last quarter?"
      - "List all returned claims for employee EMP-1023"

    The tool supports filtering by any combination of the following dimensions.
    All filters are applied with AND logic (i.e. every specified filter must match):

      employee_id  — Restrict results to a single employee (e.g. "EMP-1023").
      department   — Restrict results to a named department (e.g. "Marketing & Growth").
      category     — One of: travel, meals, software, supplies.
      time_period  — A natural-language window such as "last quarter", "this month",
                     "this year", "last 30 days".  If omitted, all dates are included.

    The tool reads directly from the pre-approved transactions ledger
    (data/transactions.csv) and never modifies any record.  It returns a
    formatted table of matching rows together with a total spend figure.

    Returns:
        A human-readable, tabular string listing every matching transaction
        with columns: transaction_id, employee_id, department, category,
        merchant, amount, date, status — plus a summary row showing the total
        record count and aggregate spend.

        If no records match the supplied filters the tool returns the string
        "No matching transactions found for the given filters."

    Raises:
        FileNotFoundError: If data/transactions.csv is missing.
        ValueError: If ``category`` is provided but is not one of the four
            permitted values (travel, meals, software, supplies).
    """
    logger.info(
        "query_spend called | employee_id=%s | department=%s | category=%s | time_period=%s",
        employee_id, department, category, time_period,
    )

    # Validate category early so the agent receives a clear error message
    if category and category.lower() not in _VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. "
            f"Must be one of: {', '.join(sorted(_VALID_CATEGORIES))}."
        )

    rows = _read_transactions()
    cutoff_date = _parse_time_period(time_period)

    filtered: list[dict] = []
    for row in rows:
        # --- employee_id filter (case-insensitive) ---
        if employee_id and row["employee_id"].lower() != employee_id.lower():
            continue

        # --- department filter (case-insensitive substring match) ---
        if department and department.lower() not in row["department"].lower():
            continue

        # --- category filter (case-insensitive exact match) ---
        if category and row["category"].lower() != category.lower():
            continue

        # --- time_period filter ---
        if cutoff_date:
            try:
                txn_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
            except ValueError:
                logger.warning("Skipping row with unparseable date: %s", row["date"])
                continue
            if txn_date < cutoff_date:
                continue

        filtered.append(row)

    logger.info("query_spend returning %d row(s).", len(filtered))
    return _format_transactions(filtered)


# ---------------------------------------------------------------------------
# Tool: draft_expense_claim
# ---------------------------------------------------------------------------

@tool(args_schema=DraftExpenseClaimInput)
def draft_expense_claim(
    employee_id: str,
    amount: float,
    category: str,
    merchant: str,
    transaction_date: str,
) -> str:
    """Create a draft expense claim that is queued for manager approval.

    Use this tool when the user explicitly asks to submit, log, or create a
    new expense claim — for example:
      - "I need to claim a $450 hotel stay from last Tuesday"
      - "Please draft an expense for my $85 Figma subscription"
      - "Submit a meals claim for dinner with a client at The Capital Grille, $95"

    IMPORTANT GUARDRAILS — read before invoking:
      1. This tool NEVER writes to the transactions ledger or any database.
         It produces a draft object only; final submission requires explicit
         manager approval through a separate workflow.
      2. Do NOT invoke this tool to update or modify an existing transaction.
         Use it only to create brand-new claims.
      3. Always verify with the user that the details (amount, merchant,
         category, date) are correct before calling this tool.

    Argument validation (enforced by the Pydantic schema):
      employee_id       — Must be a non-empty string (e.g. "EMP-1023").
      amount            — Must be a positive float.
      category          — Must be one of: travel, meals, software, supplies.
      merchant          — The vendor name exactly as it appears on the receipt.
      transaction_date  — Must be in ISO 8601 format: YYYY-MM-DD.

    Returns:
        A JSON string representing the claim draft.  The object always
        includes ``"status": "awaiting_manager_approval"`` and a
        ``"draft_id"`` so the conversation can reference the draft later.
        Example:

        .. code-block:: json

            {
                "draft_id": "DRAFT-EMP1023-20261015",
                "employee_id": "EMP-1023",
                "amount": 95.00,
                "category": "meals",
                "merchant": "The Capital Grille",
                "transaction_date": "2026-10-15",
                "status": "awaiting_manager_approval",
                "note": "Draft created by SpendAssist. Awaiting manager approval before submission."
            }

    Raises:
        ValueError: If ``category`` is not one of the four permitted values,
            if ``amount`` is not positive, or if ``transaction_date`` is not
            a valid YYYY-MM-DD date string.
    """
    logger.info(
        "draft_expense_claim called | employee_id=%s | amount=%.2f | category=%s | merchant=%s | date=%s",
        employee_id, amount, category, merchant, transaction_date,
    )

    # --- Validate category ---
    if category.lower() not in _VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. "
            f"Must be one of: {', '.join(sorted(_VALID_CATEGORIES))}."
        )

    # --- Validate amount ---
    if amount <= 0:
        raise ValueError(
            f"Amount must be a positive value; received {amount}."
        )

    # --- Validate transaction_date format ---
    try:
        parsed_date = datetime.strptime(transaction_date, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(
            f"Invalid transaction_date '{transaction_date}'. "
            "Expected format: YYYY-MM-DD (e.g. '2026-10-15')."
        )

    # --- Check submission window (Clause 5.1: claims must be within 30 days) ---
    days_since_txn = (datetime.today().date() - parsed_date).days
    policy_warning: Optional[str] = None
    if days_since_txn > 30:
        policy_warning = (
            f"WARNING — Clause 5.1 violation: This transaction occurred "
            f"{days_since_txn} days ago. Company Z policy requires claims to be "
            "submitted within 30 days of the transaction date. "
            "Manager approval may be required to override this restriction."
        )
        logger.warning("Clause 5.1 potential violation for %s: %d days old.", employee_id, days_since_txn)

    # --- Assemble the draft (no file writes) ---
    draft_id = f"DRAFT-{employee_id.replace('-', '')}-{transaction_date.replace('-', '')}"

    draft: dict = {
        "draft_id":         draft_id,
        "employee_id":      employee_id,
        "amount":           round(amount, 2),
        "category":         category.lower(),
        "merchant":         merchant,
        "transaction_date": transaction_date,
        "status":           "awaiting_manager_approval",
        "note": (
            "Draft created by SpendAssist. Awaiting manager approval before submission."
        ),
    }

    if policy_warning:
        draft["policy_warning"] = policy_warning

    logger.info("Draft created: %s", draft_id)
    return json.dumps(draft, indent=2)


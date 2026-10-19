#!/usr/bin/env python3
"""
Generate mock expense transactions for Company Z.

This script generates synthetic expense claims adhering to Company Z's business rules:
- Columns: transaction_id, employee_id, department, category, merchant, amount, date, status
- 12 distinct engineering and corporate departments
- Categories limited strictly to: travel, meals, software, supplies
- Status: approximately 1 in 8 claims are 'returned' (policy violations), rest 'approved' or 'pending'
- Uses only Python standard libraries (csv, random, datetime, pathlib)
"""

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# 12 distinct engineering and corporate departments
DEPARTMENTS = [
    # Engineering Departments
    "Frontend Engineering",
    "Backend Engineering",
    "DevOps & Infrastructure",
    "Mobile Engineering",
    "Data & AI Platform",
    "Quality Assurance & Security",
    # Corporate Departments
    "Human Resources",
    "Finance & Accounting",
    "Legal & Compliance",
    "Marketing & Growth",
    "Sales & Business Development",
    "Product Management",
]

# Strict categories permitted by Company Z expense policy
CATEGORIES = ["travel", "meals", "software", "supplies"]

# Realistic merchants mapped per category
MERCHANTS_BY_CATEGORY = {
    "travel": [
        "Delta Air Lines",
        "United Airlines",
        "American Airlines",
        "Marriott Hotels",
        "Hilton Garden Inn",
        "Uber",
        "Lyft",
        "Amtrak",
        "Airbnb",
    ],
    "meals": [
        "Starbucks",
        "Sweetgreen",
        "Chipotle Mexican Grill",
        "DoorDash",
        "Panera Bread",
        "Pret A Manger",
        "The Capital Grille",
        "Local Bistro & Cafe",
    ],
    "software": [
        "AWS Cloud Services",
        "GitHub",
        "JetBrains",
        "Figma",
        "Slack Technologies",
        "Atlassian",
        "Google Cloud",
        "OpenAI API",
        "Datadog",
    ],
    "supplies": [
        "Staples",
        "Office Depot",
        "Amazon Business",
        "Best Buy",
        "Apple Store",
        "Uline Shipping Supplies",
    ],
}

# Amount ranges (min, max) by category
AMOUNT_RANGES = {
    "travel": (45.00, 1450.00),
    "meals": (12.50, 185.00),
    "software": (20.00, 550.00),
    "supplies": (15.00, 320.00),
}


def build_employee_directory(departments: list[str], employees_per_dept: int = 3) -> dict[str, list[str]]:
    """Generate consistent employee IDs mapped to departments."""
    emp_dir = {}
    emp_counter = 1001
    for dept in departments:
        emp_dir[dept] = []
        for _ in range(employees_per_dept):
            emp_dir[dept].append(f"EMP-{emp_counter}")
            emp_counter += 1
    return emp_dir


def generate_transactions(
    num_records: int = 100,
    seed: int | None = 42,
    base_date: datetime | None = None,
) -> list[dict]:
    """
    Generate mock expense transactions matching Company Z business rules.

    Args:
        num_records: Total number of expense claims to generate (default: 100).
        seed: Random seed for deterministic generation (default: 42).
        base_date: Reference end date for transaction timestamps (default: 2026-10-15).

    Returns:
        A list of dictionaries representing the expense claims.
    """
    if seed is not None:
        random.seed(seed)

    if base_date is None:
        base_date = datetime(2026, 10, 15)

    emp_directory = build_employee_directory(DEPARTMENTS)

    # Calculate status distribution:
    # Approximately 1 in 8 claims has status 'returned' (~12.5%).
    # For 100 claims: round(100 / 8) = 13 returned claims (~13.0%).
    # The remaining claims are distributed between 'approved' and 'pending'.
    num_returned = round(num_records / 8)
    num_remaining = num_records - num_returned
    num_approved = round(num_remaining * 0.62)
    num_pending = num_remaining - num_approved

    statuses = (
        ["returned"] * num_returned
        + ["approved"] * num_approved
        + ["pending"] * num_pending
    )
    random.shuffle(statuses)

    # Ensure even department distribution across claims
    dept_sequence = []
    while len(dept_sequence) < num_records:
        dept_pool = list(DEPARTMENTS)
        random.shuffle(dept_pool)
        dept_sequence.extend(dept_pool)
    dept_sequence = dept_sequence[:num_records]

    transactions = []
    for i in range(num_records):
        txn_id = f"TXN-{i + 1:04d}"
        department = dept_sequence[i]
        employee_id = random.choice(emp_directory[department])

        category = random.choice(CATEGORIES)
        merchant = random.choice(MERCHANTS_BY_CATEGORY[category])

        min_amt, max_amt = AMOUNT_RANGES[category]
        amount = round(random.uniform(min_amt, max_amt), 2)

        # Spread dates over the preceding 90 days
        days_offset = random.randint(0, 90)
        txn_date = (base_date - timedelta(days=days_offset)).strftime("%Y-%m-%d")

        status = statuses[i]

        transactions.append(
            {
                "transaction_id": txn_id,
                "employee_id": employee_id,
                "department": department,
                "category": category,
                "merchant": merchant,
                "amount": f"{amount:.2f}",
                "date": txn_date,
                "status": status,
            }
        )

    return transactions


def save_to_csv(transactions: list[dict], output_path: Path) -> None:
    """Save transaction records to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "transaction_id",
        "employee_id",
        "department",
        "category",
        "merchant",
        "amount",
        "date",
        "status",
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)


def main():
    parser = argparse.ArgumentParser(description="Generate mock expense data for Company Z.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "transactions.csv",
        help="Target output CSV file path (default: data/transactions.csv)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of mock records to generate (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    records = generate_transactions(num_records=args.count, seed=args.seed)
    save_to_csv(records, args.output)

    returned_count = sum(1 for r in records if r["status"] == "returned")
    approved_count = sum(1 for r in records if r["status"] == "approved")
    pending_count = sum(1 for r in records if r["status"] == "pending")

    print(f"Successfully generated {len(records)} transactions to {args.output}")
    print(f"Status summary: {returned_count} returned (~1 in {round(len(records) / returned_count, 1)}), "
          f"{approved_count} approved, {pending_count} pending")


if __name__ == "__main__":
    main()


"""
test_api.py — Automated test suite for the SpendAssist /chat endpoint.

Iterates through test cases covering CSV aggregation, RAG policy queries,
empty data handling, truncation/token limits, multi-tool execution, multi-turn
drafting with memory, and failsafe/out-of-scope queries.
"""

import os
import time
import requests

API_URL = os.environ.get("SPENDASSIST_API_URL", "http://localhost:8000/chat")

TEST_CASES = [
    {
        "name": "1. Single Employee Aggregation (CSV)",
        "session_id": "tc-1",
        "messages": ["What is the total amount spent by employee EMP-1015 on travel?"]
    },
    {
        "name": "2. Policy Edge Case (RAG)",
        "session_id": "tc-2",
        "messages": ["Are there any specific rules or approvals needed for purchasing software subscriptions?"]
    },
    {
        "name": "3. The Ghost Hunt (Empty CSV)",
        "session_id": "tc-3",
        "messages": ["How much did the Quality Assurance & Security department spend on travel yesterday?"]
    },
    {
        "name": "4. Truncation Guardrail (Token Limit)",
        "session_id": "tc-4",
        "messages": ["List every single transaction for the Backend Engineering department this year."]
    },
    {
        "name": "5. Combined Tool Execution (CSV + RAG)",
        "session_id": "tc-5",
        "messages": ["My employee ID is EMP-1030. How much did I spend on meals, and what is the company per diem limit for that?"]
    },
    {
        "name": "6. Incomplete Draft (Memory Check - 3 Turns)",
        "session_id": "tc-6",
        "messages": [
            "I need to draft an expense claim for $120 for software. My ID is EMP-1010.",
            "The merchant was JetBrains and the date was yesterday.",
            "Yes."
        ]
    },
    {
        "name": "7. Strict Failsafe (Out of Scope)",
        "session_id": "tc-7",
        "messages": ["How do I request a new laptop from the IT department or report a broken keyboard?"]
    },
    {
        "name": "8. Analytics & Truncation Handling",
        "session_id": "tc-8",
        "messages": ["What are the top five merchants for the Marketing & Growth department this year?"]
    },
    {
        "name": "9. Duplicate Transaction Policy (RAG)",
        "session_id": "tc-9",
        "messages": ["I accidentally paid for the same software subscription twice. Can I submit multiple claims for the exact same transaction ID?"]
    },
    {
        "name": "10. Duplicate Claim Attempt (Memory & Policy)",
        "session_id": "tc-10",
        "messages": [
            "Draft an expense claim for a $45 meal at Chipotle on 2026-10-21. My employee ID is EMP-1008.",
            "Yes.",
            "Draft an expense claim for a $45 meal at Chipotle on 2026-10-21. My employee ID is EMP-1008.",
            "Yes."
        ]
    },
    {
        "name": "11. Prohibited Item (Alcohol Guardrail)",
        "session_id": "tc-11",
        "messages": [
            "Draft an expense claim for a $120 client dinner at The Capital Grille yesterday. $40 of that was for a bottle of wine. My employee ID is EMP-1009 and category is meals."
        ]
    }
]


def run_tests():
    print("=" * 80)
    print("Starting SpendAssist Automated Test Suite (/chat)")
    print(f"Target URL: {API_URL}")
    print("=" * 80)

    for tc in TEST_CASES:
        test_name = tc["name"]
        session_id = tc["session_id"]
        messages = tc["messages"]

        print(f"\n{'#' * 80}")
        print(f"Test: {test_name} (session_id: {session_id})")
        print(f"{'#' * 80}")

        for turn_idx, message in enumerate(messages, 1):
            if len(messages) > 1:
                print(f"\n--- Turn {turn_idx}/{len(messages)} ---")

            print(f"User Message: {message}")

            payload = {
                "session_id": session_id,
                "message": message,
            }

            try:
                response = requests.post(API_URL, json=payload, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    print(f"Agent Answer: {data.get('answer')}")
                    print(f"Citations:    {data.get('citations')}")
                    print(f"Actions:      {data.get('actions')}")
                else:
                    print(f"Error ({response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                print(f"Connection Error: Could not connect to {API_URL}.")
                print("Make sure the backend server is running (e.g. uvicorn main:app --reload).")
                return
            except Exception as e:
                print(f"Request failed: {e}")

            time.sleep(2)

    print("\n" + "=" * 80)
    print("Test Suite Execution Finished.")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()


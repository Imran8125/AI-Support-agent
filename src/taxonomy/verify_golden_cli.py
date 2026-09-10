"""Interactive CLI tool for rapid inspection & hand-verification of the Golden Evaluation Set.

Usage:
  python src/taxonomy/verify_golden_cli.py
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
from src.taxonomy.taxonomy_def import INTENT_NAMES

GOLDEN_PATH = "golden_set/golden_eval_set.json"

def main():
    if not os.path.exists(GOLDEN_PATH):
        print(f"Error: {GOLDEN_PATH} not found. Run curate_golden_set.py first.")
        return
        
    with open(GOLDEN_PATH) as f:
        data = json.load(f)
        
    print("=" * 60)
    print(f"GOLDEN EVALUATION SET VERIFIER ({len(data)} examples)")
    print("=" * 60)
    print("Available intents:")
    for idx, name in enumerate(INTENT_NAMES, 1):
        print(f"  [{idx}] {name}")
    print("\nCommands per item:")
    print("  [Enter] Keep label and move to next")
    print("  [1-9]   Change intent to that number")
    print("  [e]     Toggle escalate (True/False)")
    print("  [s]     Skip item without saving changes")
    print("  [q]     Save and quit")
    print("-" * 60)
    
    modified = False
    for idx, item in enumerate(data):
        print(f"\n--- Item {idx+1}/{len(data)} (ID: {item['eval_id']}) ---")
        print(f"Query:    \"{item['customer_text']}\"")
        print(f"Current:  Intent = [{item['golden_intent']}] | Escalate = [{item['golden_escalate']}]")
        print(f"Reason:   {item.get('golden_escalation_reason', 'N/A')}")
        
        try:
            choice = input("Action [Enter/1-9/e/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
            
        if choice == 'q':
            break
        elif choice == 'e':
            item['golden_escalate'] = not item['golden_escalate']
            print(f"  Updated Escalate -> {item['golden_escalate']}")
            modified = True
        elif choice.isdigit() and 1 <= int(choice) <= len(INTENT_NAMES):
            new_intent = INTENT_NAMES[int(choice) - 1]
            item['golden_intent'] = new_intent
            print(f"  Updated Intent -> {new_intent}")
            modified = True
            
    if modified:
        with open(GOLDEN_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved updates to {GOLDEN_PATH}")
    else:
        print("\nNo changes made.")

if __name__ == "__main__":
    main()

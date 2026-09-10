"""Deterministic Escalation Rules for AppleSupport.

Implements zero-cost regex and keyword pattern matching for critical
escalation categories (PRD §4.6):
  1. Safety, Harassment & Abuse
  2. Stolen Devices & Law Enforcement / Police Reports
  3. Explicit Legal Threats & Lawsuits
  4. Explicit Demands for Human Agent / Supervisor
"""

from __future__ import annotations

import re

# Rule 1: Legal Threats & Lawsuits
LEGAL_REGEX = re.compile(
    r'\b(lawyer|attorney|lawsuit|sue you|suing|legal action|court|small claims|arbitration|consumer court)\b',
    re.IGNORECASE
)

# Rule 2: Theft & Police / Stolen Hardware
THEFT_REGEX = re.compile(
    r'\b(stolen|theft|robbed|police report|police complaint|stole my|pickpocket|burglar)\b',
    re.IGNORECASE
)

# Rule 3: Explicit Demand for Human / Manager
HUMAN_AGENT_REGEX = re.compile(
    r'\b(speak to a human|talk to a human|real person|human agent|transfer me to|supervisor|manager|customer care executive|actual human|bot)\b',
    re.IGNORECASE
)

# Rule 4: Harassment, Profanity & Hostility
HOSTILITY_REGEX = re.compile(
    r'\b(scam|fraud|thieves|liars|robbery|scammers|fuck|fucking|bitch|bastards|asshole)\b',
    re.IGNORECASE
)

def check_hard_escalation_rules(text: str) -> tuple[bool, str | None]:
    """Evaluates text against deterministic hard escalation rules.
    Returns (should_escalate, reason_if_triggered).
    """
    if THEFT_REGEX.search(text):
        return True, "Deterministic trigger: Stolen hardware or law enforcement involvement requires human specialist handling."
        
    if LEGAL_REGEX.search(text):
        return True, "Deterministic trigger: Explicit legal action or lawsuit threat requires formal legal/risk escalation."
        
    if HUMAN_AGENT_REGEX.search(text):
        return True, "Deterministic trigger: Customer explicitly requested a human representative or manager."
        
    if HOSTILITY_REGEX.search(text):
        return True, "Deterministic trigger: High-hostility sentiment or accusation of fraud/scam warrants human review."
        
    return False, None

"""Classifier prompt: document type and language from the opening lines."""

VERSION = "classifier-v1"

PROMPT = """Above is the start of a document uploaded by a person in India.
Classify it:
- doc_type: rental_agreement (leave and licence, lease, rent agreement), insurance_policy, loan_agreement, legal_notice (advocate's notice, summons, recovery notice), scheme_rules (government scheme guidelines or forms), bank_circular (bank letter or RBI notification), or other.
- language: ISO 639-1 code of the main language of the text, e.g. en, hi, mr, ta.
- confidence: 0 to 1, how sure you are of doc_type. Use below 0.6 if the text is short, unclear or could be several types."""

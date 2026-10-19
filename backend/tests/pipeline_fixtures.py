"""A MockProvider that answers every agent with the golden responses for the sample PDF."""

import re

from tests.mocks import MockProvider, load_golden


def asked_codes(call: dict) -> set[str]:
    return set(re.findall(r"^\[(C-\d+)\]", call["context"], re.MULTILINE))


def explainer_for_request(call: dict) -> dict:
    asked = asked_codes(call)
    return {"explanations": [e for e in load_golden("explainer")["explanations"] if e["clause_code"] in asked]}


def risk_for_request(call: dict) -> dict:
    golden = load_golden("risk")
    asked = asked_codes(call)
    rent = golden["monthly_rent_rupees"] if "C-04" in asked else None
    return {"monthly_rent_rupees": rent, "clauses": [c for c in golden["clauses"] if c["clause_code"] in asked]}


def qa_for_question(call: dict) -> dict:
    answers = load_golden("qa")
    if "parking" in call["prompt"].lower():
        return answers["parking"]
    return answers["leave_early"]


def golden_provider(**overrides) -> MockProvider:
    responses = {
        "ClassifierOutput": load_golden("classifier"),
        "ClauseExtraction": load_golden("clauses"),
        "KeyFactsOutput": load_golden("key_facts"),
        "RiskOutput": risk_for_request,
        "ExplainerOutput": explainer_for_request,
        "QAOutput": qa_for_question,
    }
    responses.update(overrides)
    return MockProvider(responses)

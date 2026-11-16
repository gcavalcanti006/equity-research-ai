"""Small helpers shared by the tests (fake API client, fake model answers)."""
import json
from types import SimpleNamespace

from src.schema import METRICS


def empty_raw():
    """A model answer where nothing was found."""
    return {"company": "", "period": "", "currency": "", "is_bank": False,
            "figures": [], "guidance": [],
            "management_tone": []}


def raw_fig(metric, value, unit="millions", currency="USD", basis="quarter", page=1,
            quote="q", label="x", scope="consolidated", definition_type="reported", definition="d"):
    """One entry of the model's `figures` list."""
    return {"metric": metric, "value": value, "unit": unit, "currency": currency,
            "basis": basis, "scope": scope, "definition_type": definition_type,
            "definition": definition, "label": label, "page": page, "quote": quote}


class FakeClient:
    """Pretends to be anthropic.Anthropic(). Returns canned answers and counts calls."""

    def __init__(self, raw_answers, stop_reason="end_turn"):
        self.raw_answers = list(raw_answers)
        self.calls = []
        self.stop_reason = stop_reason
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        raw = self.raw_answers[min(len(self.calls) - 1, len(self.raw_answers) - 1)]
        block = SimpleNamespace(type="text", text=json.dumps(raw))
        return SimpleNamespace(stop_reason=self.stop_reason, content=[block])


def fig(value, unit="millions", currency="USD", basis="quarter", page=1, quote="q", **extra):
    """A cleaned figure (what validator functions receive)."""
    return {"value": value, "unit": unit, "currency": currency, "basis": basis,
            "scope": "consolidated", "definition_type": "reported", "definition": "d",
            "label": "x", "page": page, "quote": quote, **extra}


def result(**metrics):
    """A cleaned, merged result with the given metrics."""
    r = {"file": "acme_3q26_usd", "company": "acme", "currency": "USD", "pdf_company": "Acme Corp",
         "pdf_period": "3Q26", "pdf_currency": "USD", "unit": "millions", "is_bank": False,
         "net_income_definition": "reported", "kind": "non_financial",
         "metrics": {m: None for m in METRICS}, "guidance": None, "tone": None}
    r["metrics"].update(metrics)
    return r

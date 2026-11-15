"""The strict JSON schema the model must follow for each chunk.

Design choice: figures are a LIST. The model adds one entry per metric it actually finds and
nothing for metrics it does not find, so "not found" simply means "not in the list" -> null.
(A first version had one fixed block per metric with a `found` flag. The API rejected it with
"compiled grammar is too large"; one small repeated item is much cheaper for the API to compile.)
"""

# Numeric metrics. Banks have no EBITDA, but the same schema is used for every company.
METRICS = [
    "net_revenue", "ebitda", "net_income", "net_debt", "net_debt_ebitda", "capex",
    "dividends", "roe",
    "loan_portfolio", "efficiency_ratio", "cet1",   # bank metrics
]
BANK_ONLY = ["loan_portfolio", "efficiency_ratio", "cet1"]

# Which basis we prefer for each metric when it appears several times (see extractor.rank_key).
PREFERRED_BASIS = {
    "net_revenue": "quarter", "ebitda": "quarter", "net_income": "quarter", "capex": "quarter",
    "dividends": "quarter", "roe": "quarter", "efficiency_ratio": "quarter",
    "net_debt": "point_in_time", "loan_portfolio": "point_in_time", "cet1": "point_in_time",
    "net_debt_ebitda": "last_12_months",
}

UNITS = ["millions", "billions", "thousands", "percent", "times", "other"]
BASES = ["quarter", "year_to_date", "last_12_months", "point_in_time", "other"]


def build_schema():
    figure = {
        "type": "object",
        "properties": {
            "metric": {"type": "string", "enum": METRICS},
            "value": {"type": "number"},
            "unit": {"type": "string", "enum": UNITS},
            "currency": {"type": "string"},   # ISO code like USD/BRL, or "none" for ratios
            "basis": {"type": "string", "enum": BASES},
            # consolidated = whole company; segment = business segment, division or subsidiary
            "scope": {"type": "string", "enum": ["consolidated", "segment", "other"]},
            "definition_type": {"type": "string",
                                "enum": ["reported", "adjusted", "recurring_managerial", "other"]},
            # short note on exactly what this number is, e.g. "net debt = total debt less cash,
            # excludes leases" or "CET1 (not total capital ratio)"
            "definition": {"type": "string"},
            "label": {"type": "string"},      # the exact label used in the document
            "page": {"type": "integer"},
            "quote": {"type": "string"},      # the exact sentence or table row
        },
        "required": ["metric", "value", "unit", "currency", "basis", "scope", "definition_type",
                         "definition", "label", "page", "quote"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            # Header fields: empty string "" means "not stated in these pages".
            "company": {"type": "string"},
            "period": {"type": "string"},
            "currency": {"type": "string"},
            "is_bank": {"type": "boolean"},
            "figures": {"type": "array", "items": figure},
            # Lists with 0 or 1 entries: empty list = not found.
            "guidance": {"type": "array", "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "page": {"type": "integer"},
                               "quote": {"type": "string"}},
                "required": ["text", "page", "quote"], "additionalProperties": False}},
            "management_tone": {"type": "array", "items": {
                "type": "object",
                "properties": {"tone": {"type": "string", "enum": ["positive", "neutral", "cautious"]},
                               "page": {"type": "integer"}, "quote": {"type": "string"}},
                "required": ["tone", "page", "quote"], "additionalProperties": False}},
        },
        "required": ["company", "period", "currency", "is_bank", "figures", "guidance", "management_tone"],
        "additionalProperties": False,
    }


CHUNK_SCHEMA = build_schema()

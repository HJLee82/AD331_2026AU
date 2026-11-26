"""Strictly typed Pydantic v2 models for invoice extraction."""
import re
from datetime import date
from typing import List, Literal, Optional

from dateutil import parser as date_parser
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

def format_validation_errors(exc: ValidationError) -> list[str]:
    """Pydantic errors -> ['line_items.0: line_total ...', ...] for reflection prompts."""
    out = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        out.append(f"{loc}: {err['msg']}")
    return out




TOLERANCE = 0.02
CURRENCY_ALIASES = {"$": "USD", "US$": "USD", "€": "EUR", "£": "GBP"}


def parse_money(value):
    """'$1,249.50' -> 1249.5, '€ 89,00' -> 89.0. Numbers pass through."""
    if not isinstance(value, str):
        return value
    s = re.sub(r"[A-Za-z$€£\s]", "", value)      # strip symbols/codes/spaces
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})*,\d{2}", s):  # European: 1.234,56
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    if not s:
        raise ValueError(f"Could not parse money value: {value!r}")
    num = float(s)
    return -num if negative else num


class LineItem(BaseModel):
    description: str = Field(min_length=1)
    quantity: int = Field(gt=0)
    unit_price: float = Field(gt=0)
    line_total: float = Field(ge=0)

    @field_validator("unit_price", "line_total", mode="before")
    @classmethod
    def clean_money(cls, v):
        return parse_money(v)

    @model_validator(mode="after")
    def check_line_math(self):
        expected = round(self.quantity * self.unit_price, 2)
        if abs(expected - self.line_total) > TOLERANCE:
            raise ValueError(
                f"line_total {self.line_total} != quantity x unit_price ({expected})"
            )
        return self


class InvoiceExtraction(BaseModel):
    invoice_number: str = Field(min_length=1)
    vendor_name: str = Field(min_length=1)
    invoice_date: date
    line_items: List[LineItem] = Field(min_length=1)
    subtotal: float = Field(ge=0)
    tax_amount: float = Field(ge=0)
    total_amount: float = Field(ge=0)
    currency: Literal["USD", "EUR", "GBP"]

    @field_validator("subtotal", "tax_amount", "total_amount", mode="before")
    @classmethod
    def clean_money(cls, v):
        return parse_money(v)

    @field_validator("invoice_date", mode="before")
    @classmethod
    def parse_date(cls, v):
        if isinstance(v, str):
            try:
                return date_parser.parse(v, fuzzy=True).date()
            except (ValueError, OverflowError):
                raise ValueError(f"Unrecognized date {v!r}; expected YYYY-MM-DD")
        return v

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, v):
        if isinstance(v, str):
            s = v.strip().upper()
            return CURRENCY_ALIASES.get(s, s)
        return v

    @model_validator(mode="after")
    def check_totals(self):
        items_sum = round(sum(i.line_total for i in self.line_items), 2)
        if abs(items_sum - self.subtotal) > TOLERANCE:
            raise ValueError(
                f"subtotal {self.subtotal} != sum of line_totals ({items_sum})"
            )
        expected_total = round(self.subtotal + self.tax_amount, 2)
        if abs(expected_total - self.total_amount) > TOLERANCE:
            raise ValueError(
                f"total_amount {self.total_amount} != subtotal + tax ({expected_total})"
            )
        return self


class ExtractionResult(BaseModel):
    validated_data: Optional[InvoiceExtraction] = None
    attempts_used: int = 0
    error_log: List[str] = Field(default_factory=list)
    execution_status: Literal[
        "SUCCESS", "REPAIRED_VIA_REFLECTION", "REPAIRED_VIA_HEURISTIC", "DLQ_FAILED"
    ]
    partial_data: Optional[dict] = None
    dlq_path: Optional[str] = None
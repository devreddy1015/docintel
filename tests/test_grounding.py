from datetime import date
from decimal import Decimal

import pytest

from docintel import grounding


def kinds(text):
    return [(f.kind, f.value) for f in grounding.facts(text)]


def test_extracts_dates_money_percent_numbers():
    fs = kinds("EMD Rs. 9,38,100/- by 03-Oct-2026 04:00 PM; 5% of value; 180 days")
    assert ("date", date(2026, 10, 3)) in fs
    assert ("money", Decimal("938100")) in fs
    assert ("percent", Decimal("5")) in fs
    assert ("number", Decimal("180")) in fs
    # The date's digits are not also counted as separate numbers.
    assert ("number", Decimal("2026")) not in fs


@pytest.mark.parametrize(
    "text,expected",
    [
        ("₹16 lakh", Decimal("1600000")),
        ("Rs. 4.69 crore", Decimal("46900000")),
        ("INR 2,000", Decimal("2000")),
        ("3.75 lakhs", Decimal("375000")),
    ],
)
def test_money_units(text, expected):
    assert grounding.facts(text)[0].value == expected


@pytest.mark.parametrize(
    "text",
    [
        "7 October 2026",
        "07-Oct-2026",
        "October 7, 2026",
        "07/10/2026",
        "2026-10-07",
        "7th Oct 2026",
    ],
)
def test_date_formats(text):
    assert grounding.facts(text)[0].value == date(2026, 10, 7)


def test_tender_ids_are_not_numbers():
    assert grounding.facts("Tender 2026_AMU_926330_3") == []


def test_supported_answer_passes_even_in_another_format():
    source = "Earnest Money Deposit (EMD): Rs. 9,38,100/-. Last date 03-Oct-2026 04:00 PM."
    r = grounding.check([("The EMD is ₹9,38,100, due by 3 October 2026.", [source])], [source])
    assert r.ok and r.checked == 2


def test_number_not_in_sources_is_rejected():
    source = "Performance Guarantee of 5% of the tendered value."
    r = grounding.check(
        [("The performance guarantee is 3% of the tendered value.", [source])], [source]
    )
    assert not r.ok and r.unsupported == ["3%"]


def test_calculated_value_is_rejected():
    """'40% of Rs. 8,00,000' is fine; a computed 'Rs. 3,20,000' that is not written is not."""
    source = "Similar work costing not less than 40% of the estimated cost of Rs. 8,00,000/-."
    r = grounding.check([("You need similar work worth Rs. 3,20,000.", [source])], [source])
    assert not r.ok


def test_claim_with_value_but_no_citation_is_rejected():
    r = grounding.check([("The EMD is Rs. 16,000.", [])], ["The EMD is Rs. 16,000."])
    assert not r.ok and r.uncited_claims


def test_text_without_values_needs_no_citation():
    r = grounding.check([("Bids are submitted online.", [])], [])
    assert r.ok and r.checked == 0

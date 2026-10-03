"""Run: python tests/test_context.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parda.pii.context import ContextPredictor, cuts_number, filter_spans  # noqa: E402


def sp(text, sub, label, k=0):
    i = -1
    for _ in range(k + 1):
        i = text.index(sub, i + 1)
    return {"label": label, "start": i, "end": i + len(sub), "score": 0.9}


def test_dob_needs_a_birth_label():
    t = "Date of Birth: 10/11/2002\n13/03 SMS CHARGES 20,160\nजन्म तिथि: 01/02/1990"
    spans = [sp(t, "10/11/2002", "DOB"), sp(t, "13/03", "DOB"), sp(t, "01/02/1990", "DOB")]
    assert [s["start"] for s in filter_spans(t, spans)] == [spans[0]["start"], spans[2]["start"]]


def test_unspaced_aadhaar_needs_a_label_but_spaced_is_kept():
    t = "Aadhaar No: 619752410246\n08/02 UPI/DR 383598852085\nRef 6197 5241 0246"
    spans = [sp(t, "619752410246", "AADHAAR"), sp(t, "383598852085", "AADHAAR"), sp(t, "6197 5241 0246", "AADHAAR")]
    kept = filter_spans(t, spans)
    assert [t[s["start"]:s["end"]] for s in kept] == ["619752410246", "6197 5241 0246"]


def test_spans_cutting_through_a_number_are_dropped():
    t = "Debit 13,274 and 9844512345 ok"
    assert cuts_number(t, t.index("274"), t.index("274") + 3)          # "74" of "13,274" -> inside
    assert cuts_number(t, t.index("13"), t.index("13") + 2)            # "13" of "13,274" -> inside
    assert not cuts_number(t, t.index("13,274"), t.index("13,274") + 6)
    assert not cuts_number(t, t.index("9844512345"), t.index("9844512345") + 10)
    spans = [sp(t, "274", "PHONE"), sp(t, "9844512345", "PHONE")]
    assert [t[s["start"]:s["end"]] for s in filter_spans(t, spans)] == ["9844512345"]
    d = "राशि ०१,२३४ जमा"                                                  # any script's digits count
    assert filter_spans(d, [sp(d, "२३४", "PHONE")]) == []


def test_rules_can_be_chosen_and_wrap_any_predictor():
    t = "13/03 SMS"
    spans = [sp(t, "13/03", "DOB")]
    assert filter_spans(t, spans, rules=("cut",)) == spans
    class P:
        name = "p"
        def predict(self, text): return spans
    cp = ContextPredictor(P())
    assert cp.name == "p+context" and cp.predict(t) == []


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok ", k)

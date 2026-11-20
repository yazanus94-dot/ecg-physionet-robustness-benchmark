"""AAMI heartbeat category mapping for MIT-BIH annotation codes.

Per the project specification (AAMI EC57-style categories), MIT-BIH
annotation symbols map to:

- N: N, L, R, e, j
- S: A, a, J, S
- V: V, E
- F: F
- Q: /, f, Q

Annotations not in the mapping are reported as excluded; unknown codes
(not produced by the MIT-BIH database) are reported as such.

Spec #1: the alignment between annotation, position and AAMI class is
preserved via :class:`AnnotationRecord` and :func:`annotate_annotations`;
excluded/unknown symbols appear explicitly and never shift subsequent
labels.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, NamedTuple

AAMI_MAP: dict[str, str] = {
    # N: normal beats
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    # S: supraventricular
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # V: ventricular
    "V": "V",
    "E": "V",
    # F: fusion
    "F": "F",
    # Q: paced / unclassified
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}

CLASSES = ["N", "S", "V", "F", "Q"]

# Symbols defined by the MIT-BIH Arrhythmia Database (per its documentation).
# Used to distinguish "excluded by the AAMI mapping" from "unknown code".
KNOWN_MITDB_SYMBOLS = frozenset(
    [
        "N",
        "L",
        "R",
        "e",
        "j",
        "A",
        "a",
        "J",
        "S",
        "V",
        "E",
        "F",
        "Q",
        "/",
        "f",
        "T",
        "P",
        "U",
        "x",
        "+",
        "!",
        "?",
        "#",
        "'",
        "^",
        "|",
        "~",
        "s",
        "t",
        "u",
        "v",
        "w",
        "y",
        "z",
        "m",
        "n",
        "p",
        "q",
        "r",
        "H",
        "K",
        "Z",
        "B",
    ]
)


class AnnotationRecord(NamedTuple):
    """One annotation, kept in alignment with its source data.

    - ``index``: original annotation index in the annotation list (never shifts).
    - ``symbol``: raw MITDB annotation symbol.
    - ``sample``: original annotation position (the R sample).
    - ``aami_class``: AAMI class when mapped, else ``None``.
    - ``status``: ``"mapped"`` | ``"excluded"`` | ``"unknown"``.
    """

    # "index" is overridden by `tuple.index` (a method); the field is
    # required by the project spec — type-ignore the assignment conflict.
    index: int  # type: ignore[assignment]
    symbol: str
    sample: int
    aami_class: str | None
    status: str


def _validated_sample(pos: Any, i: int, signal_length: int | None = None) -> int:
    """Validate one annotation position.

    Integer-compatible, non-negative, and — when ``signal_length`` is given —
    inside the signal bounds ``[0, signal_length)``.
    """
    try:
        v = int(pos)
    except (TypeError, ValueError):
        raise ValueError(
            f"annotation sample position must be integer-compatible, "
            f"got {pos!r} at index {i}"
        )
    if v != pos:
        raise ValueError(
            f"annotation sample position must be an integer, got {pos!r} at index {i}"
        )
    if v < 0:
        raise ValueError(
            f"invalid (negative) annotation sample position: {v} at index {i}"
        )
    if signal_length is not None and v >= signal_length:
        raise ValueError(
            f"annotation sample position {v} at index {i} is outside the "
            f"signal bounds [0, {signal_length})"
        )
    return v


def annotate_annotations(
    symbols, samples=None, signal_length: int | None = None
) -> list[AnnotationRecord]:
    """Return one :class:`AnnotationRecord` per symbol, preserving alignment.

    Excluded and unknown symbols appear explicitly with their original
    ``index``/``sample`` and ``aami_class=None``; mapped symbols carry their
    class. ``samples`` is the parallel list of annotation positions; it
    defaults to the list index. ``signal_length`` — the length of the signal
    the positions refer to, when available — enables bounds validation.

    Validation (spec #4): ``symbols`` and ``samples`` must have the same
    length; every sample position must be integer-compatible and non-negative
    — otherwise an explicit ``ValueError`` is raised (nothing is silently
    truncated or dropped). When ``signal_length`` is given, positions outside
    ``[0, signal_length)`` are also rejected.
    """
    symbols = [str(s) for s in symbols]
    if samples is None:
        samples = list(range(len(symbols)))
    if len(samples) != len(symbols):
        raise ValueError(
            f"symbols and samples must have the same length "
            f"(got {len(symbols)} symbols and {len(samples)} samples)"
        )
    samples = [_validated_sample(s, i, signal_length) for i, s in enumerate(samples)]
    out: list[AnnotationRecord] = []
    for i, (s, pos) in enumerate(zip(symbols, samples)):
        s = str(s)
        if s in AAMI_MAP:
            out.append(AnnotationRecord(i, s, int(pos), AAMI_MAP[s], "mapped"))
        elif s in KNOWN_MITDB_SYMBOLS:
            out.append(AnnotationRecord(i, s, int(pos), None, "excluded"))
        else:
            out.append(AnnotationRecord(i, s, int(pos), None, "unknown"))
    return out


def classify_annotations(symbols) -> tuple[list[str], Counter, Counter]:
    """Map annotation codes to AAMI classes.

    Returns ``(mapped, excluded, unknown)``: the class sequence (only
    mapped annotations, in order), the counts of known-but-excluded
    symbols, and the counts of symbols not part of the MIT-BIH database.
    """
    mapped: list[str] = []
    excluded: Counter = Counter()
    unknown: Counter = Counter()
    for rec in annotate_annotations(symbols):
        aami = rec.aami_class
        if rec.status == "mapped" and aami is not None:
            mapped.append(aami)
        elif rec.status == "excluded":
            excluded[rec.symbol] += 1
        else:
            unknown[rec.symbol] += 1
    return mapped, excluded, unknown


def annotation_report(
    mapped: list[str], excluded: Counter, unknown: Counter
) -> dict[str, Any]:
    """Summary of annotation handling: class counts and excluded/unknown codes."""
    counts = Counter(mapped)
    return {
        "class_counts": {c: counts.get(c, 0) for c in CLASSES},
        "total_mapped": len(mapped),
        "excluded": dict(sorted(excluded.items())),
        "unknown": dict(sorted(unknown.items())),
    }

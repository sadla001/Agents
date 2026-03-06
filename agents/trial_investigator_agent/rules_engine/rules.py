from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    domain: str
    priority: int
    field: str
    operator: str
    threshold: Any
    action: str
    message: str


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _to_int(x: Any, default: int = 999999) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _coerce_num(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _compare(value: Any, operator: str, threshold: Any) -> bool:
    """
    Operator support: ==, !=, >, >=, <, <=, in (comma list or parenthesised list), contains
    """
    op = (operator or "").strip().lower()
    if op in ("==", "=", "eq"):
        v = _coerce_num(value)
        t = _coerce_num(threshold)
        if v is not None and t is not None:
            return v == t
        return str(value).strip().lower() == str(threshold).strip().lower()
    if op in ("!=", "ne"):
        return str(value).strip().lower() != str(threshold).strip().lower()

    # Numeric comparisons
    v = _coerce_num(value)
    t = _coerce_num(threshold)
    if op in (">", "gt"):
        return v is not None and t is not None and v > t
    if op in (">=", "ge"):
        return v is not None and t is not None and v >= t
    if op in ("<", "lt"):
        return v is not None and t is not None and v < t
    if op in ("<=", "le"):
        return v is not None and t is not None and v <= t

    # Membership / string helpers
    if op == "in":
        # Accept both "a, b, c" and "(a, b, c)"
        raw = str(threshold).strip().strip("()")
        items = [p.strip().lower() for p in raw.split(",")]
        return str(value).strip().lower() in items
    if op == "contains":
        return str(threshold).strip().lower() in str(value).strip().lower()

    # Unknown op → fail closed
    return False


# ---------------------------------------------------------------------------
# CONDITION column parser
# ---------------------------------------------------------------------------

# Fields that evaluate_subject resolves upstream (LBTEST split into named keys).
# When a condition references these, it was already implicitly satisfied by the
# way evaluate_subject builds the context — treat as True.
_UPSTREAM_FILTERED_FIELDS = {"lbtest"}

# Phrases that indicate a purely narrative condition (no structured logic).
# These are documentation notes, not machine-executable guards.
_NARRATIVE_PATTERNS = re.compile(
    r"^(any |all |spanning|sequential|cumulative|conflicting|no |—$)",
    re.IGNORECASE,
)

# A single condition clause: FIELD  OPERATOR  VALUE
# Supports operators: >=, <=, !=, =, >, <, IN (with optional parentheses)
_CLAUSE_RE = re.compile(
    r"^(?P<field>\w+)\s+(?P<op>>=|<=|!=|>|<|=|in)\s+(?P<value>.+)$",
    re.IGNORECASE,
)


def _is_narrative(text: str) -> bool:
    """Return True if the condition string is human-readable documentation only."""
    t = text.strip()
    if not t or t in ("—", "-", "nan"):
        return True
    return bool(_NARRATIVE_PATTERNS.match(t))


def _eval_clause(clause: str, context: Dict[str, Any]) -> bool:
    """
    Parse and evaluate a single condition clause against the context dict.
    Returns True if the clause passes OR if it is unrecognised (fail-open for
    conditions that reference fields not yet in context).
    """
    clause = clause.strip()
    m = _CLAUSE_RE.match(clause)
    if not m:
        # Cannot parse → treat as documentation, pass through
        return True

    field = m.group("field").strip()
    op    = m.group("op").strip()
    value = m.group("value").strip()

    # LBTEST conditions are already pre-filtered upstream in evaluate_subject
    if field.lower() in _UPSTREAM_FILTERED_FIELDS:
        return True

    # Field not present in context → cannot evaluate → pass through (fail-open)
    if field not in context:
        return True

    return _compare(context[field], op, value)


def _parse_condition(
    condition_raw: Any,
    context: Dict[str, Any],
) -> Tuple[bool, bool]:
    """
    Parse the CONDITION column value and evaluate it against context.

    Returns
    -------
    (passes: bool, or_mode: bool)
        passes   – whether the condition evaluates to True
        or_mode  – if True the caller should OR this result with the main
                   rule check instead of AND-ing it (handles "OR …" prefix)

    Supported syntax
    ----------------
    * Empty / "—" / narrative text        → (True,  False)  [no extra guard]
    * "OR FIELD OP VALUE"                 → (bool,  True)   [OR override]
    * "FIELD OP VALUE"                    → (bool,  False)  [AND guard]
    * "F1 OP V1 AND F2 OP V2 [AND …]"    → (bool,  False)  [multi-AND guard]
    * "FIELD IN (a, b, c)"               → (bool,  False)
    * "FIELD = A or B"   (value list)    → (bool,  False)  [OR within values]
    """
    if condition_raw is None or (isinstance(condition_raw, float)):
        return True, False

    cond = str(condition_raw).strip()

    if _is_narrative(cond):
        return True, False

    # ── OR override prefix ───────────────────────────────────────────────────
    or_mode = False
    if cond.upper().startswith("OR "):
        or_mode = True
        cond = cond[3:].strip()

    # ── Split on " AND " (case-insensitive) ─────────────────────────────────
    # Guard: don't split inside an IN(…) parenthesised list
    and_parts = re.split(r"\s+AND\s+", cond, flags=re.IGNORECASE)

    results = []
    for part in and_parts:
        part = part.strip()

        # Handle "FIELD = A or B" → field must equal A or B
        # (only when " or " appears inside the value, not as a boolean connector)
        or_value_match = re.match(
            r"^(\w+)\s*(=|==)\s*(.+)$", part, re.IGNORECASE
        )
        if or_value_match and " or " in or_value_match.group(3).lower():
            field = or_value_match.group(1)
            alternatives = [
                v.strip() for v in re.split(r"\s+or\s+", or_value_match.group(3), flags=re.IGNORECASE)
            ]
            if field not in context:
                results.append(True)          # field absent → pass-through
            else:
                results.append(
                    any(_compare(context[field], "=", alt) for alt in alternatives)
                )
            continue

        results.append(_eval_clause(part, context))

    return all(results), or_mode


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_rules(
    rules_df: pd.DataFrame,
    domain: str,
    context: Dict[str, Any],
) -> Tuple[List[RuleHit], List[str]]:
    """
    Applies PROTOCOL_RULES rows for a given domain to a context dict.

    Expected rule columns (best-effort / flexible naming):
      RULE_ID | DOMAIN | PRIORITY | PARAMETER (FIELD/VARIABLE/METRIC)
      OPERATOR | THRESHOLD | ACTION | MESSAGE | CONDITION (optional)

    Condition evaluation logic
    --------------------------
    For each rule row that matches the domain:

      1. Evaluate the primary check:  context[PARAMETER] OPERATOR THRESHOLD
      2. Evaluate the CONDITION column (if present):
            - narrative / empty        → always True (no extra guard)
            - "OR FIELD OP VALUE"      → rule fires if primary OR condition
            - "FIELD OP VALUE [AND …]" → rule fires if primary AND condition
    """
    if rules_df is None or rules_df.empty:
        return [], []

    # ── Flexible column name resolution ─────────────────────────────────────
    cols = {c.lower(): c for c in rules_df.columns}

    def col(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_rule_id   = col("RULE_ID", "ID", "RULE")
    c_domain    = col("DOMAIN")
    c_priority  = col("PRIORITY", "ORDER")
    c_field     = col("FIELD", "VARIABLE", "METRIC", "PARAMETER")
    c_operator  = col("OPERATOR", "OP", "COMPARATOR")
    c_threshold = col("THRESHOLD", "VALUE", "CUTOFF")
    c_action    = col("ACTION")
    c_message   = col("MESSAGE", "RATIONALE", "WHY", "DESCRIPTION")
    c_condition = col("CONDITION")                                    # ← NEW

    if not (c_domain and c_field and c_operator and c_threshold and c_action):
        return [], []

    subset = rules_df[
        rules_df[c_domain].astype(str).str.upper() == domain.upper()
    ].copy()

    if subset.empty:
        return [], []

    if c_priority:
        subset["_prio"] = subset[c_priority].apply(_to_int)
        subset = subset.sort_values("_prio", ascending=True)
    else:
        subset["_prio"] = 999999

    hits:    List[RuleHit] = []
    actions: List[str]     = []

    for _, r in subset.iterrows():
        rid   = str(r[c_rule_id]) if c_rule_id else "RULE"
        prio  = int(r["_prio"])
        field = str(r[c_field]).strip()
        op    = str(r[c_operator]).strip()
        thr   = r[c_threshold]
        act   = str(r[c_action]).strip()
        msg   = (
            str(r[c_message]).strip()
            if c_message and pd.notna(r.get(c_message))
            else ""
        )

        # ── Primary rule check ───────────────────────────────────────────────
        val          = context.get(field)
        main_passes  = _compare(val, op, thr)

        # ── CONDITION column check ───────────────────────────────────────────
        cond_raw     = r[c_condition] if c_condition else None
        cond_passes, or_mode = _parse_condition(cond_raw, context)

        # ── Combine: OR-mode vs AND-mode ─────────────────────────────────────
        if or_mode:
            fires = main_passes or cond_passes
        else:
            fires = main_passes and cond_passes

        if fires:
            hit = RuleHit(
                rule_id=rid,
                domain=domain,
                priority=prio,
                field=field,
                operator=op,
                threshold=thr,
                action=act,
                message=msg or f"{field} {op} {thr} met (value={val})",
            )
            hits.append(hit)
            actions.append(act)

    return hits, actions
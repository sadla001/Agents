"""
rules.py — Protocol Rules Engine for the ONCO-2024-INT Clinician Agent.

Loads 24 rules from the xlsx once at startup and evaluates them
deterministically — no LLM involved in any calculation.

Four domain evaluators wrapped in one engine facade:

  Domain 1  SymptomBurden   SB-01 → SB-05
  Domain 2  TumorMarkers    TM-01 → TM-05
  Domain 3  RenalFunction   RF-01 → RF-05
  Domain 4  AlertEngine     CA-01 → CA-10  (reads flags set by domains 1-3)

Public API
  ProtocolRulesEngine(path)         — load and initialise
  engine.evaluate_symptom_burden()
  engine.evaluate_tumor_markers()
  engine.evaluate_renal_function()
  engine.run_alert_engine()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .data import load_rules, ProtocolRule


# ══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASSES  (returned by each evaluator)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SymptomBurdenResult:
    output: str; rule_id: str; severity: str; guideline: str
    flag: Optional[str]; ecog_change: float; pro_change_pct: float

@dataclass
class TumorMarkersResult:
    output: str; rule_id: str; severity: str; guideline: str
    flag: Optional[str]; pct_change: float

@dataclass
class RenalFunctionResult:
    output: str; rule_id: str; severity: str; guideline: str
    flag: Optional[str]; decline_pct: float; recovery_pct: float; egfr_current: float

@dataclass
class AlertResult:
    alert_text: Optional[str]; rule_id: str; severity: str
    suppressed: bool; action_codes: list[str]; active_flags: dict


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 1 — SYMPTOM BURDEN (SB-01 → SB-05)
# ══════════════════════════════════════════════════════════════════════════════

class _SymptomBurdenEvaluator:
    """
    Evaluates ECOG and PRO score changes against 5 priority-ordered tiers.

    SB-01  CRITICAL  ECOG worsened >= 2 grades
    SB-02  HIGH      ECOG worsened 1 grade, confirmed >= 2 visits
    SB-03  INFO      No meaningful change (ECOG < 1 grade, PRO < 10%)
    SB-04  LOW+      PRO improved 10-29% OR ECOG improved 1 grade
    SB-05  LOW+      PRO improved >= 30% OR ECOG improved >= 1 grade
    """

    def __init__(self, rules: list[ProtocolRule]):
        self._rules = rules

    def evaluate(
        self,
        ecog_baseline: float, ecog_current: float,
        pro_baseline: float,  pro_current: float,
        confirmed_visits: int = 2,
    ) -> SymptomBurdenResult:
        ecog_change    = ecog_current - ecog_baseline
        pro_change_pct = ((pro_current - pro_baseline) / pro_baseline * 100
                          if pro_baseline > 0 else 0.0)

        for rule in self._rules:
            rid = rule.rule_id

            if rid == "RULE-SB-01" and (ecog_change >= 2 or ecog_current >= 3):
                return self._result(rule, ecog_change, pro_change_pct, "RULE-SB-01_ACTIVE")

            elif rid == "RULE-SB-02" and ecog_change == 1 and confirmed_visits >= 2:
                return self._result(rule, ecog_change, pro_change_pct, "RULE-SB-02_ACTIVE")

            elif rid == "RULE-SB-03" and abs(ecog_change) < 1 and abs(pro_change_pct) < 10:
                return self._result(rule, ecog_change, pro_change_pct, None)

            elif rid == "RULE-SB-04":
                if (ecog_change == -1 or 10 <= pro_change_pct < 30) and confirmed_visits >= 2:
                    return self._result(rule, ecog_change, pro_change_pct, None)

            elif rid == "RULE-SB-05":
                if (ecog_change <= -1 or pro_change_pct >= 30) and confirmed_visits >= 2:
                    return self._result(rule, ecog_change, pro_change_pct, None)

        return self._result(self._rules[2], ecog_change, pro_change_pct, None)  # SB-03 fallback

    @staticmethod
    def _result(rule, ecog_change, pro_change_pct, flag) -> SymptomBurdenResult:
        return SymptomBurdenResult(
            output=rule.main_panel_output, rule_id=rule.rule_id,
            severity=rule.severity, guideline=rule.guideline, flag=flag,
            ecog_change=round(ecog_change, 1), pro_change_pct=round(pro_change_pct, 1),
        )


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 2 — TUMOR MARKERS (TM-01 → TM-05)
# ══════════════════════════════════════════════════════════════════════════════

class _TumorMarkersEvaluator:
    """
    Evaluates CA-125 (or any absolute marker) % change from baseline.
    pct_change positive = rising (bad), negative = falling (good).

    TM-01  CRITICAL  Rise >= 50% confirmed x2 (or >= 100% single visit)
    TM-02  HIGH      Rise 25-49% confirmed x2
    TM-03  INFO      Change < 20% in either direction
    TM-04  LOW+      Decline 20-49% confirmed x2
    TM-05  LOW+      Decline >= 50% confirmed x2
    """

    def __init__(self, rules: list[ProtocolRule]):
        self._rules = rules

    def evaluate(
        self, baseline: float, current: float, confirmed_visits: int = 2
    ) -> TumorMarkersResult:
        if baseline <= 0:
            return self._stable(0.0)

        pct = (current - baseline) / baseline * 100

        for rule in self._rules:
            rid = rule.rule_id

            if rid == "RULE-TM-01":
                if (pct >= 50 and confirmed_visits >= 2) or pct >= 100:
                    return self._result(rule, pct, "RULE-TM-01_ACTIVE")

            elif rid == "RULE-TM-02":
                if 25 <= pct < 50 and confirmed_visits >= 2:
                    return self._result(rule, pct, "RULE-TM-02_ACTIVE")

            elif rid == "RULE-TM-03" and abs(pct) < 20:
                return self._result(rule, pct, None)

            elif rid == "RULE-TM-04" and 20 <= -pct < 50 and confirmed_visits >= 2:
                return self._result(rule, pct, None)

            elif rid == "RULE-TM-05" and -pct >= 50 and confirmed_visits >= 2:
                return self._result(rule, pct, None)

        return self._stable(pct)

    @staticmethod
    def _result(rule, pct, flag) -> TumorMarkersResult:
        return TumorMarkersResult(
            output=rule.main_panel_output, rule_id=rule.rule_id,
            severity=rule.severity, guideline=rule.guideline,
            flag=flag, pct_change=round(pct, 1),
        )

    def _stable(self, pct) -> TumorMarkersResult:
        r = next((x for x in self._rules if x.rule_id == "RULE-TM-03"), None)
        return TumorMarkersResult(
            output=r.main_panel_output if r else "Tumor markers: → stable",
            rule_id="RULE-TM-03", severity="Info",
            guideline=r.guideline if r else "ASCO Marker Guidelines 2023",
            flag=None, pct_change=round(pct, 1),
        )


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 3 — RENAL FUNCTION (RF-01 → RF-05)
# ══════════════════════════════════════════════════════════════════════════════

class _RenalFunctionEvaluator:
    """
    Evaluates eGFR trends against 5 priority-ordered tiers.

    RF-01  CRITICAL  eGFR < 30 (absolute floor, single visit)
    RF-02  HIGH      Decline >= 25% from baseline
    RF-03  MODERATE  Decline 15-24% from baseline
    RF-04  INFO      Change < 15% AND eGFR >= 30 (stable)
    RF-05  LOW+      Recovery >= 15% from nadir AND eGFR >= 45

    RF-05 uses NADIR (lowest ever recorded) as denominator, not baseline.
    """

    def __init__(self, rules: list[ProtocolRule]):
        self._rules = rules

    def evaluate(
        self,
        egfr_baseline: float, egfr_current: float, egfr_nadir: float,
        confirmed_visits: int = 1,
    ) -> RenalFunctionResult:
        decline_pct  = ((egfr_baseline - egfr_current) / egfr_baseline * 100
                        if egfr_baseline > 0 else 0.0)
        recovery_pct = ((egfr_current - egfr_nadir) / egfr_nadir * 100
                        if egfr_nadir > 0 else 0.0)

        for rule in self._rules:
            rid = rule.rule_id

            if rid == "RULE-RF-01" and egfr_current < 30:
                return self._result(rule, egfr_current, decline_pct, 0.0, "RULE-RF-01_ACTIVE")

            elif rid == "RULE-RF-02" and decline_pct >= 25:
                out = rule.main_panel_output.replace("{pct}", f"{decline_pct:.0f}")
                return RenalFunctionResult(
                    output=out, rule_id=rule.rule_id, severity=rule.severity,
                    guideline=rule.guideline, flag="RULE-RF-02_ACTIVE",
                    decline_pct=round(decline_pct, 1), recovery_pct=0.0,
                    egfr_current=egfr_current,
                )

            elif rid == "RULE-RF-03" and 15 <= decline_pct < 25:
                out = rule.main_panel_output.replace("{pct}", f"{decline_pct:.0f}")
                return RenalFunctionResult(
                    output=out, rule_id=rule.rule_id, severity=rule.severity,
                    guideline=rule.guideline, flag="RULE-RF-03_ACTIVE",
                    decline_pct=round(decline_pct, 1), recovery_pct=0.0,
                    egfr_current=egfr_current,
                )

            elif rid == "RULE-RF-04" and abs(decline_pct) < 15 and egfr_current >= 30:
                return self._result(rule, egfr_current, decline_pct, 0.0, None)

            elif rid == "RULE-RF-05" and recovery_pct >= 15 and egfr_current >= 45:
                return self._result(rule, egfr_current, decline_pct, recovery_pct, "RULE-RF-05_ACTIVE")

        return self._result(
            next(r for r in self._rules if r.rule_id == "RULE-RF-04"),
            egfr_current, decline_pct, 0.0, None,
        )

    @staticmethod
    def _result(rule, egfr_current, decline_pct, recovery_pct, flag) -> RenalFunctionResult:
        return RenalFunctionResult(
            output=rule.main_panel_output, rule_id=rule.rule_id,
            severity=rule.severity, guideline=rule.guideline, flag=flag,
            decline_pct=round(decline_pct, 1), recovery_pct=round(recovery_pct, 1),
            egfr_current=egfr_current,
        )


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 4 — CLINICAL ALERT ENGINE (CA-01 → CA-10)
# ══════════════════════════════════════════════════════════════════════════════

# Flags that contribute to the multi-critical count check in CA-01
_CRITICAL_FLAGS = frozenset({"RULE-SB-01_ACTIVE", "RULE-TM-01_ACTIVE", "RULE-RF-01_ACTIVE"})


class _AlertEngine:
    """
    Reads boolean flags set by domains 1-3 and fires the single
    highest-priority matching alert rule.  One alert per card maximum.

    CA-01  CRITICAL  >= 2 simultaneous critical flags → MDT today
    CA-02  CRITICAL  RF-01 only → Dose hold + nephrology
    CA-03  CRITICAL  SB-01 only → Reassess treatment intent
    CA-04  CRITICAL  TM-01 only → Urgent imaging
    CA-05  HIGH      RF-02 active
    CA-06  HIGH      SB-02 active
    CA-07  HIGH      TM-02 active
    CA-08  MODERATE  RF-03 active
    CA-09  INFO+     RF-05 active (renal recovery)
    CA-10  DEFAULT   No flags → suppress alert section entirely
    """

    def __init__(self, rules: list[ProtocolRule]):
        self._rules = rules

    def run(self, active_flags: dict[str, bool]) -> AlertResult:
        critical_count = sum(active_flags.get(f, False) for f in _CRITICAL_FLAGS)

        for rule in self._rules:
            rid = rule.rule_id

            if   rid == "RULE-CA-01" and critical_count >= 2:
                return self._result(rule, active_flags)
            elif rid == "RULE-CA-02" and active_flags.get("RULE-RF-01_ACTIVE") and critical_count < 2:
                return self._result(rule, active_flags)
            elif rid == "RULE-CA-03" and active_flags.get("RULE-SB-01_ACTIVE") and critical_count < 2:
                return self._result(rule, active_flags)
            elif rid == "RULE-CA-04" and active_flags.get("RULE-TM-01_ACTIVE") and critical_count < 2:
                return self._result(rule, active_flags)
            elif rid == "RULE-CA-05" and active_flags.get("RULE-RF-02_ACTIVE"):
                return self._result(rule, active_flags)
            elif rid == "RULE-CA-06" and active_flags.get("RULE-SB-02_ACTIVE"):
                return self._result(rule, active_flags)
            elif rid == "RULE-CA-07" and active_flags.get("RULE-TM-02_ACTIVE"):
                return self._result(rule, active_flags)
            elif rid == "RULE-CA-08" and active_flags.get("RULE-RF-03_ACTIVE"):
                return self._result(rule, active_flags)
            elif rid == "RULE-CA-09" and active_flags.get("RULE-RF-05_ACTIVE"):
                return self._result(rule, active_flags)
            elif rid == "RULE-CA-10":
                return AlertResult(
                    alert_text=None, rule_id=rule.rule_id, severity=rule.severity,
                    suppressed=True, action_codes=rule.action_codes, active_flags=active_flags,
                )

        return AlertResult(alert_text=None, rule_id="RULE-CA-10", severity="Info",
                           suppressed=True, action_codes=[], active_flags=active_flags)

    @staticmethod
    def _result(rule, active_flags) -> AlertResult:
        return AlertResult(
            alert_text=rule.alert_output, rule_id=rule.rule_id,
            severity=rule.severity, suppressed=False,
            action_codes=rule.action_codes, active_flags=active_flags,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE FACADE — single import point for all agent tools
# ══════════════════════════════════════════════════════════════════════════════

class ProtocolRulesEngine:
    """
    Loads all 24 rules once and wires the four domain evaluators.
    Thread-safe for read-only use.
    """

    def __init__(self, path: Path):
        all_rules = load_rules(path)
        self._sb  = _SymptomBurdenEvaluator(all_rules["SYMPTOM_BURDEN"])
        self._tm  = _TumorMarkersEvaluator(all_rules["TUMOR_MARKERS"])
        self._rf  = _RenalFunctionEvaluator(all_rules["RENAL"])
        self._ca  = _AlertEngine(all_rules["CLINICAL_ALERT"])
        self.total_rules = sum(len(v) for v in all_rules.values())
        self.rule_counts = {d: len(v) for d, v in all_rules.items()}

    def evaluate_symptom_burden(
        self, ecog_baseline, ecog_current, pro_baseline, pro_current, confirmed_visits=2
    ) -> SymptomBurdenResult:
        return self._sb.evaluate(ecog_baseline, ecog_current, pro_baseline, pro_current, confirmed_visits)

    def evaluate_tumor_markers(
        self, baseline, current, confirmed_visits=2
    ) -> TumorMarkersResult:
        return self._tm.evaluate(baseline, current, confirmed_visits)

    def evaluate_renal_function(
        self, egfr_baseline, egfr_current, egfr_nadir, confirmed_visits=1
    ) -> RenalFunctionResult:
        return self._rf.evaluate(egfr_baseline, egfr_current, egfr_nadir, confirmed_visits)

    def run_alert_engine(self, active_flags: dict[str, bool]) -> AlertResult:
        return self._ca.run(active_flags)

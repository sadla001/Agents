"""
S3: Policy evaluation tool.

Implements the 4 policy rules:
  1. Semantic retrieval: use ICD10_DX + NDC to find the continuation-of-coverage
     clause in the policy reference table (simulating RAG retrieval).
  2. Gatekeeper check: does calculated data satisfy retrieved policy text?
  3. Audit log: generate word-for-word justification linking data to policy clause.
  4. Verdict: all pass -> Approve; data missing -> Pend; fail -> Deny.
"""

from payer_agent.tools.data_loader import get_dataframes


def evaluate_policy(
    patient_id: str,
    pdc: float,
    adherence_met: bool,
    improvement_met: bool,
    stable_disease: bool,
    stable_disease_weeks: float,
    has_new_lesion: bool,
    has_progressive_disease: bool,
    mixed_response: bool,
    pa_on_file: bool,
) -> dict:
    """Evaluates medical policy criteria for continued coverage.

    Looks up the applicable policy using the patient's ICD10_DX and NDC,
    then evaluates continuation, denial, and pend rules against the
    calculated adherence and improvement data.

    Args:
        patient_id: The patient identifier, e.g. 'PT-001'.
        pdc: Proportion of Days Covered value (0.0 to 1.0).
        adherence_met: Whether PDC >= 0.80.
        improvement_met: Whether objective tumor reduction >= 30%.
        stable_disease: Whether stable disease is documented.
        stable_disease_weeks: Duration of stable disease in weeks (0 if N/A).
        has_new_lesion: Whether a new lesion was detected.
        has_progressive_disease: Whether progressive disease criteria are met.
        mixed_response: Whether response is mixed (improvement + new lesion).
        pa_on_file: Whether prior authorization is valid and on file.

    Returns:
        dict: Policy match result with policy_id, section_ref, criterion_status,
              decision (Approve/Deny/Pend), and detailed rationale.
    """
    claims, _, _, policy_df = get_dataframes()
    patient_claims = claims[claims["PATIENT_ID"] == patient_id]

    if patient_claims.empty:
        return {"status": "error", "error_message": f"No claims found for {patient_id}"}

    first_claim = patient_claims.iloc[0]
    icd10 = first_claim["ICD10_DX"]
    ndc = first_claim["NDC"]

    # Policy lookup: match on ICD10_DX and NDC
    matched_policy = None
    for _, policy_row in policy_df.iterrows():
        applicable_icd = str(policy_row["APPLICABLE_ICD10"])
        applicable_ndc = str(policy_row["APPLICABLE_NDC"])
        if icd10 in applicable_icd and ndc in applicable_ndc:
            matched_policy = policy_row
            break

    if matched_policy is None:
        return {
            "status": "error",
            "error_message": (
                f"No matching policy found for ICD10={icd10}, NDC={ndc}. "
                "Cannot evaluate coverage criteria."
            ),
        }

    policy_id = matched_policy["POLICY_ID"]
    policy_name = matched_policy["POLICY_NAME"]
    continuation_rule = str(matched_policy["CONTINUATION_RULE_TEXT"])
    denial_rule = str(matched_policy["DENIAL_RULE_TEXT"])
    pend_rule = str(matched_policy["PEND_RULE_TEXT"])

    # --- Evaluate denial first (PD overrides everything) ---
    if has_progressive_disease and not mixed_response:
        rationale_parts = []
        rationale_parts.append(
            f"Progressive disease confirmed per {policy_id} {denial_rule.split(':')[0]}."
        )
        if has_new_lesion:
            rationale_parts.append("New lesion detected on imaging.")
        rationale_parts.append("Coverage discontinued per policy denial criteria.")

        return {
            "status": "success",
            "patient_id": patient_id,
            "policy_id": policy_id,
            "policy_name": policy_name,
            "section_ref": denial_rule.split(":")[0].strip() if ":" in denial_rule else "Denial",
            "continuation_rule": continuation_rule,
            "denial_rule": denial_rule,
            "criterion_status": "Not Met",
            "decision": "Deny",
            "rationale": " ".join(rationale_parts),
        }

    # --- Evaluate pend criteria (mixed/equivocal response) ---
    if mixed_response:
        rationale_parts = [
            f"Mixed response detected per {policy_id} {pend_rule.split(':')[0]}.",
            "Primary markers show improvement but new secondary lesion detected.",
            "Case requires Medical Director review per policy pend criteria.",
        ]

        return {
            "status": "success",
            "patient_id": patient_id,
            "policy_id": policy_id,
            "policy_name": policy_name,
            "section_ref": pend_rule.split(":")[0].strip() if ":" in pend_rule else "Pend",
            "continuation_rule": continuation_rule,
            "pend_rule": pend_rule,
            "criterion_status": "Partial",
            "decision": "Pend",
            "rationale": " ".join(rationale_parts),
        }

    # --- Evaluate continuation criteria ---
    criteria_checks = {}
    rationale_parts = []
    section_ref = matched_policy["SECTION_REF"]

    # Check adherence (PDC >= 80%)
    criteria_checks["pdc_met"] = adherence_met
    if adherence_met:
        rationale_parts.append(f"PDC {pdc*100:.1f}% >= 80% threshold: MET.")
    else:
        rationale_parts.append(f"PDC {pdc*100:.1f}% < 80% threshold: NOT MET.")

    # Check objective improvement or stable disease
    is_chemo_policy = "CHEMO" in policy_id.upper()

    if is_chemo_policy:
        # Chemotherapy: requires stable disease >= 8 weeks
        sd_met = stable_disease and (stable_disease_weeks or 0) >= 8
        criteria_checks["response_met"] = sd_met
        if sd_met:
            rationale_parts.append(
                f"Stable disease documented for {stable_disease_weeks} weeks "
                f"(>= 8 weeks required): MET."
            )
        else:
            if stable_disease:
                rationale_parts.append(
                    f"Stable disease documented but only {stable_disease_weeks} weeks "
                    f"(< 8 weeks required): NOT MET."
                )
            else:
                rationale_parts.append(
                    "No stable disease documented as required by chemotherapy policy: NOT MET."
                )
    else:
        # Immunotherapy: requires >= 30% tumor reduction on >= 2 visits
        criteria_checks["response_met"] = improvement_met
        if improvement_met:
            rationale_parts.append(
                "Objective tumor reduction >= 30% confirmed by imaging: MET."
            )
        else:
            rationale_parts.append(
                "Objective tumor reduction < 30% threshold: NOT MET."
            )

    # Check no progressive disease
    criteria_checks["no_pd"] = not has_progressive_disease
    if not has_progressive_disease:
        rationale_parts.append("No progressive disease detected: MET.")
    else:
        rationale_parts.append("Progressive disease signals present: NOT MET.")

    # Check PA on file
    criteria_checks["pa_valid"] = pa_on_file
    if pa_on_file:
        rationale_parts.append("Valid prior authorization on file: MET.")
    else:
        rationale_parts.append("No valid prior authorization on file: NOT MET.")

    all_met = all(criteria_checks.values())
    any_missing = False  # in real system, could detect missing data

    if all_met:
        decision = "Approve"
        criterion_status = "Met"
        rationale_parts.append(
            f"All continuation criteria under {policy_id} {section_ref} satisfied. "
            "Coverage continuation approved."
        )
    elif any_missing:
        decision = "Pend"
        criterion_status = "Partial"
        rationale_parts.append("Insufficient data to determine; pended for review.")
    else:
        decision = "Deny"
        criterion_status = "Not Met"
        failed = [k for k, v in criteria_checks.items() if not v]
        rationale_parts.append(
            f"Continuation criteria not fully met (failed: {', '.join(failed)}). "
            "Coverage denied."
        )

    return {
        "status": "success",
        "patient_id": patient_id,
        "policy_id": policy_id,
        "policy_name": policy_name,
        "section_ref": section_ref,
        "continuation_rule": continuation_rule,
        "denial_rule": denial_rule,
        "pend_rule": pend_rule,
        "criteria_checks": criteria_checks,
        "criterion_status": criterion_status,
        "decision": decision,
        "rationale": " ".join(rationale_parts),
    }

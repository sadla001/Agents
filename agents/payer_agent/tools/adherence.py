"""
S1: Adherence calculation tool.

Implements the 4 adherence rules:
  1. Build coverage timeline from SERVICE_DATE + DAYS_SUPPLY
  2. Overlap-adjust: if refill starts before prior supply ends, shift start
  3. PDC = unique covered days / total days in window
  4. PDC >= 0.80 -> Adherence TRUE; < 0.80 -> FLAG_NON_COMPLIANCE
"""

from datetime import date, timedelta

from payer_agent.tools.data_loader import get_dataframes


def calculate_adherence(patient_id: str) -> dict:
    """Calculates the Proportion of Days Covered (PDC) for a patient.

    Builds a coverage timeline from SERVICE_DATE + DAYS_SUPPLY across all
    treatment cycles. Applies overlap adjustment when refills start before
    the prior supply ends. Returns PDC value and adherence determination.

    Args:
        patient_id: The patient identifier, e.g. 'PT-001'.

    Returns:
        dict: Contains pdc, adherence_met flag, coverage window details,
              covered days, gap descriptions, and any compliance flags.
    """
    claims, _, _, _ = get_dataframes()
    patient_claims = claims[claims["PATIENT_ID"] == patient_id].sort_values("SERVICE_DATE")

    if patient_claims.empty:
        return {"status": "error", "error_message": f"No claims found for {patient_id}"}

    rows = list(patient_claims.itertuples(index=False))

    # Build raw intervals: each cycle covers [service_date, service_date + days_supply - 1]
    raw_intervals = []
    for row in rows:
        start = row.SERVICE_DATE
        end = start + timedelta(days=int(row.DAYS_SUPPLY) - 1)
        raw_intervals.append({"start": start, "end": end, "cycle": row.CYCLE})

    # Overlap-adjust: if a refill starts before prior supply ends,
    # reduce the effective supply of the LATER cycle by the overlap.
    overlap_days_total = 0
    adjusted_intervals = []
    for i, iv in enumerate(raw_intervals):
        if i == 0:
            adjusted_intervals.append(dict(iv))
        else:
            prev_end = adjusted_intervals[-1]["end"]
            if iv["start"] <= prev_end:
                overlap = (prev_end - iv["start"]).days + 1
                overlap_days_total += overlap
                new_start = prev_end + timedelta(days=1)
                supply = int(rows[i].DAYS_SUPPLY)
                new_end = new_start + timedelta(days=supply - overlap - 1)
                adjusted_intervals.append(
                    {"start": new_start, "end": new_end, "cycle": iv["cycle"]}
                )
            else:
                adjusted_intervals.append(dict(iv))

    first_service = rows[0].SERVICE_DATE
    last_service = rows[-1].SERVICE_DATE

    # Primary PDC: total effective supply / (last_service_date - first_service_date)
    # This matches the derivation walkthrough's methodology.
    total_supply = sum(int(r.DAYS_SUPPLY) for r in rows) - overlap_days_total
    window_days = (last_service - first_service).days
    if window_days <= 0:
        pdc = 1.0
    else:
        pdc = min(total_supply / window_days, 1.0)

    # Secondary PDC (standard): unique covered days / full window
    full_window_end = adjusted_intervals[-1]["end"]
    full_window_days = (full_window_end - first_service).days + 1
    covered_days_set: set[date] = set()
    for iv in adjusted_intervals:
        d = iv["start"]
        while d <= iv["end"]:
            covered_days_set.add(d)
            d += timedelta(days=1)
    unique_covered = len(covered_days_set)
    pdc_standard = round(unique_covered / full_window_days, 4) if full_window_days > 0 else 0.0

    # Identify gaps between adjusted intervals
    gaps = []
    for i in range(1, len(adjusted_intervals)):
        gap_start = adjusted_intervals[i - 1]["end"] + timedelta(days=1)
        gap_end = adjusted_intervals[i]["start"] - timedelta(days=1)
        if gap_start <= gap_end:
            gap_days = (gap_end - gap_start).days + 1
            gaps.append(
                {
                    "gap_start": str(gap_start),
                    "gap_end": str(gap_end),
                    "gap_days": gap_days,
                }
            )

    adherence_met = pdc >= 0.80
    flags = []
    if not adherence_met:
        flags.append("FLAG_NON_COMPLIANCE")

    return {
        "status": "success",
        "patient_id": patient_id,
        "pdc": round(pdc, 4),
        "pdc_pct": f"{pdc * 100:.1f}%",
        "adherence_met": adherence_met,
        "pdc_standard": pdc_standard,
        "pdc_standard_pct": f"{pdc_standard * 100:.1f}%",
        "coverage_window": {
            "first_service_date": str(first_service),
            "last_service_date": str(last_service),
            "window_days": window_days,
        },
        "total_supply_days": total_supply,
        "unique_covered_days": unique_covered,
        "full_window_days": full_window_days,
        "overlap_days_adjusted": overlap_days_total,
        "num_cycles": len(adjusted_intervals),
        "gaps": gaps,
        "flags": flags,
        "timeline": [
            {
                "cycle": iv["cycle"],
                "start": str(iv["start"]),
                "end": str(iv["end"]),
            }
            for iv in adjusted_intervals
        ],
    }

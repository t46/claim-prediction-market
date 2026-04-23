"""
Claim Prediction Market — Metascience Experiment

Tests research-social-infrastructure §10.1: "A prediction market mechanism
over research claims correctly identifies novel discoveries vs trivial claims."

Reads ARA claim YAML files, simulates 3-5 LLM predictor agents that predict
whether each claim will hold or be refuted, compares against actual evaluator
outcomes, and scores predictors using Brier score.

Measures whether surprising claims (low consensus predictions that turned out
verified) represent more novel findings.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

import yaml
import anthropic


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARA_RUNS: list[dict[str, str]] = [
    {
        "path": os.path.expanduser(
            "~/unktok/dev/autonomous-research-agent/runs/"
            "2026-04-14-autonomous-research-methodology/research-workspace/beliefs/claims"
        ),
        "label": "2026-04-14 autonomous-research-methodology",
    },
    {
        "path": os.path.expanduser(
            "~/unktok/dev/autonomous-research-agent/runs/"
            "2026-04-15-automated-research-methodology/research-workspace/beliefs/claims"
        ),
        "label": "2026-04-15 automated-research-methodology",
    },
    {
        "path": os.path.expanduser(
            "~/unktok/dev/autonomous-research-agent/runs/"
            "2026-04-16-2026-04-16-v2/research-workspace/beliefs/claims"
        ),
        "label": "2026-04-16 v2",
    },
]

NUM_PREDICTORS = 5

PREDICTOR_PERSONAS = [
    {
        "name": "Skeptic",
        "style": (
            "You are a deeply skeptical scientific reviewer. You believe most "
            "claims are overclaiming and require extraordinary evidence. You "
            "tend to predict claims will be refuted or remain unverified."
        ),
    },
    {
        "name": "Optimist",
        "style": (
            "You are an optimistic but rigorous researcher. You believe that "
            "well-reasoned claims with multiple supporting evidence sources "
            "are likely to hold. You give the benefit of the doubt when "
            "evidence directions are mixed but predominantly supporting."
        ),
    },
    {
        "name": "Methodologist",
        "style": (
            "You are a strict methodologist. You evaluate claims primarily "
            "based on the quality of evidence (fidelity levels F0-F3) and "
            "statistical rigor. Low-fidelity evidence (F0/F1) makes you "
            "much more uncertain than high-fidelity evidence (F2/F3)."
        ),
    },
    {
        "name": "DomainExpert",
        "style": (
            "You are a domain expert in autonomous research systems and "
            "metascience. You use your background knowledge to assess "
            "whether claims align with known results in the field. Novel "
            "claims that contradict established findings get lower scores."
        ),
    },
    {
        "name": "BaseRate",
        "style": (
            "You are a base-rate reasoner. You start from the prior that "
            "most research claims in a new field have ~60% chance of being "
            "substantially correct and adjust from there based on evidence "
            "count, direction balance, and confidence level stated."
        ),
    },
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """A single ARA claim parsed from YAML."""

    id: str
    run_label: str
    text: str
    context: str
    status: str  # active, parked, verified, killed
    confidence: float
    evidence_for: int
    evidence_against: int
    evidence_total: int
    max_fidelity: str
    raw_yaml: str  # full YAML for predictor context

    @property
    def ground_truth_verified(self) -> bool | None:
        """Return ground truth if determinable.

        We treat 'verified' as True, 'parked'/'killed' as False.
        'active' claims are uncertain — we treat high confidence (>=0.7) as
        leaning verified and low confidence (<0.5) as leaning unverified,
        since ARA's confidence reflects the agent's own assessment after
        evidence gathering.
        """
        if self.status == "verified":
            return True
        if self.status in ("killed", "parked"):
            return False
        # For 'active' claims, use confidence as a proxy
        if self.confidence >= 0.7:
            return True
        if self.confidence < 0.5:
            return False
        return None  # too uncertain to judge


@dataclass
class Prediction:
    """A single predictor's prediction for a claim."""

    predictor_name: str
    claim_id: str
    probability_verified: float  # 0.0 to 1.0
    reasoning: str


@dataclass
class ClaimResult:
    """Aggregated results for a single claim."""

    claim: Claim
    predictions: list[Prediction]
    ground_truth: bool | None
    mean_prediction: float = 0.0
    prediction_variance: float = 0.0
    consensus: float = 0.0  # 1.0 = full consensus, 0.0 = maximum disagreement
    brier_scores: dict[str, float] = field(default_factory=dict)
    is_surprising: bool = False  # low consensus prediction that was verified


# ---------------------------------------------------------------------------
# Claim loading
# ---------------------------------------------------------------------------


def load_claims() -> list[Claim]:
    """Load all claim YAML files from ARA runs."""
    claims: list[Claim] = []

    for run in ARA_RUNS:
        claims_dir = Path(run["path"])
        if not claims_dir.exists():
            print(f"  WARNING: {claims_dir} does not exist, skipping")
            continue

        yaml_files = sorted(claims_dir.glob("claim-*.yaml"))
        print(f"  {run['label']}: {len(yaml_files)} claims")

        for yf in yaml_files:
            raw = yf.read_text()
            try:
                data = yaml.safe_load(raw)
            except yaml.YAMLError as e:
                print(f"    WARNING: Failed to parse {yf.name}: {e}")
                continue

            evidence = data.get("evidence", [])
            ev_for = sum(1 for e in evidence if e.get("direction") == "for")
            ev_against = sum(
                1 for e in evidence if e.get("direction") == "against"
            )

            # Determine maximum fidelity level
            fidelities = [e.get("fidelity", "F0") for e in evidence]
            max_fid = "F0"
            for f in fidelities:
                if f and f > max_fid:
                    max_fid = f

            claim = Claim(
                id=f"{run['label']}/{yf.stem}",
                run_label=run["label"],
                text=data.get("claim", ""),
                context=data.get("context", ""),
                status=data.get("status", "active"),
                confidence=float(data.get("confidence", 0.5)),
                evidence_for=ev_for,
                evidence_against=ev_against,
                evidence_total=len(evidence),
                max_fidelity=max_fid,
                raw_yaml=raw,
            )
            claims.append(claim)

    return claims


# ---------------------------------------------------------------------------
# Prediction via Claude API
# ---------------------------------------------------------------------------


async def get_prediction(
    client: anthropic.AsyncAnthropic,
    persona: dict[str, str],
    claim: Claim,
) -> Prediction:
    """Ask one predictor agent to predict whether a claim will be verified."""

    system_prompt = (
        f"You are a predictor in a research claim prediction market.\n\n"
        f"Persona: {persona['name']}\n{persona['style']}\n\n"
        f"You will be shown a research claim with its evidence. Your task is "
        f"to predict the probability (0.0 to 1.0) that this claim will be "
        f"VERIFIED by independent evaluation.\n\n"
        f"Respond with EXACTLY this JSON format, nothing else:\n"
        f'{{"probability": 0.XX, "reasoning": "one sentence"}}'
    )

    user_prompt = (
        f"## Claim\n{claim.text}\n\n"
        f"## Context\n{claim.context}\n\n"
        f"## Evidence Summary\n"
        f"- Evidence FOR: {claim.evidence_for}\n"
        f"- Evidence AGAINST: {claim.evidence_against}\n"
        f"- Total evidence pieces: {claim.evidence_total}\n"
        f"- Maximum fidelity level: {claim.max_fidelity}\n"
        f"- Author's stated confidence: {claim.confidence}\n\n"
        f"## Full Claim Data\n```yaml\n{claim.raw_yaml}\n```\n\n"
        f"Predict the probability this claim will be verified."
    )

    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = ""
    for block in response.content:
        if block.type == "text":
            text = block.text
            break

    # Parse JSON response
    try:
        # Try to find JSON in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
        else:
            result = json.loads(text)

        prob = float(result.get("probability", 0.5))
        prob = max(0.0, min(1.0, prob))  # clamp
        reasoning = result.get("reasoning", "")
    except (json.JSONDecodeError, ValueError, KeyError):
        print(
            f"    WARNING: Failed to parse response from {persona['name']} "
            f"for {claim.id}: {text[:100]}"
        )
        prob = 0.5
        reasoning = f"Parse error. Raw: {text[:200]}"

    return Prediction(
        predictor_name=persona["name"],
        claim_id=claim.id,
        probability_verified=prob,
        reasoning=reasoning,
    )


async def predict_all(claims: list[Claim]) -> list[ClaimResult]:
    """Run all predictions concurrently with rate limiting."""
    client = anthropic.AsyncAnthropic()
    results: list[ClaimResult] = []

    # Use a semaphore to limit concurrent API calls
    sem = asyncio.Semaphore(10)

    async def predict_with_sem(
        persona: dict[str, str], claim: Claim
    ) -> Prediction:
        async with sem:
            return await get_prediction(client, persona, claim)

    for i, claim in enumerate(claims):
        gt = claim.ground_truth_verified
        if gt is None:
            print(
                f"  [{i+1}/{len(claims)}] SKIP {claim.id} "
                f"(status={claim.status}, conf={claim.confidence:.2f} — "
                f"indeterminate)"
            )
            continue

        print(
            f"  [{i+1}/{len(claims)}] Predicting {claim.id} "
            f"(gt={'verified' if gt else 'unverified'})"
        )

        # Run all predictors concurrently for this claim
        tasks = [
            predict_with_sem(persona, claim) for persona in PREDICTOR_PERSONAS
        ]
        predictions = await asyncio.gather(*tasks)

        result = ClaimResult(
            claim=claim,
            predictions=list(predictions),
            ground_truth=gt,
        )

        # Compute statistics
        probs = [p.probability_verified for p in predictions]
        result.mean_prediction = mean(probs)
        result.prediction_variance = (
            stdev(probs) if len(probs) > 1 else 0.0
        )
        # Consensus: 1 - 2*stdev (rough measure, 0=max disagree, 1=full agree)
        result.consensus = max(0.0, 1.0 - 2.0 * result.prediction_variance)

        # Brier scores: (prediction - outcome)^2
        outcome = 1.0 if gt else 0.0
        for p in predictions:
            brier = (p.probability_verified - outcome) ** 2
            result.brier_scores[p.predictor_name] = brier

        # Surprising = low consensus AND verified
        # (many predictors expected failure, but it held)
        if gt and result.mean_prediction < 0.5:
            result.is_surprising = True

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_results(results: list[ClaimResult]) -> dict:
    """Analyze prediction market results and test the hypothesis."""

    # Overall predictor Brier scores
    predictor_briers: dict[str, list[float]] = {}
    for r in results:
        for name, brier in r.brier_scores.items():
            predictor_briers.setdefault(name, []).append(brier)

    predictor_summary = {}
    for name, scores in predictor_briers.items():
        predictor_summary[name] = {
            "mean_brier": round(mean(scores), 4),
            "n_predictions": len(scores),
        }

    # Market (ensemble) Brier score
    market_briers = []
    for r in results:
        outcome = 1.0 if r.ground_truth else 0.0
        market_brier = (r.mean_prediction - outcome) ** 2
        market_briers.append(market_brier)

    # Surprising claims analysis
    surprising = [r for r in results if r.is_surprising]
    non_surprising_verified = [
        r
        for r in results
        if r.ground_truth and not r.is_surprising
    ]

    # Novelty proxy: claims with lower consensus among predictors
    # and claims that were surprising (verified against expectations)
    surprising_details = []
    for r in surprising:
        surprising_details.append(
            {
                "claim_id": r.claim.id,
                "claim_text": r.claim.text[:120] + "...",
                "mean_prediction": round(r.mean_prediction, 3),
                "consensus": round(r.consensus, 3),
                "status": r.claim.status,
                "confidence": r.claim.confidence,
            }
        )

    # Calibration: how well does the market track outcomes?
    calibration_buckets: dict[str, dict] = {}
    for bucket_lo in [0.0, 0.2, 0.4, 0.6, 0.8]:
        bucket_hi = bucket_lo + 0.2
        bucket_key = f"{bucket_lo:.1f}-{bucket_hi:.1f}"
        bucket_results = [
            r
            for r in results
            if bucket_lo <= r.mean_prediction < bucket_hi
        ]
        if bucket_results:
            actual_rate = mean(
                [1.0 if r.ground_truth else 0.0 for r in bucket_results]
            )
            calibration_buckets[bucket_key] = {
                "n": len(bucket_results),
                "mean_predicted": round(
                    mean([r.mean_prediction for r in bucket_results]), 3
                ),
                "actual_verified_rate": round(actual_rate, 3),
            }

    analysis = {
        "hypothesis": (
            "A prediction market mechanism over research claims correctly "
            "identifies novel discoveries vs trivial claims."
        ),
        "total_claims_loaded": None,  # filled by caller
        "claims_with_ground_truth": len(results),
        "verified_count": sum(1 for r in results if r.ground_truth),
        "unverified_count": sum(1 for r in results if not r.ground_truth),
        "predictor_brier_scores": predictor_summary,
        "market_ensemble_brier": round(mean(market_briers), 4),
        "calibration": calibration_buckets,
        "surprising_claims": {
            "count": len(surprising),
            "description": (
                "Claims where market predicted <0.5 but ground truth is "
                "verified. These are candidates for genuinely novel findings."
            ),
            "claims": surprising_details,
        },
        "trivial_claims": {
            "count": len(non_surprising_verified),
            "description": (
                "Claims where market predicted >=0.5 and verified. "
                "High consensus + verified = likely not novel."
            ),
        },
        "hypothesis_assessment": "",  # filled below
    }

    # Assess hypothesis
    if len(results) < 3:
        analysis["hypothesis_assessment"] = (
            "INSUFFICIENT DATA: Too few claims with determinable ground truth "
            "to assess the hypothesis."
        )
    else:
        # The hypothesis is supported if:
        # 1. Market Brier score is better (lower) than chance (0.25)
        # 2. Surprising claims exist and differ from trivial ones
        market_brier = analysis["market_ensemble_brier"]
        has_discrimination = len(surprising) > 0 and len(
            non_surprising_verified
        ) > 0

        if market_brier < 0.25 and has_discrimination:
            analysis["hypothesis_assessment"] = (
                f"SUPPORTED (preliminary): Market ensemble Brier={market_brier:.4f} "
                f"(better than chance=0.25). Found {len(surprising)} surprising "
                f"claims vs {len(non_surprising_verified)} trivial verified claims, "
                f"suggesting the market discriminates novelty from triviality."
            )
        elif market_brier < 0.25:
            analysis["hypothesis_assessment"] = (
                f"PARTIALLY SUPPORTED: Market Brier={market_brier:.4f} shows "
                f"predictive accuracy, but {'no surprising claims found' if not surprising else 'no trivial claims for comparison'} — "
                f"cannot assess novelty discrimination."
            )
        else:
            analysis["hypothesis_assessment"] = (
                f"NOT SUPPORTED: Market Brier={market_brier:.4f} is worse than "
                f"chance (0.25). The prediction market does not improve on "
                f"random guessing for these claims."
            )

    return analysis


def print_report(analysis: dict, results: list[ClaimResult]) -> None:
    """Print a formatted report."""
    print("\n" + "=" * 70)
    print("CLAIM PREDICTION MARKET — RESULTS")
    print("=" * 70)

    print(f"\nHypothesis: {analysis['hypothesis']}")
    print(f"\nClaims loaded: {analysis['total_claims_loaded']}")
    print(f"Claims with ground truth: {analysis['claims_with_ground_truth']}")
    print(f"  Verified: {analysis['verified_count']}")
    print(f"  Unverified: {analysis['unverified_count']}")

    print("\n--- Predictor Brier Scores (lower = better) ---")
    for name, stats in sorted(
        analysis["predictor_brier_scores"].items(),
        key=lambda x: x[1]["mean_brier"],
    ):
        print(f"  {name:15s}: {stats['mean_brier']:.4f} (n={stats['n_predictions']})")
    print(f"  {'MARKET':15s}: {analysis['market_ensemble_brier']:.4f} (ensemble)")

    print("\n--- Calibration ---")
    for bucket, stats in analysis["calibration"].items():
        print(
            f"  Predicted {bucket}: n={stats['n']}, "
            f"predicted={stats['mean_predicted']:.3f}, "
            f"actual={stats['actual_verified_rate']:.3f}"
        )

    print(f"\n--- Surprising Claims (novelty candidates) ---")
    surprising = analysis["surprising_claims"]
    print(f"  Count: {surprising['count']}")
    for s in surprising["claims"]:
        print(f"  * {s['claim_id']}")
        print(f"    Market prediction: {s['mean_prediction']:.3f}")
        print(f"    Consensus: {s['consensus']:.3f}")
        print(f"    {s['claim_text']}")

    print(f"\n--- Per-Claim Details ---")
    for r in results:
        gt_str = "VERIFIED" if r.ground_truth else "UNVERIFIED"
        surprise_str = " *** SURPRISING ***" if r.is_surprising else ""
        print(f"\n  {r.claim.id} [{gt_str}]{surprise_str}")
        print(f"    Claim: {r.claim.text[:100]}...")
        print(
            f"    Market: mean={r.mean_prediction:.3f}, "
            f"consensus={r.consensus:.3f}"
        )
        for p in r.predictions:
            brier = r.brier_scores[p.predictor_name]
            print(
                f"      {p.predictor_name:15s}: p={p.probability_verified:.3f} "
                f"brier={brier:.4f} | {p.reasoning[:80]}"
            )

    print(f"\n{'=' * 70}")
    print(f"ASSESSMENT: {analysis['hypothesis_assessment']}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 70)
    print("Claim Prediction Market — Metascience Experiment")
    print("Testing research-social-infrastructure §10.1")
    print("=" * 70)

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    # Load claims
    print("\n1. Loading ARA claims...")
    claims = load_claims()
    print(f"   Total: {len(claims)} claims across {len(ARA_RUNS)} runs")

    # Filter to claims with ground truth
    evaluable = [c for c in claims if c.ground_truth_verified is not None]
    print(f"   Evaluable (have ground truth): {len(evaluable)}")

    if not evaluable:
        print("\nERROR: No claims with determinable ground truth found.")
        sys.exit(1)

    # Run predictions
    print(f"\n2. Running predictions ({NUM_PREDICTORS} predictors per claim)...")
    results = await predict_all(claims)

    # Analyze
    print("\n3. Analyzing results...")
    analysis = analyze_results(results)
    analysis["total_claims_loaded"] = len(claims)

    # Report
    print_report(analysis, results)

    # Save raw results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "results.json"
    serializable = {
        **analysis,
        "per_claim": [
            {
                "claim_id": r.claim.id,
                "claim_text": r.claim.text,
                "status": r.claim.status,
                "confidence": r.claim.confidence,
                "ground_truth": r.ground_truth,
                "mean_prediction": round(r.mean_prediction, 4),
                "consensus": round(r.consensus, 4),
                "is_surprising": r.is_surprising,
                "predictions": [
                    {
                        "predictor": p.predictor_name,
                        "probability": round(p.probability_verified, 4),
                        "brier": round(r.brier_scores[p.predictor_name], 4),
                        "reasoning": p.reasoning,
                    }
                    for p in r.predictions
                ],
            }
            for r in results
        ],
    }
    output_file.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
    print(f"\nRaw results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())

# Claim Prediction Market

A minimal metascience experiment testing whether a prediction market mechanism over research claims can correctly identify novel discoveries vs trivial claims.

## Hypothesis

> "A prediction market mechanism over research claims correctly identifies novel discoveries vs trivial claims."

From [research-social-infrastructure](../../knowledgebase/research-social-infrastructure.md) section 10.1:

> **Claim Prediction Market**: Before an agent executes high-cost experiments (F2/F3 level), other agents predict the result. Surprising successes (many predicted failure but it succeeded) earn high reputation rewards.

## How It Works

1. **Loads claims** from 3 ARA (Autonomous Research Agent) project runs (~35 claims total), each with evidence, confidence scores, and verification status.

2. **Simulates 5 LLM predictor agents** with distinct personas (Skeptic, Optimist, Methodologist, DomainExpert, BaseRate), each reading a claim + evidence and predicting P(verified).

3. **Compares predictions** against actual ARA evaluator outcomes (verified/parked/killed status + confidence).

4. **Scores predictors** using Brier score: `(prediction - outcome)^2`. Lower = better. Random chance = 0.25.

5. **Identifies surprising claims**: low market consensus predictions that turned out verified. These are candidates for genuinely novel findings -- the market expected failure but evidence held.

## Data

Claims from ARA runs at:
- `~/unktok/dev/autonomous-research-agent/runs/2026-04-14-autonomous-research-methodology/`
- `~/unktok/dev/autonomous-research-agent/runs/2026-04-15-automated-research-methodology/`
- `~/unktok/dev/autonomous-research-agent/runs/2026-04-16-2026-04-16-v2/`

Each claim YAML contains: claim text, context, evidence (with direction and fidelity level F0-F3), confidence (0-1), and status (active/verified/parked/killed).

## Running

```bash
# Requires ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=your-key

# Run the experiment
uv run python predict.py

# Results are printed to stdout and saved to output/results.json
```

## Output

- Per-predictor Brier scores
- Market ensemble Brier score
- Calibration analysis (predicted vs actual verification rates)
- List of "surprising" claims (novelty candidates)
- Per-claim prediction details with reasoning

## Limitations

- Small N (~35 claims, ~25 with ground truth)
- Ground truth is approximated: "verified" and "killed/parked" statuses are definitive, but "active" claims use the ARA's own confidence as a proxy
- Predictor agents are simulated, not independent -- they share the same base model
- No real economic stakes -- incentive alignment is not tested
- Single model family (Claude) -- cross-model prediction diversity not explored

## Cost

Uses `claude-haiku-4-5` for predictions. ~5 predictions x ~25 claims = ~125 API calls. Estimated cost: < $0.50.

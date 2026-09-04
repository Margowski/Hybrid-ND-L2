# ND-L2 Pilot Results

Runs: 5 seeds; training samples per seed: 700; test samples: 250; Gaussian perturbation sigma: 0.08.

| Model | Memory | Context | Self | Relation | Macro clean | Macro noisy | Retention | ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Causal attention | 1.000 ± 0.000 | 0.939 ± 0.018 | 0.725 ± 0.024 | 1.000 ± 0.000 | 0.916 ± 0.009 | 0.764 ± 0.022 | 0.834 ± 0.021 | 0.074 ± 0.009 |
| ND state | 1.000 ± 0.000 | 0.839 ± 0.028 | 0.729 ± 0.033 | 1.000 ± 0.000 | 0.892 ± 0.008 | 0.812 ± 0.009 | 0.911 ± 0.015 | 1.569 ± 0.060 |
| ND-L2 feedback | 1.000 ± 0.000 | 0.858 ± 0.014 | 0.732 ± 0.035 | 1.000 ± 0.000 | 0.898 ± 0.008 | 0.855 ± 0.010 | 0.953 ± 0.005 | 4.905 ± 0.189 |
| ND-L2 learned gate | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.725 ± 0.042 | 1.000 ± 0.000 | 0.931 ± 0.010 | 0.920 ± 0.010 | 0.988 ± 0.005 | 5.759 ± 0.209 |

## Decision rule

The gated follow-up hypothesis is supported only if gated ND-L2 has higher mean macro clean accuracy **and** higher mean noisy accuracy than both baselines. Runtime is reported as a cost and is not folded into accuracy.

## Result

**Verdict: supported in this pilot.** The clean winner is ND-L2 learned gate; the noisy winner is ND-L2 learned gate. ND-L2 encoding is 77.5x the measured attention-baseline runtime in this unoptimized NumPy implementation.

## Scope

This is a fixed-encoder synthetic pilot. It does not establish superiority to a fully trained Transformer and makes no claim about consciousness or subjective emotion.

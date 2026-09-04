# ND-L2 Cognitive State Benchmark

This repository is a **falsifiable first prototype** for comparing three ways of
representing long-lived cognitive state:

1. `CausalAttentionBaseline`: a causal multi-head attention feature encoder.
2. `NDStateModel`: explicit memory, context, affect, self, relation and semantic
   state blocks.
3. `NDL2FeedbackModel`: the same conceptual factors represented as discretized
   square-integrable fields with stable, vector-valued cross-feedback.
4. `NDL2GatedFeedbackModel`: an ablation-compatible extension with a learned
   six-parameter overwrite gate for fast context changes.
5. `NDL2VirtualGatedFeedbackModel`: the gated model with transactional
   shadow-state updates and lazy materialization of memory and relation fields.
6. `NDL2JitVirtualGatedFeedbackModel`: the vectorized virtual model with its
   remaining time recurrence compiled to native CPU code by optional Numba JIT.

It is deliberately small enough to run with NumPy on a CPU. It is **not** a
claim that the third model is conscious, emotional, or superior to a fully
trained production Transformer. The attention baseline and both state models
use fixed encoders and equally sized trained ridge-classification readouts.
This isolates representation and state persistence in a controlled pilot.

## Mathematical definition

Let the factors be

\[
\Psi_t=(M_t,C_t,E_t,I_t,R_t,S_t)\in
\mathcal H_M\oplus\mathcal H_C\oplus\mathcal H_E\oplus
\mathcal H_I\oplus\mathcal H_R\oplus\mathcal H_S.
\]

For the L2 model, each factor is a finite Galerkin approximation of an
`L2` field:

\[
f_j(z,t)=\sum_{k=0}^{K_j-1} a_{j,k}(t)\phi_{j,k}(z),
\qquad a_j(t)\in\mathbb R^{K_j\times d_j}.
\]

The state update is

\[
\Psi_{t+1}=\mathcal D\Psi_t+\mathcal Bx_t+
\sigma\!\left(\mathcal K(\Psi_t)\Psi_t\right),
\]

where `D` is a contractive diffusion/decay operator and `K` is a block
operator matrix. Its entries map one field into another, so feedback is a
vector-valued operator rather than a scalar reward. Stability is enforced by
contractive coefficients and clipping in this finite pilot.

For the gated follow-up, the context field is updated by

\[
C_{t+1}=(1-g_t)\mathcal D C_t+g_t\,B_Cx_t,
\qquad g_t=\sigma(w^\top\varphi(x_t)).
\]

The six coefficients in `w` are fitted from token-local context-write
indicators in the training sequences. This tests whether learned selective
overwrite fixes the stability/plasticity conflict. It does not yet test
unsupervised discovery of semantic context shifts.

## Virtual state and commit semantics

The virtual model retains the projections required by nonlinear feedback, but
stores full memory and relation fields as time-local injection deltas. For the
linear eager update

\[
z_t=q_t(Az_{t-1}+u_t),
\]

it materializes the algebraically equivalent state only after a successful
sequence:

\[
z_T=\sum_{\tau=1}^{T}
\left(\prod_{k=\tau}^{T}q_k\right)A^{T-\tau}u_\tau.
\]

The virtual state is committed only after materialization and envelope checks.
If computation fails, the previous commit remains unchanged. This is exact for
the pilot while the eager clipping envelope is inactive; the implementation
raises rather than silently diverging if that safety condition is violated.

## Execution recommendation

The preferred production path for this pilot is:

1. batch-vectorized virtual state;
2. Numba JIT for a long-lived process, warmed once at startup;
3. CPU execution for the present state size.

The JIT module is optional: install it with `pip install -e '.[jit]'`. JIT has
a measured cold-start cost, whereas the warm path preserves the reference
features to numerical tolerance. A GPU is not recommended at this sequence
length and state size because transfer and launch overhead would dominate.

The benchmark tests the preregistered pilot hypothesis:

> Multidimensional L2 feedback improves joint state coherence and robustness
> relative to equal-width fixed-feature baselines, while incurring measurable
> runtime cost.

## Tasks and metrics

Every synthetic sequence contains keyed facts, keyed relations, context
changes, affect events, distractors and a final query. Four independently
scored targets are generated from the latent sequence:

- memory persistence (retrieve an early keyed value),
- context coherence (retain the last context),
- self-state consistency (integrate signed affect events),
- relational consistency (retrieve a keyed relation).

Robustness is the macro accuracy under Gaussian input perturbation. Runtime is
encoder wall-clock time per sample. All models expose 192 features and have the
same 3,667 trained readout parameters.

## Run

```bash
python -m nd_l2_benchmark.run --seeds 0 1 2 3 4 --output results
python -m unittest discover -s tests -v
```

From the repository root, either install with `pip install -e .` or set
`PYTHONPATH=src`.

## Interpretation limits

- Synthetic tasks test mechanisms, not general language intelligence.
- Fixed encoders do not constitute a full end-to-end trained Transformer.
- A positive result is a reason for a larger learned benchmark, not proof of
  an information-theoretic advantage.
- Multiple seeds quantify initialization variance, not dataset-domain shift.

## Hybrid LLM integration

`nd_l2_benchmark.hybrid` implements the first secure integration boundary:
one large LLM call returns both a user answer and a semantic event; the local
state engine accepts or rejects the resulting virtual update and commits only
after validation. A GitHub Actions workflow is included at
`.github/workflows/hybrid_turn.yml`. It reads `OPENAI_API_KEY` only from the
GitHub Actions secret and uploads the updated `.hybrid/state.json` as an
artifact; it never commits or prints a secret.

This is an integration prototype. The semantic event currently comes from the
large LLM in the same call as the answer, avoiding a second latency-heavy call.
The next benchmark replaces it with a trained lightweight semantic encoder and
tests hidden semantic context changes.

## Hybrid LLM integration

`nd_l2_benchmark.hybrid` implements the first secure integration boundary:
one large LLM call returns both a user answer and a semantic event; the local
state engine accepts or rejects the resulting virtual update and commits only
after validation. A GitHub Actions workflow is included at
`.github/workflows/hybrid_turn.yml`. It reads `OPENAI_API_KEY` only from the
GitHub Actions secret and uploads the updated `.hybrid/state.json` as an
artifact; it never commits or prints a secret.

This is an integration prototype. The semantic event currently comes from the
large LLM in the same call as the answer, avoiding a second latency-heavy call.
The next benchmark replaces it with a trained lightweight semantic encoder and
tests hidden semantic context changes.

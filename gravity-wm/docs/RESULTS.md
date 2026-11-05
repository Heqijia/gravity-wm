# Experiment log: gravity-constrained post-training

Running record of what has been measured, under what settings, and what the numbers do and do not support.  `design.md` is the companion document: it records every deviation from `WM_gravity` / `WM_gravity_theory` and the measurement that forced it.  This file records the *results*.

A Chinese version of this document is kept in sync at [`RESULTS.zh.md`](RESULTS.zh.md) -- same sections, same numbers.

New findings are appended to [§10 Update log](#10-update-log) with a date, so the body stays a clean statement of the current best configuration and the history stays auditable.

Last full suite: **2026-11-02**, 256 jobs, all exit 0. Artifacts: `/fs/scratch/PAS3151/qijia387/gwm/runs/suite/`, tables in `.../suite/report/`.

---

## 1. The claim under test

Minimise the JEPA loss subject to a free-fall constraint evaluated on rollouts under **counterfactual** gravities for which no video exists:

$$\min_{\theta}\; L(\theta)\quad \text{s.t.} \quadC(\theta) \;=\; \mathbb{E}_{g' \sim U(G)}\!\left[\, c(\theta; g') \,\right] \;\le\; \varepsilon$$

$c$ is measured by decoding the predicted latents with a *frozen* position probe $h$ and comparing the decoded motion against the free fall that $g'$ implies.  Algorithm 1 alternates RESTORE (descend $C$) and OPTIMISE (descend $L$), with a one-sided PCGrad projection when the two conflict near the boundary.

Three things have to be true for this to be worth anything, and each is a separate experiment:

| # | claim | where it is tested |
|---|---|---|
| C1 | the constraint improves counterfactual prediction | §6.1 headline |
| C2 | it does not destroy the world model | `JEPA L` column throughout |
| C3 | the returned iterate carries Theorem 1's feasibility certificate, which a penalty does not | §6.3 |

C3 is the one that distinguishes the method from a penalty. C1 alone does not: a tuned penalty gets to the same accuracy (§6.3).

---

## 2. Experimental policy

The decisions below are deliberate and most of them cost real experiments to establish. They are listed because changing one silently invalidates the comparison.

**The published planar datasets are not used.** Their windows contain contacts, which every hypothesis in the notes excludes; and at $\mathrm{fps} = 16$ in a view 10.4 world units tall, one step of gravity displaces the object by **0.19 pixel** — below any probe accuracy, so the constraint would be pure noise. `gwm/freefall_data.py` generates the contact-free regime the theory assumes, with closed-form kinematics (free-fall residual `1.5e-14`, against `mj_step`'s `5e-3` drift).

**The read-out is changed from the published model, and reported as such.** With the published `[CLS]` read-out the latent does not carry object position (probe error 33.4 px = chance), so the constraint has no margin to work with. `--readout spatial_map` keeps the patch-spatial axis and reaches 0.18 px. The `[CLS]` negative result is *run*, not asserted (§6.5).

**Two independent probes.** $h$ defines the constraint and $h_{\mathrm{eval}}$ measures it, trained on disjoint episode sets (`episode_overlap_between_probes = 0`). Splits are by episode, never by frame — a frame-level split leaks.

**$\varepsilon$ is calibrated on the predictor's own factual rollout, not on real latents.** The `real_latents` tolerance is **unattainable**: the predictor's factual rollout error is 43x the probe noise, so `I_O = empty` and Theorem 1 is vacuous. Measured consequence of using it anyway: the world model is traded away entirely (JEPA L 63.5, worse than the pretrained 50.9) for no certificate. See §6.4 `ab_eps_real`.

**$\varepsilon$ is calibrated on the recorded training-gravity episode set only**, by `heldout_train_gravity_source_range`, not by a gravity filter — a filter also admits the nearby points of the wide held-out grid, whose contexts are not training-distribution windows. Using the whole held-out split inflated $\varepsilon$ 1.7x and left the constraint **never activated** (50 RESTORE steps in 4000).

**The counterfactual gravity filter rejects, never resamples.** Resampling trains on an easier distribution than evaluation (acceptance 1.000 vs 0.835–0.999), which breaks Corollary 2's $\kappa = 1$. `max_resamples = 0`.

**The constraint is in Proposition 1's position form, not Definition 1's acceleration form.** This is the single largest effect in the whole study and it is measured two ways (§6.2, `design.md §"Definition 1 is the wrong objective to descend"`): Definition 1 is blind to the sign structure that determines position error (525x discrepancy on matched perturbations) and is 3–4 orders of magnitude worse conditioned. The position form needs no extra information — the implied velocity $v = (p_t - p_{t-1})/\mathrm{d}t + g\,\mathrm{d}t/2$ uses two *context* positions only, no future frames.

**`feas` is checked in the form the run was trained in.** Both forms share a zero set and nothing else (they differ by $\sim \mathrm{d}t^{-4}$). The form now comes from the run's own recorded command and a contradicting flag is an error.

**Averaging is across seeds of one configuration, never across configurations.** A tolerance sweep and a penalty-weight sweep are different experiments.

**The reported iterate is $\theta_{n^*}$, the certified one, not $\theta_N$.**

---

## 3. Datasets

Generated by `gwm/freefall_data.py`. Ball, 128x128 px, 9.846 px per world unit, view 13.0 units tall, safe box $x \in [-5.25, 5.25]$, $z \in [1.25, 11.75]$. Actions are `(impulse_x, impulse_z, g)`; gravity enters as an action coordinate. 17 344 episodes each: 16 000 train, 800 held-out *at the training gravity* (the contexts the counterfactual protocol needs), the rest a wide held-out gravity grid.

| dataset | $\mathrm{d}t$ | frames | train gravity | $g\,\mathrm{d}t^2$ (px) | use |
|---|---|---|---|---|---|
| `ball_v1` | 0.16 | 16 | $\mathcal{N}(4, 0.5^2)$ | 1.01 | main |
| `ball_v2` | 0.22 | 12 | $\mathcal{N}(4, 0.5^2)$ | 1.91 | replication at coarser sampling |
| `ball_narrow` | 0.16 | 16 | $\mathcal{N}(4, 0.05^2)$ | 1.01 | the regime Remark 11 needs |

Evaluation gravity grid $g' \in [1, 9]$ step 0.5. Free-fall residual of the generator: `1.47e-14`. Rejection sampling acceptance 1.000 (`rise` ranges are solved analytically so every episode stays in frame).

`ball_narrow` exists because on $\mathcal{N}(4, 0.5^2)$ the pretrained Semigroup-JEPA predictor **already extrapolates gravity** — its counterfactual violation is only 20% above its factual one (25.1 vs 20.9) where a gravity-blind predictor would sit at 21.9 — so there is little for the constraint to repair and Theorem 2 has nothing to separate.

---

## 4. Models

Pretraining is upstream Semigroup-JEPA, unmodified except for the read-out (`scripts/pretrain.py` injects the builder into `sg_jepa.training`; the loop, validation, checkpointing and strict-reload check are upstream code).

`configs/ball_sg_jepa_gru.yaml`: ViT-tiny encoder, patch 8, `embed_dim` 256, `history_size` 8, GRU predictor (hidden 512, depth 3, `film_concat` conditioning, dropout 0.1, residual init 0.1), projector hidden 2048. Objective: rollout-only (`prediction_weight` 0, `rollout_weight` 1), `rollout_horizon` 6, `rollout_gamma` 0.95, `sigreg_weight` 0.09. Training: 40 epochs, batch 32, AdamW `5e-4` + Muon `1e-3`, weight decay `1e-3`, warmup-cosine-cooldown, grad clip 1.0, bf16.

| model | base | read-out | $h$ (px) | $h_{\mathrm{eval}}$ (px) | SNR | margin m=1 | margin m=3 |
|---|---|---|---|---|---|---|---|
| `spatial_s42` | GRU, seed 42 | `spatial_map` | 0.187 | 0.184 | 5.49 | 14.4 | 452.9 |
| `spatial_s43` | GRU, seed 43 | `spatial_map` | 0.178 | 0.155 | 6.50 | 18.8 | 547.0 |
| `lewm1step_s42` | one-step LeWM objective | `spatial_map` | 0.175 | 0.177 | 5.69 | 19.1 | 504.4 |
| `ssm_s42` | SSM predictor | `spatial_map` | 0.190 | 0.183 | 5.52 | 15.5 | 405.6 |
| `narrow_s42` | GRU on `ball_narrow` | `spatial_map` | 0.119 | 0.165 | 6.12 | 28.1 | 877.6 |
| `v2_s42` | GRU on `ball_v2` | `spatial_map` | 0.075 | 0.125 | 15.28 | 147.9 | 6036.1 |
| `cls_s42` | GRU, published read-out | `cls` | 33.429 | 33.398 | 0.03 | **0.004** | **0.12** |

`margin = c_blind / eps` is the discriminative margin: how much larger a gravity-blind predictor's violation is than the tolerance. It is the signal the constraint has to work with and it needs to be `>= ~3`. The `cls` row is why the read-out had to change; the stride column is why Lemma 1 was generalised to strided differences ($m^2$ amplification). **That generalisation has never been exercised in training — see §6.13: `--stride` is recorded but unused in `posttrain.py`, and the $m = 2, 3, 4$ checkpoints are bit-identical.**

---

## 5. Post-training parameters

`scripts/posttrain.py`. Trainable: `predictor`, `pred_proj`, `action_encoder`; encoder and both probes frozen. Frozen modules in `eval()`, trainable in `train()` with all BatchNorm/Dropout pinned to `eval` (cuDNN RNN backward requires train mode).

| parameter | value | note |
|---|---|---|
| `steps` | 4000 | |
| `lr` | per model | `1e-4` (GRU), `1.6e-5` (LeWM-one-step), `1.4e-6` (SSM) -- matched by measured gradient scale, see below |
| `batch_size` / `constraint_batch_size` | 192 / 384 | |
| `history` / `horizon` / `stride` | 8 / 6 / 1 | `v2`: 6 / 6 / 1 |
| `constraint_form` | `position` | Proposition 1; §6.2 |
| `epsilon_source` | `factual_rollout` | §2 |
| $\rho$ | 0.25 | $\varepsilon = (1 + \rho)\, c_{\text{real}}$ |
| `gravity_low` / `high` | 1.0 / 9.0 | $g' \sim U(G)$ |
| `gate` ($\gamma$) | **0.9** | PDF says 0.5, which violates Remark 1's $(\gamma - \zeta_C)(1 + \rho) > 1$ |
| `exit_factor` | **0.9** | PDF's 0.8 equals `E[c_real]` exactly at `rho = 0.25` |
| `restore_direction` | **`polyak`** | PDF's fixed $\nabla C/\varepsilon$ **diverges**; §6.4 |
| `restore_target_factor` | 0.8 | Polyak step targets the level set $\{C = 0.8\varepsilon\}$ |
| `ema_decay` | 0.9 | $\hat{C}$ for mode selection |
| `momentum` | 0.0 | |
| `grad_clip` | `1e4` | safety net, inactive by design; clipped fraction is reported |
| `in_frame_filter` | on | reject, never resample |
| `prime_batches` | 8 | consumed by *every* variant so batch sequences match |
| `seed` | 42, 43 | |

Deviations from the PDFs are in bold and each is justified by a measurement in `design.md`. $c$ is also **squared** relative to the method note, which the theory note itself observes is non-differentiable at $a = \bar{g}$.

### Learning rate is per model, by measured gradient scale

Post-training takes a plain SGD step, so the step *length* is `lr ||grad L||` and one learning rate does not serve every predictor.  Measured at $\theta_0$ by `scripts/measure_gradient_scale.py` (24 batches, median):

| model | `L(theta_0)` | median $\lVert \nabla L \rVert$ | step at `lr = 1e-4` | `lr` for step 0.045 |
|---|---|---|---|---|
| `spatial_s42` (GRU) | 48.6 | 556 | 0.056 | `8.1e-5` |
| `lewm1step_s42` | 392.6 | 2 858 | 0.286 | `1.6e-5` |
| `ssm_s42` | 8.26 | **32 768** | **3.28** | **`1.4e-6`** |

The GRU's `lr = 1e-4` is within 20% of the rule, which is why it was never noticed.  The SSM's gradient is **59x** the GRU's; at `lr = 1e-4` its step is 3.28, and the `1e4` clip engaged at step 0, silently turning the analysed step into a normalised one of length 1.0 -- exactly the failure mode `design.md` warns about for `grad_clip`.  Continued JEPA training *alone* then diverges on it ($L$ 8.7 -> 1320 in 200 steps), so neither the acceleration-form nor the position-form "SSM failure" says anything about the constraint.

### The five variants

| variant | what it does |
|---|---|
| `pretrained` | $\theta_0$, no post-training — the floor |
| `jepa` | continued JEPA training only — the "did the constraint do anything" control |
| `factual` | the constraint at $g' = g$ — isolates *counterfactual* evaluation |
| `penalty` | `L + lambda C`, fixed $\lambda$ — the method it has to beat |
| `counterfactual` | the full method |

`factual` is the sharpest control: same constraint, same machinery, evaluated only where video exists.

---

## 6. Results

Columns throughout: `pos in` = counterfactual position RMSE at the final rollout step for $|g' - g| \le 2\sigma$; `pos OOD` = $> 4\sigma$; **`pos far`** = $g' \ge g + 3$, the band where the pretrained response saturates; `resp far` = realised gravity response as a fraction of the exact one (1 correct, 0 gravity-blind); `slope` = px per unit $|g' - g|$, the quantity Theorem 2 separates; `JEPA L` = rollout loss, sum form; `feas` = Theorem 1's certificate. All positions in pixels.

`pos far` is reported separately because the pretrained predictor's gravity response **saturates rather than being absent** — ratio 0.94 at $g' = 1$ falling to 0.47 at $g' = 9$. A symmetric $|g' - g| > 2$ band mixes the half it handles with the half it does not and hides exactly what the constraint repairs.

### The headline: the separation grows as the data shows less gravity

The three datasets differ in one thing, the spread of the training gravity, and therefore in how much the object's motion varies with it.  Measured as pixels of vertical displacement at the end of a six-step rollout, one standard deviation of the training gravity moves the object 2.27 px on `ball_v1`, 0.23 px on `ball_narrow` -- against a probe accuracy of 0.17 px -- and exactly nothing on `ball_single`.

| regime | 1 sd of train g, in px at h=6 | pretrained resp | continued JEPA | full method | gain | slope |
|---|---|---|---|---|---|---|
| `ball_v1`, $g \sim \mathcal{N}(4, 0.5^2)$ | 2.27 | +0.640 | 6.386 px | **4.489 px** | **1.4x** | +1.05 -> +0.63 |
| `ball_narrow`, $g \sim \mathcal{N}(4, 0.05^2)$ | 0.23 | -0.385 | 27.059 px | **2.992 px** | **9.0x** | +6.50 -> +0.59 |
| `ball_single`, `g == 4` exactly | 0.00 | -0.451 | 28.097 px | **2.716 px** | **10.3x** | +6.36 -> +0.46 |

Two seeds each.  Read it in two directions.

Down the `pretrained resp` column: where the training gravity varies enough to be visible above the probe noise, Semigroup-JEPA **already extrapolates** (+0.640) and a constraint has little to add.  Where it does not, the pretrained predictor responds with the **wrong sign** (-0.385, -0.451): conditioned on a larger gravity it moves the object *up*.  Continued JEPA training does not repair that -- it cannot, because the data contains no evidence about gravity -- and neither does the constraint evaluated at $g' = g$.  Only the counterfactual constraint does.  This is Remark 11's regime, and it is where the method is for.

Across the `full method` column: **2.72, 2.99, 4.49 px** -- the method lands in the same place whatever the data showed, while the baseline degrades from 6.4 to 28 px.  That asymmetry is Theorem 2's content.  The theorem bounds the constrained predictor's counterfactual error by a quantity that does not depend on the gravity shift, and bounds a gravity-blind predictor's from *below* by one that grows with it; a method whose absolute error is flat across three regimes in which the baseline's rises 4.4x is that statement, measured.

The honest qualification is the first row.  On the paper's own training distribution the gain is 1.4x and a swept penalty beats the method's frontier (§6.3), so the case does not rest on `ball_v1`.  It rests on what happens when the data stops carrying the physics.

Regenerate with `results_tables.py --section regimes`.

### 6.1 The five variants on `ball_v1`, position form, 2 seeds

`spatial_s42` + `spatial_s43`, $n = 2$.

| configuration | pos in | pos OOD | **pos far** | **resp far** | **slope** | JEPA L | feas |
|---|---|---|---|---|---|---|---|
| pretrained $\theta_0$ | 4.219 | 7.980 | 9.344 | 0.640 | +1.643 | 50.850 | – |
| continued JEPA training | 3.203 | 5.655 | 6.386 | 0.809 | +1.047 | **19.732** | no |
| constraint at $g' = g$ | 3.083 | 5.688 | 6.503 | 0.805 | +1.124 | 16.537 | no |
| penalty $\lambda = 1$ | 3.189 | 5.631 | 6.430 | 0.793 | +1.067 | 18.134 | no |
| **full method** | **2.756** | **4.207** | **4.489** | **0.850** | **+0.631** | 20.926 | **yes** |

The full method is **1.42x** better than continued JEPA training in the far band and cuts the Theorem-2 slope by **40%**, on both seeds, at a JEPA loss within 6% of it. `factual` and `penalty(1)` are indistinguishable from continued training — so the gain comes specifically from *counterfactual* evaluation, not from having a constraint.

### 6.2 The constraint form is the decisive design choice

Same models, same tolerance, same everything — only `constraint_form` differs.

| form | pos far | resp far | slope | JEPA L |
|---|---|---|---|---|
| Definition 1 (acceleration) | 6.846 | 0.723 | +1.279 | 16.610 |
| **Proposition 1 (position)** | **4.489** | **0.850** | **+0.631** | 20.926 |
| *(continued JEPA, for reference)* | *6.386* | *0.809* | *+1.047* | *19.732* |

In Definition 1's form the full method is **worse than doing nothing**. Two independent mechanisms, both measured:

1. **Sign blindness.** Lemma 7 makes position error a signed, time-weighted sum of residuals; $c$ averages their squares. At matched position cost, a systematic residual ($a \mathrel{+}= 0.6$) scores 0.315 and i.i.d. noise ($\mathrm{sd} = 0.12$) scores 165.4 — a **525x** discrepancy. Driving $C$ from 57.7 to 19.2 cut residual RMS 7.59 -> 4.38 while the bias barely moved 3.86 -> 3.31, and terminal position error got *worse*, 11.2 -> 15.2 px.
2. **Conditioning.** Optimising decoded positions directly: Definition 1 is stable only to `lr = 2e-4` (diverges at `5e-4`) and reaches `3.8e-2` of the starting distance in 4000 steps; the position form is stable to `lr = 3e-1` and reaches `1e-5`. The second difference squares the integral's already-poor conditioning; the position form inverts it analytically.

### 6.3 The certificate: what separates the method from a penalty

$C/\varepsilon$ at the returned iterate, measured in the run's own constraint form.

| configuration | $C/\varepsilon$ s42 | $C/\varepsilon$ s43 | certified | pos far | JEPA L |
|---|---|---|---|---|---|
| continued JEPA training | 1.58 | 1.49 | no | 5.850 | 20.985 |
| constraint at $g' = g$ | 1.57 | 1.54 | no | 5.634 | 15.732 |
| penalty $\lambda = 1$ | 1.43 | 1.55 | no | 5.471 | 16.270 |
| penalty $\lambda = 10$ | 1.27 | – | no | 5.581 | 19.080 |
| penalty $\lambda = 100$ | 0.88 | – | yes | 4.329 | 20.831 |
| **full method, $\varepsilon$** | **0.85** | **0.95** | **yes** | 4.448 | 19.122 |
| full method, $\varepsilon \times 0.8$ | 0.92 | – | yes | 4.317 | 22.443 |
| full method, $\varepsilon \times 0.6$ | 0.96 | – | yes | **3.227** | 27.711 |
| full method, $\varepsilon \times 0.45$ | no certified iterate | – | no | 2.735 | 71.265 |
| full method, $\varepsilon \times 0.3$ | no certified iterate | – | no | 3.356 | 69.392 |

Read this table as a statement about **what the two methods certify**, not about accuracy:

- The method lands inside whatever tolerance it is given — 0.85, 0.92, 0.96 of $\varepsilon$ at three tolerances — and **declines to certify** when the tolerance is below what the predictor can attain ($\varepsilon \times 0.45$ and below, where the calibration's own attainability gap already predicts `I_O = empty`). That is Theorem 1 behaving as advertised.
- The penalty reaches the same place only at $\lambda = 100$, and the only way to find that out was to sweep $\lambda$ over two orders of magnitude and measure $C$ afterwards.
- **At matched feasibility the accuracies are within noise** (penalty 4.329 vs method 4.448). The honest claim is the certificate and the frontier, not a win on accuracy at the default tolerance.

The method's frontier extends to 3.227 px / response 0.906 at $\varepsilon \times 0.6$, which no tried $\lambda$ reaches. Whether a larger $\lambda$ gets there is still being measured (§8).

### 6.4 Ablations of the method's own knobs

`spatial_s42`, position form, otherwise default. $n = 1$.

| ablation | pos far | resp far | slope | JEPA L | feas |
|---|---|---|---|---|---|
| default | 4.448 | 0.832 | +0.663 | 19.122 | yes |
| `gamma = 0.5` (the PDF's value) | 4.208 | 0.858 | +0.570 | 19.085 | yes |
| no hysteresis | 4.410 | 0.842 | +0.614 | 18.211 | yes |
| no projection | 4.214 | 0.863 | +0.584 | 21.335 | yes |
| no in-frame filter | 3.710 | 0.898 | +0.433 | 21.027 | yes |
| RESTORE normalised | 4.626 | 0.838 | +0.700 | 27.233 | yes |
| **RESTORE = $\nabla C/\varepsilon$** (the PDF's) | **19.496** | **0.166** | +1.883 | **22581.6** | no |
| **$\varepsilon$ from `real_latents`** | 3.001 | 0.904 | +0.248 | **63.458** | no |

Two decisive, four that do not matter:

- The PDF's fixed $\nabla C/\varepsilon$ RESTORE step **collapses**: JEPA loss 22582, response 0.166. This is the full-scale confirmation of the divergence that motivated the Polyak step ($\lVert \nabla C \rVert$ grew 209 -> 533, $C$ rose 0.18 -> 2.33, $L$ 0.5 -> 403 in the small-scale run).
- `real_latents` tolerance is unattainable and trades the entire world model (63.5, worse than the pretrained 50.9) for no certificate.
- $\gamma$, hysteresis, projection and the in-frame filter all land within 0.7 px of default and all certify. **At this tolerance they do not matter** — a negative result, reported as one. The in-frame filter costs 0.74 px, which is the price of Corollary 2's $\kappa = 1$.

### 6.5 Base models other than the GRU

**In progress** — these were only ever run in the acceleration form, which §6.2 shows is the wrong objective, so the numbers below say nothing about the method. Rerunning all five in the position form; see §8.

Acceleration form, $n = 1$, for the record:

| base model | best counterfactual pos far | vs pretrained | note |
|---|---|---|---|
| `lewm1step_s42` | 31.454 | **worse** | acceleration form |
| `ssm_s42` | 44.299 | worse, `JEPA L = inf` | **step size, not the method** -- see §5 |
| `narrow_s42` | 11.713 | 2.31x better | acceleration form |
| `cls_s42` | 50.108 | 10.1x better, but probe is at chance | see below |
| `v2_s42` | – | report regenerating | label mismatch, data intact |

`cls_s42` is the negative result the read-out section predicts and it is worth reading carefully: every number is ~50 px because the probe is at chance (33.4 px) and the margin is 0.004. The "10.1x better than pretrained" is the post-training repairing a predictor whose decoded positions were meaningless to begin with (506 px), not the constraint doing physics. **The method requires a read-out that carries position**, and this row is the measurement that says so.

### 6.6 Error against rollout horizon

The evaluation is multi-step throughout: an autoregressive rollout of $H$ steps from an 8-frame context, and every table above reports the error at the **final** step. Each sweep row also records `position_rmse_by_step`, so the per-step curve needs no extra run.

`narrow_s42`, far band $g' \ge g+3$, position RMSE in px:

| variant | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 | growth |
|---|---|---|---|---|---|---|---|
| pretrained $\theta_0$ | 2.26 | 3.79 | 6.43 | 11.16 | 18.07 | 27.11 | 12.0x |
| continued JEPA training | 2.18 | 3.86 | 6.73 | 11.68 | 18.81 | 28.05 | 12.9x |
| constraint at $g' = g$ | 2.17 | 3.87 | 6.78 | 11.79 | 18.94 | 28.21 | 13.0x |
| penalty $\lambda = 1$ | 1.15 | 1.58 | 2.20 | 3.65 | 6.28 | 10.53 | 9.2x |
| **full method** | 1.18 | 1.67 | **1.97** | **2.33** | **2.74** | **4.36** | **3.7x** |

This is a stronger statement than any single final-step number, and it is the one the theory actually makes. Corollary 2 bounds the position error by $(2s+1)\delta + \mathrm{d}t^2 \sqrt{H s(s+1)(2s+1)/6}\,\sqrt{C}$, in which $C$ multiplies the term that *grows with the step index*. So reducing $C$ should flatten the curve rather than shift it down, and it does: the growth factor falls from 12.9x to 3.7x, and the advantage over continued JEPA training compounds from 1.85x at $h = 1$ to **6.4x at $h = 6$**.

It also places the comparison with the penalty precisely. At $h = 1$ and $h = 2$ the penalty is marginally *better* (1.15 / 1.58 against 1.18 / 1.67); from $h = 3$ the method pulls away and ends 2.4x ahead. **The method's benefit is specifically in long-horizon error accumulation, not in one-step accuracy** -- which is the property a world model is used for.

The near band shows the same shape (growth 2.4x against 5.3x), so this is not only an out-of-distribution effect, and the latent error improves too (15.45 -> 14.18 at $h = 6$), so it is not bought by degrading the world model.

Regenerate with `results_tables.py --section horizon [--horizon-model M]`.

### 6.7 The step budget a model needs is predictable

Three failures that looked unrelated -- the SSM predictor diverging, the rotating square paying 6.3x on the JEPA loss, and the certificate never appearing in the headline regime -- have one cause, and it is computable before training starts.

| model | $\varepsilon$ | $C_0$ | $C_0/\varepsilon$ | RESTORE | OPTIMISE | $L$ start | $L$ end | ratio |
|---|---|---|---|---|---|---|---|---|
| `spatial_s42` | 0.0668 | 0.133 | 2.0 | 2402 | 1598 | 48.79 | 25.20 | 0.52x |
| `narrow_s42` | 0.0517 | 0.907 | 17.5 | 1767 | 2233 | 50.42 | 11.42 | 0.23x |
| `narrowfx_s42` | 0.0255 | 0.687 | 26.9 | 3240 | 760 | 14.38 | 9.94 | 0.69x |
| `ssm_s42` | 0.0182 | 1.596 | 87.5 | **4000** | **0** | 8.72 | **810.37** | **92.89x** |
| `ax_s42` | 0.0141 | 1.489 | 105.7 | **4000** | **0** | 13.37 | **1183449** | **88534x** |
| `single_s42` | 0.0137 | 0.755 | 55.0 | 3708 | 292 | 14.58 | 13.45 | 0.92x |
| `sq_single_s42` | 0.0082 | 0.831 | 100.9 | **4000** | **0** | 11.45 | **72.35** | **6.32x** |

Correlation of $\log \varepsilon$ with $\log(C_0/\varepsilon)$: **-0.891** (n = 7).

**The three runs with zero OPTIMISE steps are exactly the three whose JEPA loss blew up.** Every run with at least 292 of them ended at or below its starting loss. Nothing else in the table predicts the outcome: not $\varepsilon$, not $C_0$, not the architecture, not the dataset.

The mechanism is a property of how $\varepsilon$ is calibrated. It is `(1 + rho)` times the predictor's *own* factual rollout violation, while $C_0$ is set by how wrong its *counterfactual* is -- a quantity that depends on the physics the data omits, not on model quality. So a predictor that is better factually gets a **tighter** tolerance at an unchanged $C_0$, and therefore starts relatively further outside it. That is the -0.891 correlation. `sq_single_s42` has the best probe of any model here (0.072 px on the training split), which is exactly why its $\varepsilon$ is the tightest and its $C_0/\varepsilon$ the largest.

Two consequences follow, and both are now acted on rather than discovered the hard way.

**A run that never reaches OPTIMISE cannot certify, by construction.** Theorem 1(c) reads the stationarity certificate off an OPTIMISE iterate, so C3 is unavailable in those runs no matter how small $C$ becomes -- `sq_single_s42` drove $C$ from 0.831 to 0.009, within 1.1x of its tolerance, and still returned no certified iterate.

**The budget is the knob, and $C_0/\varepsilon$ sizes it.** Tripling the budget moves $C/\varepsilon$ monotonically toward the tolerance and improves *both* axes (`narrow_s42`: 1.36 -> 1.070 at 12000 steps, position error 3.398 -> 3.250, $L$ 13.682 -> 8.496). `ball_ax` was measured at `C_0/eps = 105.7` **before** committing its runs, which correctly predicted that 4000 steps would be wasted; its 24000-step and staged-tolerance arms were queued on the strength of that prediction rather than after the fact.

`posttrain_report.json` now records `starved_of_optimise_steps` and `constraint_at_start_over_epsilon` in its summary, so this is a line in the report rather than an investigation. Regenerate the table with `results_tables.py --section budget`.

**Superseded in part by §6.9.** This section also attributes the missing feasibility certificate on `ball_narrow` and `ball_single` to the budget. That is wrong. Budget does drive the *training-time* constraint below $\varepsilon$ — to $0.69\varepsilon$ at 36000 steps on `ball_single` — and the certificate still does not appear, because the evaluator re-measures $C$ on held-out windows and finds $1.49\varepsilon$ there. The obstacle is a generalisation gap that **widens** with the budget, not a shortage of steps; §6.9 has the measurement.

### 6.8 `ball_square` at the budget its $C_0/\varepsilon$ implied

The rotating square is the first environment where the step budget was sized from $C_0/\varepsilon$ *before* the runs were committed rather than after a wasted one, and it produced the strongest result in the study. At the default 4000 steps the run spent every step in RESTORE, never reached OPTIMISE, and drove the JEPA loss from 11.45 to 72.35; §6.7 predicted that from $C_0/\varepsilon = 100.9$ alone. At 24000 steps the same configuration lands inside the tolerance and improves *both* axes at once.

| configuration | steps | pos in | pos OOD | pos far | resp far | slope | JEPA $L$ |
|---|---|---|---|---|---|---|---|
| `pretrained` | — | 1.259 | 15.208 | 26.519 | **−0.347** | +6.280 | 11.880 |
| continued JEPA | 4000 | 1.025 | 15.251 | 26.284 | −0.329 | +6.178 | 7.726 |
| constrained at $g'=g$ | 4000 | 1.247 | 15.297 | 26.674 | −0.355 | +6.318 | 11.409 |
| penalty, $\lambda = 1$ | 4000 | 1.116 | 10.523 | 17.486 | 0.041 | +4.251 | 9.202 |
| counterfactual | 4000 | 1.581 | 2.228 | 2.903 | 1.055 | +0.406 | **75.448** |
| **counterfactual** | **24000** | **0.983** | **1.597** | **2.192** | **1.011** | **+0.356** | **8.744** |

The 24000-step run reduces the far-field position error from 26.519 px to 2.192 px, a factor of **12.1**, turns a response ratio of $-0.347$ — wrong in sign, so the pretrained model accelerates the square the wrong way when asked for stronger gravity — into $1.011$, where $1.0$ is exact, and flattens the slope from $+6.280$ to $+0.356$ px per unit of $|g' - g|$. It does this while ending at a JEPA loss of 8.744, *below* the pretrained model's 11.880, so nothing was traded away for the counterfactual accuracy.

Comparing the 4000-step and 24000-step rows isolates what the budget actually buys. The counterfactual accuracy is already present at 4000 steps: 2.903 px and a response of 1.055, within 32% and 4% of the 24000-step values. What the extra 20000 steps recover is the world model, $L = 75.448 \to 8.744$, a factor of 8.6. On this environment the constraint teaches the conditioning path quickly and the OPTIMISE phase exists to pay back the damage RESTORE did on the way, which is the division of labour Theorem 1 describes and the first place in this study where it is visible as a clean before-and-after.

The staged-tolerance arm is a weaker result that should not be read as a success. Stage one, at $\varepsilon \times 8$, did reach feasibility and is one of only two counterfactual runs in the whole study that certified — but at a tolerance eight times looser than the calibrated one, and its far-field error is 6.860 px, three times the 24000-step run's. Stage two then starved (0 OPTIMISE steps) and ended at $L = 64.861$. Annealing the tolerance is not what fixed this environment; budget is.

### 6.9 The certificate is blocked by a generalisation gap, not by the step budget

This corrects §6.7. That section attributes the missing feasibility certificate on `ball_narrow` and `ball_single` to the step budget, on the evidence that more budget moves $C/\varepsilon$ monotonically toward the tolerance. The first half of that is right and the conclusion drawn from it is wrong. Budget does drive the *training-time* constraint below $\varepsilon$, and the certificate still does not appear, because `evaluate.py` re-measures $C$ independently on held-out windows and tests $C_{\text{eval}} \le (1 + \zeta_c)\varepsilon$ there.

| model | steps | $C_{\text{train}}/\varepsilon$ | $C_{\text{eval}}/\varepsilon$ | gap | certified |
|---|---|---|---|---|---|
| `ball_v1` | 4000 | 0.90 | 0.91 | 1.0× | **yes** |
| `ball_square`, $\varepsilon \times 8$ | 4000 | 1.04 | 1.06 | 1.0× | **yes** |
| `ball_narrow` | 4000 | 0.90 | 1.36 | 1.5× | no |
| `ball_narrow` | 12000 | 0.78 | 1.07 | 1.4× | no |
| `ball_narrow` | 36000 | 0.86 | 1.15 | 1.3× | no |
| `ball_single` | 4000 | 0.95 | 1.76 | 1.8× | no |
| `ball_single` | 12000 | 0.73 | 1.57 | 2.1× | no |
| `ball_single` | 36000 | 0.69 | 1.49 | **2.2×** | no |
| `ball_square` | 24000 | 0.93 | 2.18 | 2.3× | no |

On `ball_single` the budget works exactly as §6.7 says it does, taking $C_{\text{train}}/\varepsilon$ from 0.95 to 0.69 across a ninefold increase in steps, and the evaluation constraint stays near $1.5\varepsilon$ throughout. **The gap widens as the budget grows**, from 1.8× to 2.2×, which is the signature of the optimiser fitting the training windows more tightly rather than learning the constrained behaviour. Spending nine times the compute moved the certificate no closer and made the generalisation worse.

The two runs that do certify say where the difference comes from. Both have a gap of 1.0×, and both are cases where the constraint transfers: `ball_v1` trains on a wide range of gravities, so the windows the constraint is fit on and the windows it is tested on are drawn from the same place, and the `ball_square` staged arm has a tolerance loose enough that a 1.06× overshoot still fits inside it. On `ball_single` and `ball_narrow` the training data contains one gravity, or few, so the counterfactual constraint is being fit on a narrow window set and does not carry over. That is a statement about the data, not about the optimiser, and it is the real obstacle between this method and a certified result in the regime where its accuracy gain is largest.

Across all 117 evaluated runs 48 are feasible at evaluation; across the 33 counterfactual runs, 15 are. The runs with the largest train-to-evaluation amplification are all `factual` ones, up to 165× on `ball_ax`, but that comparison is not meaningful: the factual variant constrains only $g' = g$ while the evaluator measures $C$ over the whole counterfactual sweep, so the two numbers are different quantities and the ratio should be ignored for those rows.

### 6.10 Ablations of the method's own knobs, on `ball_single`

§6.4 ran these on `ball_v1` and found every knob irrelevant, which was a property of that regime rather than of the method: on `ball_v1` the constraint has little work to do, since the pretrained model already responds to gravity. Rerunning them on `ball_single`, where the constraint does all of the work, separates two of them sharply and leaves four still irrelevant.

| ablation | pos in | pos OOD | pos far | resp far | slope | JEPA $L$ |
|---|---|---|---|---|---|---|
| default | 1.384 | 1.884 | 2.433 | 0.935 | +0.356 | 14.672 |
| $\gamma = 0.5$ (the PDF's value) | 1.357 | 1.937 | 2.568 | 0.915 | +0.395 | 12.866 |
| no hysteresis | 1.196 | 1.901 | 2.618 | 0.914 | +0.463 | 9.785 |
| no projection | 1.384 | 1.884 | 2.433 | 0.935 | +0.356 | 14.672 |
| no in-frame filter | 1.389 | 1.864 | **2.330** | 0.941 | +0.307 | 14.666 |
| acceleration form | 1.183 | 6.864 | **12.426** | **0.337** | +3.101 | 7.610 |
| RESTORE normalised | 3.587 | 9.746 | **13.985** | **0.245** | +3.069 | 62.123 |
| RESTORE $= \nabla C/\varepsilon$ (the PDF's) | 2.472 | 2.960 | 3.322 | 0.953 | +0.361 | **108.201** |
| $\varepsilon$ from `real_latents` | 2.106 | 2.355 | 2.543 | 1.038 | +0.155 | 56.678 |

| ablation | $C_0/\varepsilon$ | $C_{\text{end}}/\varepsilon$ | starved | RESTORE | OPTIMISE | projected |
|---|---|---|---|---|---|---|
| default | 55.0 | 0.95 | no | 3708 | 292 | 0 |
| $\gamma = 0.5$ | 55.0 | 0.85 | no | 3712 | 288 | 8 |
| no hysteresis | 55.0 | 0.97 | no | 3132 | 868 | 21 |
| no projection | 55.0 | 0.95 | no | 3708 | 292 | 0 |
| no in-frame filter | 64.1 | 0.91 | no | 3786 | 214 | 0 |
| acceleration form | 5.4 | 0.91 | no | 1851 | 2149 | 33 |
| RESTORE normalised | 55.0 | **21.26** | **yes** | 4000 | 0 | 0 |
| RESTORE $= \nabla C/\varepsilon$ | 55.0 | 1.30 | **yes** | 4000 | 0 | 0 |
| $\varepsilon$ from `real_latents` | **645.5** | 15.63 | **yes** | 4000 | 0 | 0 |

The constraint form is the larger of the two decisive choices, and the gap is much wider here than on `ball_v1`. The acceleration form gives 12.426 px against the position form's 2.433 px, a factor of **5.1**, where on `ball_v1` the same substitution cost only 1.5×. Its response ratio of 0.337 against 0.935 says the same thing in the units Theorem 2 separates. Notably the acceleration form reaches feasibility comfortably ($C_{\text{end}} = 0.91\varepsilon$, 2149 OPTIMISE steps) and has the better JEPA loss — it satisfies its own constraint and that constraint is the wrong one, which is exactly the failure §6.2 describes and the reason the position form was adopted.

The RESTORE direction is the second, and it only shows up in this regime. Normalising the RESTORE step gives 13.985 px, a factor of **5.7** worse than the default, and it is the one ablation that cannot reduce the constraint at all: it ends at $21.26\varepsilon$, having started from $55\varepsilon$, with zero OPTIMISE steps. On `ball_v1` the same change cost 4%. The PDF's $\nabla C/\varepsilon$ direction reaches $1.30\varepsilon$ but starves and ends at $L = 108.201$, seven times the default.

Four knobs remain irrelevant, now on a regime that could have shown them up: $\gamma$, hysteresis, the in-frame filter and the projection all land within 0.3 px of the default, and the best far-field number in the table belongs to the ablation that removes the in-frame filter. The $\varepsilon$ source matters only through $C_0/\varepsilon$: drawing it from `real_latents` gives $\varepsilon$ so tight that $C_0/\varepsilon = 645.5$, which starves the run by the §6.7 mechanism, and its accuracy is nonetheless fine because the constraint was driven hard — the cost lands on $L = 56.678$ and on the certificate.

The `starved_of_optimise_steps` flag fires on exactly the three ablations whose JEPA loss is damaged, and on no others.

### 6.11 The single-sided projection is effectively dead code

The no-projection ablation in §6.10 reproduces the default run to every printed digit, and the reason is that the projection never fired in either: both report 0 projected steps. Counting across every run in the study that reached OPTIMISE at all, the projection fired on **4176 of 419553 OPTIMISE steps, or 0.995%**, in 34 runs out of the 334 that have OPTIMISE steps.

| run | projected / OPTIMISE | share |
|---|---|---|
| `ssm_s42` $\varepsilon \times 40$ | 89 / 299 | 29.8% |
| `lewm1step_s42` factual | 455 / 2055 | 22.1% |
| `ball_v1` seed 43 | 659 / 3561 | 18.5% |
| `ball_v1`, $\varepsilon$-scale 0.9 | 474 / 3549 | 13.4% |
| `ball_v1`, $\varepsilon$-scale 1.0 | 478 / 3677 | 13.0% |
| `ball_single` (all variants) | 0 / 292 | **0%** |

The activity is concentrated on `ball_v1` at 13–19%, and it is zero on `ball_single`, which is the regime the headline result rests on. The component is not wrong — Proposition 2 says the single-sided projection leaves $C$ unchanged to first order, and that property has a regression test — but its trigger condition, $\hat{C} > \gamma\varepsilon$ together with $\langle \nabla C, \nabla L \rangle < 0$, is almost never met in practice. **The honest statement is that it could be removed from the method with no measurable effect on any result reported here.**

### 6.12 `ball_ax`: the two-dimensional counterfactual fails, and not because of the second dimension

`ball_ax` adds a horizontal acceleration $a_x$ to the action schema, held at exactly zero throughout training, and asks for a counterfactual in which both $g'$ and $a_x'$ move. It is the cleanest test in the study of whether the method teaches a direction of physics that is wholly absent from the data, because the quantity being measured starts at zero rather than at a wrong value: the pretrained model's horizontal response ratio is 0.064, against the $-0.347$ it shows for gravity on `ball_square`. Every arm failed, and the failure is not explained.

| arm | steps | $\varepsilon$ | $C_0/\varepsilon$ | $C_{\text{end}}/\varepsilon$ | OPTIMISE | $L$ start | $L$ end |
|---|---|---|---|---|---|---|---|
| counterfactual | 4000 | 0.0141 | 105.7 | 33.57 | **0** | 13.37 | $1.18 \times 10^6$ |
| counterfactual | 24000 | 0.0141 | 105.7 | 37.92 | **0** | 13.37 | $7.57 \times 10^{12}$ |
| staged, stage 1 | 4000 | 0.1127 | 13.2 | 4.12 | **0** | 13.37 | $4.47 \times 10^4$ |
| staged, stage 2 | 4000 | **10.8887** | 0.04 | 0.18 | 4000 | $4.24 \times 10^4$ | 520.95 |
| $g'$ only, $a_x' = 0$ | 8000 | 0.0141 | 94.0 | 15.13 | **0** | 13.37 | $3.30 \times 10^4$ |
| $a_x'$ only, $g' = 4$ | 8000 | 0.0141 | 76.5 | 22.90 | **0** | 13.37 | 633.30 |

The budget remedy that worked on `ball_square` fails here. Six times the steps moved $C/\varepsilon$ from 105.7 only to 37.92 — the wrong direction from 33.57 at 4000 steps — while the JEPA loss reached $7.57 \times 10^{12}$. For contrast, `ball_square` started from a nearly identical $C_0/\varepsilon = 100.9$ and reached $0.93$. So RESTORE on this model reduces the constraint slowly and damages the predictor quickly, which is qualitatively unlike every one-dimensional case measured.

The decomposition was run to find out whether the horizontal direction is the hard part, and the answer is that it is not. Constraining $g'$ alone on this checkpoint, which is the same scalar problem that `ball_single` solves comfortably — $55.0 \to 0.95$ in 4000 steps — gets only $94.0 \to 15.13$ in 8000 steps with zero OPTIMISE steps. Constraining $a_x'$ alone behaves no worse, $76.5 \to 22.90$. **Both halves of the two-dimensional counterfactual fail on this model, and the scalar half fails here while succeeding elsewhere, so what distinguishes `ball_ax` is the checkpoint and not the dimensionality of the counterfactual.** Finding out what about it is different is open; the four-coordinate action schema, the retrained action encoder and the dataset are all unexcluded.

One positive signal survives. The horizontal response ratio does move under the constraint, from the pretrained 0.064 to 0.421 in the staged stage-one run, and to 0.239 even in the 4000-step run whose loss reached $10^6$. The constraint does reach a conditioning path that training never activated. What is missing is any run that does so while keeping the world model.

The staged arm here is invalid for a reason that was my error rather than the method's, recorded in §10. Stage one damaged the predictor's factual rollout from 0.0113 to 8.711, and stage two then recalibrated $\varepsilon$ against *that* model, producing $\varepsilon = 10.8887$ — **773 times** the original tolerance. The constraint was vacuous at that tolerance, stage two spent all 4000 steps in OPTIMISE, and it drove the horizontal response back down from 0.421 to 0.014. The run measures nothing about staging. `posttrain.py` now accepts `--epsilon` to pin an absolute tolerance across stages, with a regression test, but no corrected staged run has been done.

### 6.13 `--stride` has never affected post-training

The strided generalisation of Lemma 1 to $m$-step differences, with its $m^2$ amplification, is quoted in §4 as the reason the read-out had to change, and runs at $m = 2, 3, 4$ exist on both `ball_v1` and `ball_single`. **None of them tested it.** The checkpoints produced at $m = 2$, $m = 3$ and $m = 4$ are bit-identical to each other on both datasets, with a maximum element-wise weight difference of exactly $0$ across 24.5 M parameters, and on `ball_single` they are bit-identical to the $m = 1$ run as well.

The cause is that `stride` is plumbed into `PostTrainConfig`, recorded correctly in `posttrain_report.json` — the reports do show `'stride': 1, 2, 3, 4` — and then never used. In `gwm/posttrain.py` the name appears exactly once, at line 622, in the diagnostic field `dt_effective`. It is used properly in `gwm/evaluate.py`, so the evaluation-side stride is real and the $[\mathrm{CLS}]$ margin figures quoted in §10 at strides 1 and 4 stand; it is the training-side stride that is a no-op.

Nothing in §6 depended on a strided training result, so no reported number changes. What has to be withdrawn is the implicit claim that the strided objective has been exercised: it has not, on any dataset, and the $m^2$ amplification remains untested in training.

### 6.14 A scene with contacts, and what the constraint can still say in it

Every dataset up to §6.13 is contact-free by construction, because the free-fall identity of Lemma 1 -- the identity the constraint is the residual of -- holds only under uniform acceleration. The physics is therefore a single family: parabolas, exactly, to $7\times10^{-7}$ world units. `bounce_single` adds a floor, so the world model has to learn something the constraint says nothing about, while the constraint keeps its exact meaning on the windows between bounces.

Three measurements decided the design before any of it was built, and all three could have killed the experiment.

| measurement | value | what it decided |
|---|---|---|
| free-fall residual *between* contacts, `mj_step` with RK4 at a 0.0005 substep | $7.2\times10^{-7}$ world units, $6.5\times10^{-6}$ px | the integrator can be the source of truth; the constraint loses no signal to drift |
| residual on a window *spanning* a contact | 1.26 world units, **12.4 px** | such windows must be excluded, not counted as violations: the one-step gravity displacement is 1.008 px |
| realised restitution | $e = 0.897$ | physical. The first `solref` tried returned $e = 1.60$ -- the object left the floor faster than it arrived |

The restitution has to be *measured*. `solref="0.002 0.05"` gains energy at every bounce, and nothing downstream would have reported that; the generator now measures what it produced and refuses to write a dataset with $e \ge 1$.

**The constraint speaks only where Lemma 1 holds.** That is the whole design, and it needs three filters rather than one.

*The context must be contact-free.* The position form starts its counterfactual rollout from an implied velocity $v = (p_t - p_{t-1})/\mathrm{d}t + g\,\mathrm{d}t/2$ read off the last two context frames, and that read is a free-fall quantity which is wrong across a bounce.

*The counterfactual path must stay off the floor,* and this is tested under $g'$ rather than under $g$. A window that never touches the floor at $g = 4$ can bounce at $g' = 9$, and the identity fails there too. The test is exact, because between contacts the counterfactual path *is* a parabola: the first positive root of $z_0 + v_z s - \tfrac{1}{2}g' s^2 = z_{\text{rest}}$ is $s^* = \big(v_z + \sqrt{v_z^2 + 2g'(z_0 - z_{\text{rest}})}\big)/g'$, and the window is rejected when $s^* \le H\mathrm{d}t$. Verified against `mj_step` on 40 of 40 episodes.

*The tolerance must be calibrated on contact-free windows too.* This one is easy to miss and fatal: $\varepsilon$ is $(1+\rho)$ times the predictor's factual rollout violation, so a bounce inflates it and the constraint goes vacuous. It is the same failure the staged tolerance hit from the other direction in §6.12, where a damaged factual rollout produced $\varepsilon$ at 773 times the original.

Both filters reject without resampling, as the in-frame filter does, so $P$ stays the law Corollary 2 needs with $\kappa = 1$.

The episodes are 32 frames rather than 16, and that was forced by the arithmetic rather than chosen. A $\text{history} + \text{horizon} = 14$ frame window has only three starts in a 16-frame episode, so a bounce anywhere in the context kills most of them: pushing the bounce density up took the constraint-usable share from 93% to 19%. At 32 frames there are 19 starts, and the dataset lands at **51.8% of windows usable by the constraint and 37.8% containing a bounce in the horizon** -- 37% are both, which is the regime the experiment needs.

The gates, all measured before committing a post-training run:

| gate | `bounce_single_r08` | `ball_single` | `cls_s42` (the known failure) |
|---|---|---|---|
| probe $h$ | 0.260 px | 0.061 px | 33.429 px |
| probe $h_{\mathrm{eval}}$ | 0.206 px | 0.195 px | 33.398 px |
| SNR | **3.88** | 16.5 | 0.03 |
| $\varepsilon$ | 0.0157 | 0.0137 | — |
| $C_0/\varepsilon$ | **36.2** | 55.0 | — |

SNR 3.88 is well short of the contact-free models' 8.5--16.5 and well clear of the $[\mathrm{CLS}]$ read-out's 0.03. It is usable, and it is the reason this section's numbers carry less weight than §6.8's.

Getting even that far took two corrections that had nothing to do with the question. The object was generated at `half_size` 0.45 where every other dataset uses 0.80, which is 44% of the diameter and cost the probe a factor of 3.3 (0.870 px, SNR 1.16); and the default pretraining collapsed the representation on this data, which §6.15 covers.

---

## 7. How to read a row

- `pos far` is the number to watch for C1. `pos in` is nearly saturated by probe noise and separates nothing.
- `resp far` is the mechanistic one: it says whether the model *moves the right distance* when gravity changes, independent of its baseline accuracy.
- `slope` is Theorem 2's quantity. The theorem bounds the full method's error by something that does not depend on the gravity shift, while a gravity-blind predictor's error is bounded *below* by something that grows with it. A flat slope for the method and a rising one for the ablations is the separation; absolute errors alone do not show it.
- `JEPA L` is C2. A large value means the world model was traded away.
- `feas` is C3, and only comparable between runs with the same constraint spec (the `kkt` block records the run's and nulls `feasible` on a mismatch).
- $n$ is **seeds of one configuration**. Never read a row averaged across configurations.

---

## 8. Open items

| item | status |
|---|---|
| **the certificate in the regime the story rests on** | **open, and the explanation changed (§6.9).** Not a budget problem: 36000 steps reaches $C_{\text{train}} = 0.69\varepsilon$ on `ball_single` and the evaluator still measures $1.49\varepsilon$ on held-out windows. The train-to-evaluation gap *widens* with the budget. Needs a change that makes the constraint generalise — more counterfactual windows, or a tolerance that accounts for the gap — not more steps |
| **why `ball_ax` fails (§6.12)** | **open.** Both halves of the 2D counterfactual starve on this checkpoint, including the scalar half that `ball_single` solves in 4000 steps. The four-coordinate action schema, the retrained action encoder and the dataset are all unexcluded. Next probe: run the `ball_single` counterfactual on a `ball_ax`-schema model trained with $a_x$ present but constant, to separate schema from data |
| **corrected staged tolerance** | **not run.** The staged arms on `ball_ax` and `ball_square` are invalid — each stage recalibrated $\varepsilon$ against the damaged model. `--epsilon` now pins it; the experiment has not been redone |
| **strided training objective (§6.13)** | **never tested.** `--stride` is recorded but unused in `posttrain.py`. Wiring it through is a code change, not a run |
| the single-sided projection | **effectively dead (§6.11).** 0.995% of OPTIMISE steps study-wide, 0% on `ball_single`. Candidate for removal; kept for now because Proposition 2 refers to it |
| position form on the other 5 base models | done (§6.5) |
| penalty frontier $\lambda \in \{300, 1000, 3000\}$ | done (§6.3) |
| second seed of the tolerance frontier | done; headline is $n = 2$ |
| learning rate swept on the real model | **partly answered, not swept.** `measure_gradient_scale.py` sets it per model from the measured $\lVert \nabla L \rVert$, which found the SSM divergence (§5). Whether a different step length would beat the GRU's 0.045 is untested |
| SSM and LeWM-one-step baselines at the matched `lr` | done. The diverged position-form runs are kept as `post_mpdiv_*` |
| Polyak gap uses the minibatch $C_B$ while mode selection uses the EMA $\hat{C}$ | mismatch produces occasional zero-length RESTORE steps; deliberately **not** changed mid-suite so batches stay comparable |
| only $n = 1$ for ablations, the frontier, `ball_square` and `ball_ax` | headline is $n = 2$. The $5.1\times$ and $5.7\times$ ablation gaps in §6.10 are large enough that a second seed is unlikely to reverse them, but they are single-seed |

---

## 9. Reproducing

```bash
# the whole suite, two GPU jobs at a time, idempotent and restartable
cd work
setsid nohup bash orchestrate.sh >> orchestrate.log 2>&1 &
gwm-status          # phase log, running jobs, completed counts, failures
gpuwatch            # live GPU + queue panel

# one configuration by hand
python scripts/posttrain.py --checkpoint CKPT --cache CACHE --probe-dir PROBES \
  --output-dir OUT --variant counterfactual --constraint-form position \
  --epsilon-source factual_rollout --steps 4000 --lr 1e-4 \
  --history 8 --horizon 6 --stride 1 --gravity-low 1 --gravity-high 9 --seed 42
python scripts/evaluate.py --checkpoint OUT/checkpoint.pt --cache CACHE \
  --probe-dir PROBES --output OUT/evaluation.json \
  --posttrain-report OUT/posttrain_report.json --history 8 --horizon 6 --stride 1
python scripts/report.py --evaluations 'RUNS/*/post_mp_*/evaluation.json' --group mp

# what learning rate does this checkpoint want?  (step length lr*||grad L||)
python scripts/measure_gradient_scale.py --checkpoint CKPT --cache CACHE

# regenerate every table in section 6 of this document, as markdown
python scripts/results_tables.py --suite RUNS/suite [--section headline|forms|certificate|ablations|models]
```

The tables in §6 are produced by `results_tables.py` and pasted in, so updating a section is a copy-paste and checking one is a diff.  The prose is hand-written; the numbers are not.

`orchestrate.sh` must not be edited while it is running — bash reads a script incrementally and the running shell resumes at a byte offset that an edit moves. One-off sweeps go in their own file.

Phases: `mainpos` `posfrontier` `allpos` `narrow` `basepost` `clspost` `ablate` `strides` `v2` `frontier` `forms` `baselines` `main` `report`.

Timing, on 2x A100-80G with 2 jobs in flight: post-training 523 s median (4000 steps), evaluation ~15 s, pretraining ~2 h/model, dataset generation ~12 min. The 2026-11-02 suite was 256 jobs over ~7 h.

---

## 10. Update log

### 2026-11-02 — first complete suite, and a units bug in the certificate

256 jobs, all exit 0. Headline (§6.1) and form comparison (§6.2) established on two seeds.

**Bug found and fixed: the feasibility certificate was computed in the wrong units.** `evaluate.py`'s `--constraint-form` defaulted to `acceleration` and `orchestrate.sh` never passed it, so every position-form run was post-trained against a tolerance calibrated in Proposition 1's form and then evaluated against a violation measured in Definition 1's form — a $\sim \mathrm{d}t^{-4}$ mismatch, in numbers `eps = 0.0668` against `C_out = 16.02`, ratio **240**. Every `feas` cell read `no`, including for runs comfortably inside their own tolerance.

Scope: the `kkt` block only. The accuracy columns are computed directly from decoded positions against closed-form truth, so §6.1 was unaffected. Corollary 2's bound and Lemma 7's pathwise term deliberately use the stride-one acceleration form and were also correct.

Fixed by taking the form from the run's own recorded command (`constraint_form_for`), erroring on a contradicting flag, and recording `spec.form` in the evaluation payload — whose absence is why this survived a whole suite. The `kkt` block now also records the run's `history`/`horizon`/ `stride` and nulls `feasible` when they differ. Regression test: `tests/test_cli_contracts.py::test_evaluation_takes_the_constraint_form_from_the_run`.

**Same root cause mislabelled a diagnostic column.** `free_fall_violation` returns `(c, accelerations, targets)` in one form and `(c, positions, targets)` in the other; the sweep read its residual off whichever was configured, so `acceleration_rmse` held a position residual for position-form runs (0.278 where the acceleration residual was 5.32). Now read from the stride-one acceleration call the sweep already makes for Corollary 2. Test: `tests/test_probe_and_eval.py::test_acceleration_diagnostics_do_not_depend_on_the_constraint_form`.

With the certificate corrected, §6.3 became the clearest result in the study: the method lands inside three different tolerances and declines to certify an unattainable one, where the penalty needs a two-order $\lambda$ sweep to get inside at all.

**Also corrected:** this document's predecessor text in `README.md` quoted a `[CLS]` margin of 0.68; the measured value from `probe_report.json` is 0.004 at stride 1, rising only to 0.14 at stride 4. Either way the read-out is unusable, but the number is now the measured one.

**Process note:** `orchestrate.sh` was patched while a copy of it was still running and the running shell died on `name: unbound variable`. Nothing was lost — that process had finished its last phase — but new phases now go in only when the orchestrator is idle.

### 2026-11-02, later — the SSM baseline was diverging on step size, not on the method

`allpos` (the position form on every other base model) failed on `ssm_s42/factual` with `FloatingPointError: constraint value is not finite`. The cause is not the constraint. `measure_gradient_scale.py` (new) measures $\lVert \nabla L \rVert$ at $\theta_0$: the SSM's is **32 768**, 59x the GRU's 556. At the suite-wide `lr = 1e-4` that is a step of 3.28, and the `1e4` gradient clip engaged at step 0 -- turning the analysed step into a normalised one of length 1.0, the exact failure `design.md` warns about for `grad_clip`. Continued JEPA training *alone* then diverges on it ($L$ 8.7 -> 1320 in 200 steps).

So both reported "SSM failures" -- `JEPA L = inf` in the acceleration form and this one in the position form -- are failures of the step size. Neither says anything about the constraint, and reporting either as a result about the method would have been wrong.

`lewm1step_s42` was also mis-scaled (step 0.286, 5x the GRU's) but survived: `C_end = 0.435` inside `eps = 0.489` with 1386 RESTORE steps.

Queued `work/lr_matched.sh`: both baselines rerun at a step length matched to the GRU's 0.045 by the measured rule, all five variants each, so the comparison uses one rule for every base model. The diverged runs are kept as `post_mpdiv_*`.

This partly answers the standing "learning rate never swept on the real model" item: it is now *set* per model from a measurement, which is what found the bug. It is still not a sweep.

### 2026-11-03 — a single training gravity, a second seed, and the regime table

`--train-std 0` (`ball_single`, `g == 4.0` exactly) and a fixed gravity normalisation give the cleanest result in the study, and a second seed on both it and `narrowfx` confirms it. The new headline is not any single row but how the separation depends on the regime (§6 head): the gain is 1.4x where the training gravity varies visibly, 9.0x where it varies below the probe noise, and 10.3x where it does not vary at all, while the method's **absolute** error stays at 2.7-4.5 px across all three and the baseline's rises from 6.4 to 28.

Two findings from building it.

**The gravity coordinate's normalisation was an unnoticed confound.** It is z-scored with the training data's own standard deviation, which on `ball_narrow` is 0.05: training inputs spanned +-4 in normalised units while a counterfactual $g'$ in [1, 9] arrived as **-60 to +100**. The narrow result was obtained with the conditioning path driven 25x outside the range it ever saw. Fixing the scale to the counterfactual law's own spread (2.309) removes that; the result survives and improves (JEPA-loss cost 17% -> 14%, and it is the configuration that certifies). As a side effect the *pretrained* JEPA loss falls from 51.0 to 14.9 -- the data's own 0.05 was amplifying gravity noise 20-fold into the conditioning path, and the predictor was fitting it. `scripts/pretrain.py --gravity-scale` / `--accel-x-scale`.

**With `g` constant the measured scale collapses to the `1e-6` floor**, so a counterfactual input would reach `5e6`. The fix is required there, not merely advisable.

The `ball_single` ladder is worth reading as a whole, because each rung rules out an explanation: pretrained -0.451, continued JEPA training -0.414 (more compute does not help), the constraint at $g' = g$ -0.453 (the constraint machinery alone does not help), a fixed penalty -0.181 (still gravity-blind), the counterfactual constraint **+0.908**. Only the counterfactual evaluation moves it, which is the claim the five variants exist to test.

### 2026-11-03 — the penalty frontier on `ball_single`, and the budget

The weak point of the comparison was that on `ball_v1` a swept penalty ($\lambda = 300$) beat the method's whole certified frontier.  Swept on `ball_single`, the regime the case rests on:

| configuration | pos far | resp far | slope | JEPA L | $C/\varepsilon$ |
|---|---|---|---|---|---|
| continued JEPA training | 27.246 | -0.373 | +6.642 | 7.360 | 66.00 |
| penalty $\lambda = 10$ | 16.211 | 0.134 | +3.966 | 9.636 | 21.77 |
| penalty $\lambda = 100$ | 4.826 | 0.796 | +1.096 | **8.448** | 3.20 |
| penalty $\lambda = 300$ | 3.118 | 0.891 | +0.516 | 13.902 | 2.15 |
| penalty $\lambda = 1000$ | 2.685 | 0.950 | +0.276 | 17.582 | 2.27 |
| **full method, $\varepsilon$** | **2.433** | 0.935 | +0.356 | **14.672** | **1.76** |
| full method, $\varepsilon \times 0.8$ | 2.957 | 0.943 | +0.239 | 26.107 | -- |
| full method, $\varepsilon \times 0.6$ | 2.647 | 1.026 | +0.175 | 82.284 | -- |
| full method, $\varepsilon \times 0.45$ | 3.033 | 1.051 | +0.168 | 71.657 | -- |

Sweeping $\lambda$ does rescue the penalty here too -- from -0.373 to 0.950 -- so the method's advantage over a *tuned* penalty is not an order of magnitude anywhere.  But the ordering is now the other way round from `ball_v1`: the method at its default tolerance **dominates $\lambda = 1000$ on both axes** (2.433 px at $L$ 14.672 against 2.685 px at 17.582) and trades favourably against $\lambda = 300$.  It also reaches that point without a sweep, and with `C/eps = 1.76` against the penalty's 2.15-2.27.

Two things fall out of the table that are worth stating plainly.

**Tightening the tolerance does not help here.**  $\varepsilon \times 0.8$, `x 0.6` and `x 0.45` all give a *worse* position error than the default while costing 1.8x to 5.6x the JEPA loss and certifying nothing.  On `ball_v1` the frontier was real; on `ball_single` the default $\varepsilon$ is already the right operating point, and the useful knob is the step budget, not the tolerance.

**The certificate is budget-limited, and measurably so.**  Nothing certifies in this regime at 4000 steps.  Tripling the budget moves $C/\varepsilon$ monotonically toward the tolerance and improves *both* axes:

| | 4000 steps | 12000 steps | threshold |
|---|---|---|---|
| `narrow_s42` $C/\varepsilon$ | 1.36 | **1.070** | 1.042 |
| `single_s42` $C/\varepsilon$ | 1.76 | 1.566 | 1.105 |
| pos far (2-seed mean) | 3.398 | **3.250** | |
| JEPA L (2-seed mean) | 13.682 | **8.496** | |

`narrow_s42` misses its own tolerance by 2.7% after 12000 steps.  A 36000-step run is queued to settle whether the certificate is reachable here at all; C3 is the claim that separates the method from a penalty and it is currently unsupported in the headline regime.

### 2026-11-03, later — the square succeeds, the certificate claim is withdrawn, and three bugs

**`ball_square` is now the strongest result in the study (§6.8).** Its budget was sized from $C_0/\varepsilon = 100.9$ before the runs were committed, which is the first time the §6.7 rule was used predictively rather than diagnostically. At 24000 steps the far-field position error falls from 26.519 px to 2.192 px, a factor of 12.1, the response ratio goes from $-0.347$ to $1.011$, and the JEPA loss ends at 8.744 against the pretrained 11.880 — better on both axes at once. Comparing against the starved 4000-step run isolates what the budget buys: the counterfactual accuracy was already there at 4000 steps, and the extra 20000 steps recovered the world model, $L = 75.448 \to 8.744$.

**Withdrawn: the budget explanation for the missing certificate (§6.9).** §6.7 concluded that `ball_narrow` and `ball_single` fail to certify because the step budget starves them. The premise holds and the conclusion does not. At 36000 steps on `ball_single` the training-time constraint reaches $0.69\varepsilon$ and the certificate still does not appear, because `evaluate.py` re-measures $C$ on held-out windows and finds $1.49\varepsilon$. The gap **widens** with the budget, $1.8\times \to 2.1\times \to 2.2\times$ across 4000, 12000 and 36000 steps, which is the optimiser fitting the training windows rather than learning the constrained behaviour. Nine times the compute moved the certificate no closer and made the generalisation worse. The two runs that do certify both have a gap of $1.0\times$, and both are cases where the constraint transfers — a wide training range on `ball_v1`, or a tolerance loose enough to absorb the overshoot. This is a limitation of the data, not of the optimiser, and it is the real obstacle in the regime where the accuracy gain is largest.

**The ablations separate once they are run where the constraint matters (§6.10).** §6.4 found every knob irrelevant on `ball_v1`. On `ball_single` the constraint form costs $5.1\times$ in far-field position error, against $1.5\times$ on `ball_v1`, and the RESTORE direction costs $5.7\times$, against 4%. The acceleration form reaches feasibility comfortably with the better JEPA loss and is still $5.1\times$ worse — it satisfies the wrong constraint, which is §6.2's argument in the regime that can actually show it. Four knobs — $\gamma$, hysteresis, the in-frame filter, the projection — remain within 0.3 px.

**The single-sided projection is dead code (§6.11).** It fired on 4176 of 419553 OPTIMISE steps across the whole study, 0.995%, in 34 of 334 eligible runs, and never once on `ball_single`. The no-projection ablation reproduces the default to every printed digit for that reason. Proposition 2 is not wrong, but its trigger condition is almost never met, and the component could be removed with no measurable effect on any reported result.

**`ball_ax` fails and is not explained (§6.12).** Every arm starved. Six times the budget moved $C/\varepsilon$ from 105.7 only to 37.92 while the JEPA loss reached $7.57 \times 10^{12}$, where `ball_square` reached $0.93$ from a nearly identical start. The decomposition shows the horizontal direction is not the hard part: constraining $g'$ alone on this checkpoint — the same scalar problem `ball_single` solves in 4000 steps — gets $94.0 \to 15.13$ in 8000 steps with zero OPTIMISE steps, and constraining $a_x'$ alone is no worse. What distinguishes `ball_ax` is the checkpoint, not the dimensionality. One positive signal: the horizontal response does move under the constraint, 0.064 to 0.421, so the constraint reaches a conditioning path training never activated; no run yet does so while keeping the world model.

**Bug, mine: a staged tolerance relaxed its own successor's constraint.** $\varepsilon$ is calibrated against the *current* checkpoint's factual rollout, so in a staged schedule a stage that damages the model loosens the next stage's tolerance. On `ball_ax`, stage one took the factual violation from 0.0113 to 8.711 and stage two then calibrated $\varepsilon = 10.8887$, **773 times** the original. The constraint was vacuous, stage two spent all 4000 steps in OPTIMISE, and it drove the horizontal response back from 0.421 to 0.014. That arm measures nothing. `posttrain.py` now takes `--epsilon` to pin an absolute tolerance across stages, with a regression test; no corrected staged run has been done.

**Bug, mine: `starved_of_optimise_steps` fired on the `pretrained` variant.** That variant takes no steps, so "no OPTIMISE step" is vacuously true and read as the failure the flag is meant to catch. Now guarded on `step > 0`, with a test distinguishing 0/4000 from 0/0.

**Bug: `--stride` has never affected post-training (§6.13).** It is plumbed into `PostTrainConfig` and recorded correctly in `posttrain_report.json`, and then used only in the diagnostic field `dt_effective` at `gwm/posttrain.py:622`. The $m = 2, 3, 4$ checkpoints are bit-identical to each other on both `ball_v1` and `ball_single`, maximum element-wise weight difference exactly $0$ across 24.5 M parameters. No reported number depends on a strided training result, but the claim that the strided generalisation of Lemma 1 has been exercised is withdrawn — it has not been, on any dataset. Evaluation-side stride is used properly and the $[\mathrm{CLS}]$ margin figures at strides 1 and 4 stand.

**Process note.** The orchestrating script was edited while running for the third time in this study, and for the third time the running shell resumed mid-statement at a shifted byte offset. The queue order is now a data file, `work/queue.list`, read by a `runner.sh` that does not change; appending a queue is safe at any moment and editing the runner is never necessary. The runner also detects a queue already running as a child of an earlier runner and waits for it instead of starting a second copy.

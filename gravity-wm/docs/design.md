## The latent read-out decides whether any of this can run

The constraint acts on the latent the predictor outputs, so the position probe
is only as accurate as the position information that reaches that latent.  On
this task the published ``[CLS]`` read-out does not carry it, and that is not a
matter of training longer:

| checkpoint / feature | probe error | variance explained |
|---|---|---|
| random encoder, `emb` (`[CLS]` -> projector) | 34.98 px | 0.0003 |
| trained encoder (ep 9), `emb` | 34.60 px | 0.0215 |
| random encoder, patch map | 34.98 px | 0.0003 |
| trained encoder (ep 9), patch map | **6.23 px** | **0.9683** |

Training is therefore what teaches the ViT where the object is, and the
read-out is what discards it.  The `[CLS]` probe error is flat from epoch 4 to
epoch 16 (28.98, 29.79, 33.91, 33.95 px), and is unchanged by a 36x larger
probe (9.4M parameters), by probing an eight-frame window instead of one frame,
and by per-dimension standardisation; ZCA whitening of the latent recovers only
0.34, so what survives sits in low-variance directions.  Nearest-neighbour
decoding is *worse* than predicting the mean, at exactly the value a latent
independent of position would give.

The mechanism is that a pooled patch feature ``sum_p alpha_p t_p`` carries
location through the position embeddings in ``t_p`` only if ``alpha_p`` depends
on which patches the object occupies.  Uniform mean pooling averages every
position embedding into the same constant, so location cancels exactly; the
same holds for DINO-WM's mean-pooled features.  ``gwm/spatial_lewm.py``
therefore provides two replacement read-outs, changing nothing else about the
model.  ``attn_pool`` *can* keep location but, started from the uniform weights
it is initialised to, the objective does not drive it to concentrate and its
latent stays at chance.  ``spatial_map`` compresses each patch token to two
channels and keeps the spatial axis.

After **one epoch**, ``spatial_map`` gives a probe error of **0.96 px**
(variance explained 0.9992) at a validation rollout loss of 0.067 -- better
than the `[CLS]` run reaches in nine epochs (0.088).  The resulting margins:

| stride | `g dt_eff^2` (px) | measured `c_real` | `eps` | `c_blind` | margin |
|---|---|---|---|---|---|
| 1 | 1.01 | 2.4e1 | 3.1e1 | 22.7 | **0.74** |
| 2 | 4.03 | 2.6e0 | 3.2e0 | 18.9 | 5.82 |
| 3 | 9.07 | 6.3e-1 | 7.9e-1 | 15.0 | **18.98** |
| 4 | 16.13 | 2.3e-1 | 2.9e-1 | 11.2 | 38.78 |

The stride-one row is the quantitative justification for the strided
generalisation of Lemma 1: at this sampling rate the constraint is *not*
discriminative even with a sub-pixel probe, because the one-step gravity
displacement is a single pixel.  Stride three makes it 19x.

## The tolerance has to be attainable, and `c_real` on real latents may not be

`eps = (1 + rho) E[c_real]` with `c_real` measured on *real* future latents
isolates the probe's noise, which is what the method note intends: the true
dynamics then satisfy the constraint in expectation, so enforcing it cannot
bias the predictor away from them. But `C(theta)` is measured on *predicted*
latents, and the predictor's own rollout error adds to the violation
irreducibly. Measured on the control run, with a probe accurate to 0.18 pixel:

| tolerance source | `E[c_real]` | `eps` |
|---|---|---|
| `real_latents` (the note's definition) | 3.8e-2 | 4.7e-2 |
| `factual_rollout` (same violation on the predictor's own rollout at `g' = g`) | 1.6e0 | 2.0e0 |

The predictor's rollout error in probe coordinates is **43x** the probe's
noise, so the note's tolerance sits far below anything the predictor can
reach. With it, Algorithm 1 never leaves RESTORE: `IO` stays empty, the
returned iterate carries no certificate, and every conclusion of Theorem 1 is
vacuous -- which is exactly what the first control run did over 3000 steps,
while the JEPA loss rose by a factor of twelve.

`factual_rollout` measures the same violation on the predictor's own factual
rollout, so the tolerance is attainable at `theta_0` by construction and the
constraint reads "do not become less physically consistent when the gravity
changes than you already are at the gravity you were trained on" -- which is
the generalisation the method claims. It also makes condition (20) of
Theorem 2(c) hold by construction. Both are computed and reported on every
run, and their ratio is the diagnostic.

## Step sizes have to be calibrated, or the ablation is a straw man

Two step sizes needed measurement rather than a default.

*RESTORE.* The note's direction is `grad C_B / eps`. With a fixed step size,
`||grad C||` grows as the constraint gets harder, so a step calibrated at the
first RESTORE iteration becomes too long later: observed, RESTORE began
*increasing* the violation it was meant to reduce (0.18 to 2.33) while the
JEPA loss went from 0.5 to 403. The default is now Polyak's step for the
level set `{C = 0.8 eps}`, whose length adapts to the gradient norm by
construction and which is what the switching-subgradient analysis uses.

*OPTIMISE.* `continued JEPA training` is only a fair ablation if its step size
decreases `L`. Swept on the control run:

| `lr` | `L` before | `L` after | ratio |
|---|---|---|---|
| 1e-3 | 5.8e-1 | 1.3e5 | 220279 |
| 1e-4 | 5.8e-1 | 2.6e-1 | 0.43 |

At `1e-3` the ablation diverges, and a comparison against it would flatter the
method by a wide margin. `control_experiment.py --lr-sweep` picks the largest
step size that actually descends, and the pipeline uses it.

## Definition 1 is the wrong objective to descend, and Proposition 1 fixes it

Definition 1 constrains the squared second differences of the decoded
positions. Proposition 1 says driving that to zero puts the decoded positions
on the counterfactual trajectory through the decoded context, so the two have
the same zero set. They do not have the same landscape, and on this problem
the difference decides whether the method helps at all.

*The squared form is blind to the sign structure that decides position error.*
By Lemma 7 the position error at step `s` is a signed, time-weighted sum of
the acceleration residuals, `dt^2 sum_j (s-j)(a_j - gbar_j)`, while Definition
1 averages their squares. Two perturbations costing the same in position:

| perturbation | Definition 1 | position form |
|---|---|---|
| a systematic extra acceleration of 0.6 | 0.315 | 0.0224 |
| i.i.d. noise, sd 0.12 | **165.4** | 0.0271 |

The squared form over-penalises the unsystematic residual by a factor 525. Its
gradient is therefore dominated by the component that is cheap to remove and
nearly irrelevant to position -- which is what the runs showed: driving the
violation from 57.7 to 19.2 cut the residual's RMS from 7.59 to 4.38 while its
signed part barely moved, 3.86 to 3.31, and the terminal position error got
*worse*, 11.2 px to 15.2 px.

*The squared form is also far worse conditioned.* Optimising the decoded
positions directly -- the cleanest possible stand-in for a predictor -- with
4000 SGD steps from the same start:

| form | largest stable step | distance to the exact path, as a fraction of the start |
|---|---|---|
| Definition 1 | 2e-4 (diverges at 5e-4) | 3.8e-2 |
| position | 3e-1 | 1e-5 |

A second difference squares the ill-conditioning of integration; the position
form inverts it in closed form. Three to four orders of magnitude separate
them.

*The position form needs nothing extra.* Lemma 1's implied velocity
`v = (p_t - p_{t-1})/dt + g dt/2` comes from the two decoded *context*
positions, so the target trajectory
`p_t + v s dt + g' (s dt)^2 / 2` uses only real latents and the two gravities.
No future frame is involved, which is the property the whole construction
rests on.

Measured on real pixels at the same tolerance, with everything else identical:

| | counterfactual position error (far band) | JEPA loss |
|---|---|---|
| pretrained | 7.91 px | 49.6 |
| continued JEPA training | 5.85 px | 21.0 |
| Definition 1 | 5.93 px | 15.3 |
| **position form** | **4.45 px** | **19.1** |
| position form at 0.6x the tolerance | **3.23 px** | 27.7 |

Definition 1 is indistinguishable from doing nothing; the position form beats
continued JEPA training on both axes at the same tolerance, and trades loss
for physics along a usable frontier when the tolerance is tightened.
`--constraint-form` selects between them and both are kept.

## Protocol

1. **data** -- contact-free projectile episodes; training gravity
   `g ~ N(4, 0.5^2)` as in the Semigroup-JEPA planar protocol, held-out
   episodes both at the training gravity (the real windows of the
   counterfactual protocol) and on a wide grid (the harder setting of
   Remark 8(iii), which Corollary 2 does not cover).
2. **pretrain** -- Semigroup-JEPA, unmodified, one model per seed.
3. **cache** -- encode every episode once. The encoder and projector are frozen
   for the rest of the study, so this is exact, and it makes the JEPA branch,
   the constraint branch and the evaluation consume byte-identical latents.
4. **probes** -- the position probe `h` and the evaluation probe `h_eval`, on
   disjoint halves of the labelled episodes and from different initialisations,
   then frozen. `eps = (1 + rho) E[c_real]` on held-out real windows.
5. **posttrain** -- `pretrained`, `jepa`, `factual`, `penalty`,
   `counterfactual`.
6. **evaluate** -- all with `h_eval`, never with `h`.

## What is measured

* counterfactual acceleration error against the exact switched trajectory, as a
  function of the conditioning gravity, inside and outside the training range;
* counterfactual position error per rollout step, which is the quantity
  Corollary 2 bounds and Theorem 2 separates;
* that bound, evaluated at the measured `delta` and the measured stride-one
  `C(theta)` -- stride one because that is the form Lemma 7 is derived for;
* `L(theta)` on real windows, to check post-training does not degrade the world
  model;
* the disagreement between `h` and `h_eval` on predicted latents, which is
  Remark 8(i)'s correction term and needs no labels;
* the three KKT residuals of Theorem 1(c), and `zeta_C` from the gap between
  the moving average that selects the mode and a large-batch estimate of `C`;
* the slope of counterfactual error against gravity shift, which is the
  separation of Theorem 2: flat for the full method, rising for a
  gravity-blind one.

## Known limitations

* The distribution `P` pairs a real window with `g' ~ U(G)` *restricted* to
  pairs whose exact counterfactual trajectory stays inside the rendered view,
  because the tolerance argument needs the rollout to stay within the range of
  positions the probe was trained on. The same restriction is applied at
  evaluation, so the test law is dominated by `P` with density one on the kept
  set, and the acceptance rate is reported.
* Gravity is given per training episode, as in the main text of the method
  note. Estimating it from the context is not implemented.
* Corollary 2's guarantee covers the contact-free rollout of length `H` on
  which the residual is controlled. It does not extend to longer rollouts, nor
  to latent or frame prediction error -- Proposition 2 in the theory note's
  Appendix F shows the latter cannot follow from `L` and `C` alone.

## The certificate was being checked in the wrong units

The headline accuracy numbers and the feasibility certificate come from
different parts of `scripts/evaluate.py`, and only the first was correct.

`evaluate.py` built its `ConstraintSpec` from a `--constraint-form` flag whose
default was `acceleration`, while `orchestrate.sh` never passed the flag.  So
every position-form run was post-trained against a tolerance calibrated in the
position form of Proposition 1 and then evaluated against a violation measured
in the acceleration form of Definition 1.  The two have the same zero set and
nothing else: with `dt = 0.16` they differ by a factor of order `dt^-4`.
Measured, `eps = 0.0668` against `C_out = 16.02`, a ratio of 240, so the
reported certificate read `feasible = False` for every run including the ones
that were comfortably inside their own tolerance.

With the form taken from the run's own recorded command instead of a default,
the certificate says what Theorem 1 claims it should:

| configuration | C/eps, seed 42 | C/eps, seed 43 | certified |
|---|---|---|---|
| continued JEPA training | 1.58 | 1.49 | no |
| constraint at `g' = g` | 1.57 | 1.54 | no |
| penalty, `lambda = 1` | 1.43 | 1.55 | no |
| penalty, `lambda = 10` | 1.27 | -- | no |
| penalty, `lambda = 100` | 0.88 | -- | yes |
| **full method, `eps`** | **0.85** | **0.95** | **yes** |
| full method, `eps x 0.8` | 0.92 | -- | yes |
| full method, `eps x 0.6` | 0.96 | -- | yes |
| full method, `eps x 0.45` | no certified iterate | -- | no |

Two things in that table are the point of the method.  The full method lands
inside whatever tolerance it is given -- 0.85, 0.92, 0.96 of `eps` at three
different tolerances -- and stops when the tolerance is below what the
predictor can attain (`eps x 0.45`, where the calibration's own attainability
gap already predicted `I_O = empty`).  The penalty reaches the same place only
at `lambda = 100`, and the only way to learn that was to sweep `lambda` over
two orders of magnitude and measure `C` afterwards.  That is the separation the
notes argue for, and it is a statement about *what the two methods certify*,
not about accuracy: at matched feasibility the penalty's counterfactual error
(4.33 px) and the method's (4.45 px) are within noise of each other.

Two guards now make the mismatch impossible rather than unlikely.
`constraint_form_for` takes the form from `--posttrain-report` and raises if an
explicit flag contradicts it, and the evaluation payload records `spec.form`,
whose absence is why this went unseen for a whole suite.  The `kkt` block also
records the run's `history`, `horizon` and `stride` and sets `feasible` to null
when they differ from the evaluation's, since those change `C`'s scale too.

### The same root cause mislabelled the acceleration diagnostics

`free_fall_violation` returns `(c, accelerations, targets)` in the acceleration
form and `(c, positions, targets)` in the position form.  The sweep read its
residual off whichever form was configured, so for position-form runs the
`acceleration_rmse` and `acceleration_bias` columns held a *position* residual
-- 0.278 where the acceleration residual was 5.32.  The sweep already makes a
second, stride-one acceleration call for Corollary 2's bound, which is the form
Lemma 7 is stated in; the diagnostics now come from that call, so the column
means the same thing whichever form the run was constrained with.

### Editing a running bash script

`orchestrate.sh` was patched while a copy of it was still executing, and bash
reads a script incrementally: the running shell resumed at a byte offset that
no longer began a statement and died on `name: unbound variable`.  Nothing was
lost -- that process had already finished its last phase -- but new phases now
go in only when the orchestrator is idle, and one-off sweeps such as
`work/penalty_far.sh` live in their own file.

## One learning rate does not serve every predictor

Post-training takes a plain SGD step, so the step *length* is `lr ||grad L||`.
`scripts/measure_gradient_scale.py` measures that norm at `theta_0`, 24
batches, median:

| model | `L(theta_0)` | `||grad L||` | step at `lr = 1e-4` | `lr` for step 0.045 |
|---|---|---|---|---|
| `spatial_s42` (GRU) | 48.6 | 556 | 0.056 | `8.1e-5` |
| `lewm1step_s42` | 392.6 | 2 858 | 0.286 | `1.6e-5` |
| `ssm_s42` | 8.26 | 32 768 | 3.28 | `1.4e-6` |

The GRU's `lr = 1e-4` is within 20% of the rule, which is why a single
suite-wide value went unquestioned.  The SSM's gradient is 59x larger; its step
at `lr = 1e-4` is 3.28, and the `1e4` clip engaged at step 0.  That is the
failure this file already warned about for `grad_clip` -- "a clip that engages
on every RESTORE step would silently turn the analysed direction into a
normalised one" -- except it engaged on the OPTIMISE steps as well, from the
first one, giving a step of length 1.0 regardless of the gradient.  Continued
JEPA training *alone* then diverges on the SSM (`L` 8.7 -> 1320 in 200 steps).

Both previously reported SSM failures -- `JEPA L = inf` under Definition 1's
form and `FloatingPointError` under Proposition 1's -- are therefore failures
of the step size, not of the constraint.  The learning rate is now set per
model from the measured norm, by the same rule for every base model, and the
clipped fraction that `SwitchingOptimizer.summary` already reports is the
quantity that should have been read at the first sign of trouble.

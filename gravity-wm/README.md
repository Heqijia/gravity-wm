# Gravity-constrained post-training of JEPA world models

Post-trains the predictor of a pretrained [Semigroup-JEPA](https://github.com/sg-jepa/sg-jepa)
world model so that, for any gravity it is conditioned on, the object motion
encoded in its predicted latents obeys free fall. Physics enters through a
single constraint, evaluated with a frozen position probe on rollouts under
counterfactual gravities for which no video exists. A switching gradient
update alternates between restoring feasibility and decreasing the JEPA loss.

Implements the algorithm analysed in `WM_gravity_theory` and the ablations of
`WM_gravity`.

- **`docs/RESULTS.md`** -- the experiment log: results, protocol, every
  parameter, and an append-only update log. Start here.
  ([`RESULTS.zh.md`](docs/RESULTS.zh.md) is the Chinese version.)
- `docs/design.md` -- every place the implementation departs from those notes,
  and the measurement that forced it.

## Layout

```
gwm/freefall_data.py   contact-free projectile dataset (MuJoCo + closed-form kinematics)
gwm/spatial_lewm.py    latent read-outs: the published [CLS] one and two replacements
gwm/latents.py         pretrained-model loading, module freezing, the latent cache
gwm/probe.py           position probe h and the independent evaluation probe h_eval
gwm/constraint.py      Definition 1: the free-fall violation c and its targets
gwm/switching.py       Algorithm 1: RESTORE / OPTIMISE, PCGrad projection, EMA
gwm/posttrain.py       the driver and the five variants
gwm/evaluate.py        counterfactual sweep, Corollary 2 bound, KKT residuals
```

## Running it

```bash
bash scripts/run_all.sh data pretrain cache probes posttrain evaluate summary
```

Each stage is independent and skips work that is already done, so a stage can
be rerun on its own. Two allocated GPUs are used by running two jobs at a
time; `NGPU`, `SEEDS`, `VARIANTS`, `TAG`, `STRIDE` and the rest are environment
variables with defaults at the top of the script. `SLURM_HOLD_JOBID` must name
a running allocation, because the stages attach to it with `srun --overlap`.

The stages:

| stage | what it does |
|---|---|
| `data` | renders the contact-free projectile episodes |
| `pretrain` | trains Semigroup-JEPA, one model per seed |
| `cache` | encodes every episode once with each frozen encoder |
| `probes` | fits `h` and the independent `h_eval`, then calibrates `eps` |
| `posttrain` | the five variants per seed |
| `evaluate` | counterfactual sweep, Corollary 2 bound, ablation metrics |
| `supervise` | scores checkpoints for probe accuracy and constraint margin |
| `summary` | the comparison table |

Two scripts are useful on their own:

```bash
# Does a checkpoint support the constraint at all?  Probe accuracy and the
# discriminative margin at each candidate stride, from measured quantities.
python scripts/supervise.py --runs RUN_DIR --data DATA --log out.tsv --include-best

# Does the algorithm do what the theory says, with the theory's assumptions met?
# Replaces only the encoder by an affine map of the true state, so the probe is
# accurate and what is measured is Algorithm 1 rather than a checkpoint.
python scripts/control_experiment.py --data DATA --output-dir OUT
```

## Two things worth knowing before you run it

**The published planar datasets cannot be used.** Their windows contain
contacts, which every hypothesis in the notes excludes, and at `fps = 16` in a
view 10.4 world units tall one step of gravity displaces the object by 0.19
pixel. That displacement is the entire signal a second difference of decoded
positions has to resolve, so the constraint would be pure probe noise.
`gwm/freefall_data.py` generates the regime the theory assumes instead.

**The published `[CLS]` read-out does not expose object position.** With the
encoder trained, a probe of the ViT's spatial token map locates the object to
6.2 pixels while the same probe of the latent reaches chance, and no amount of
probe capacity, window length or feature normalisation closes the gap. The
constraint then has a discriminative margin of 0.004 at stride one, rising
only to 0.14 at stride four, where it needs at least about 3.
`--readout spatial_map` keeps the spatial token axis and reaches a sub-pixel
probe within one epoch, at a *lower* rollout loss than the `[CLS]` run; it is
reported as a modification of the published model, not as the published model.
`docs/design.md` has the measurements, `docs/RESULTS.md` the `[CLS]` run itself.

## Tests

```bash
python -m pytest tests/ -q
```

The suite checks the parts where being wrong would be silent: that the strided
generalisation of Lemma 1 gives each term its exact target and that the
violation vanishes on the exact switched trajectory at every stride; that
`c = 0` still pins every predicted position (Proposition 1); that the
projection satisfies Proposition 2; that the switching method converges to a
closed-form constrained optimum and certifies feasibility where a fixed penalty
does not (Remark 10); that the scene's launch algebra matches the rendered
path; and that the frozen latent pathway is bit-identical across forward passes
so the constraint stays a function of one sample.

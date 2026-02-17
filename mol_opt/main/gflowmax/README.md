# GFlowMax (DBReplay-style)

This method integrates a tabular DBReplay-style flow optimizer into the `mol_opt` benchmark loop.

It reuses the same molecular environment as `moldqn` (`main/moldqn/environments/environments.py`) so that:

- chemistry transitions remain aligned with existing methods,
- budget accounting (`max_oracle_calls`) is handled by the benchmark harness,
- results are logged through the same `BaseOptimizer` pipeline.

## Run

```bash
python main/gflowmax/run.py --oracles QED --seed 0
```

To run multiple seeds:

```bash
python main/gflowmax/run.py --oracles QED --seed 0 1 2
```

## Notes

- `score_nonterminal_steps: false` makes GFlowMax query the oracle only on terminal transitions (stop or horizon), which avoids spending oracle budget on intermediate rewards that are not used by the DBReplay loss.
- `treat_noop_as_terminal: true` treats no-op (same SMILES action) as an explicit stop action.

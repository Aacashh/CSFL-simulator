# SCOPE-FD claim verification against `runs/runs_scope_revised`

Re-derives every number in the manuscript straight from the run artifacts, so
the paper can be checked against the data rather than against the aggregation
that produced the figures.

    python load.py       # inventory: families, runs, method-arms
    python tables.py     # Table II, both blocks
    python sweeps.py     # Dirichlet, K, IID
    python rest.py       # scale, cross-dataset, privacy, dropout, public set
    python rest2.py      # the same, sub-grouped by the swept parameter
    python ginilaw2.py   # the exact Gini law, campaign-wide
    python ratio.py      # Fig. 7 ratio-gap and the Spearman correlation

## Notes for anyone re-running this

- A run directory counts as complete if it has `compare_results.json`.
  Older orchestrator versions wrote no `status` key at all, so filtering on
  `status == "complete"` silently drops 8 of the 17 families.
- A "selector run" is one (configuration, method, seed) arm with method in
  {`scope_fd`, `scope_fd_debt_only`}. There are 459 of them outside the
  dropout family, which is the number the paper now cites.
- The dropout family is held out of the Gini-law test because a selected
  client may fail to return, so realised participation is not intended
  participation. The plain arms inside `channel_energy` are *not* held out --
  only the `scope_fd_channel_aware` variant breaks the law there, and it does
  so through the energy budget, exactly as Section VI-J describes.

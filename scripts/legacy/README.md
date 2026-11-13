# Legacy scripts

Scripts that ran the §0a–§0j experiments, committed as they were on 2026-11-13, so that
every number in `docs/design.md` has its driver in the repo. They used the old paths
(`~/m0/...` on the Altra, `~/forge-gpu/...` on the desktop): `docs/storage.md` maps them to
the current tree. They are a record, not something to re-run as-is. New drivers go in
`scripts/<family>/`.

- `ampere/`: from `~/m0/` and `~/` on the Altra (benchmark sweeps, baseline and imatrix drivers, eval
  chains, the MTP graft scripts). `multicorpus.py` was left out because it is a copy of
  `surgery/multicorpus.py`.
- `desktop/cloud-session1/`: the Brev session drivers (design §0j.9).

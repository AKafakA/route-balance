# Archived: the Block and Cara era's documents (migrated 8 Oct 2026)

These are the plans, handoffs, reviews, reports, specifications and notes of the project's earlier eras, copied from
the two repositories that were retired from the operator on 8 Oct 2026 when `route-balance` became the only working
repository:

| folder | from | state copied |
|---|---|---|
| `Block/` | `~/Code/llm/Block` (branch `v3-campaign` at 984ea18, working tree: the Cara and RouteBalance v1 to v3 campaigns, the NeurIPS 2026 submission and rebuttal, the MLSys plans of 24 to 25 Sept) | every `.md`, `.txt`, `.tex`, `.bib` and `.rst` under 2 MB outside the data, model, result and log folders |
| `Block_exp/` | `~/Code/llm/Block_exp` (branch `main` at 90cfd32, 14 commits ahead of its GitHub remote, whose push is disabled: the Block and Astrolabe SoCC and SYSTOR experiments and revision) | the same rule |

Left out on purpose: two files that carry credentials (`cara_paper/active/APR26_PREFLIGHT.md`,
`cara_paper/handoffs/2026-04-15_handoff_v2.md`) and the two token files; they are in the full-tree packs only.

The complete trees, data, models and results are on Drive, `routebalance-backup/archive-block-era/` (the packs
`block-pack-01-data`, `block-pack-02-code-docs-models`, `blockexp-pack-01-all`, with member lists and sha256, each
pack holding the repository's `.git` as well), and Block's git history also as the bundle in
`routebalance-backup/code-pack-02-git-bundles/`. The NeurIPS-era results (`rb-exp`) are the
packs `nips-pack-01` to `03` there and on CSD3 (`hpc-work/llm/rb-exp-nips-packs/`).

These documents are historical records. They are not maintained, they describe code and data that no longer exist
on the operator, and the repository's hygiene gate skips this folder (`docs/archived/`) because they name the
machines and people of their time.

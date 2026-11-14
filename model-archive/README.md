# Canonical P2I model structure archive

This directory in **PtwoI/p2i-kg** owns the persistent generated model structures.
`manifest.json` records catalog revision, generation commit, runtime-source hashes,
model IDs, and the SHA-256 of each `snapshots/<runtime_id>.json` file.

These are inspectable **random-initialized demonstration structures**, often using
smaller configurations. They are not trained weights or trained model checkpoints.
The KG catalog and runtime factories remain in `src/p2i_kg/`.

The portfolio's `apps/p2imap/snapshots/` is only a byte-for-byte deployment copy.
Never edit a snapshot there. See `docs/publication.md` for regeneration and checks.
Git history keeps older archive versions; CI artifacts are disposable build output.

## Initial import

The initial 79 snapshots were copied without modification from the verified KG CI
build for `e54bb63a918e4957ba551ebfca8c173e7034d34f` (the same bytes previously
published to the portfolio). The runtime source files were compared to that Git
revision before recording their hashes. No model was retrained or recreated.

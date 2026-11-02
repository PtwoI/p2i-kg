# Publish the architecture map

Every ontology extension includes the Pages update. Do not stop after the KG PR merges.
The existing architecture-extension automation owns the cross-repository publication,
using the user-authorized GitHub connection. No long-lived deployment token is needed.

1. Merge tested P2I dependencies first, then the KG PR. Pin the latest KG main SHA.
2. The runtime CI job exports and validates artifact `p2imap-<SHA>`, including every
   catalog model and its real P2I structure. Use that artifact, or build identically:

   ```sh
   python -m p2i_kg pages ./new-p2imap --source-revision <KG_MAIN_SHA>
   python -m p2i_kg verify-pages ./new-p2imap
   ```

   Install matching p2i-core/p2i-gui checkouts, CPU Torch/TorchVision and the KG
   language extra. The build creates a temporary database from the bundled catalog;
   a maintainer's stale or locally curated database is never used or overwritten.
   Every model needs a unique runtime and a working scope description. Missing runtime,
   missing/incorrect structure, or changed file checksums must fail publication.
3. Starting from current `K0ykwon/k0ykwon.github.io` main, replace generated files
   only under `apps/p2imap/`, in one commit. Remove only files under that path which
   the preceding manifest managed and the new manifest omits. Preserve all other
   homepage content. Create a PR, wait for checks and merge it.
4. Wait for GitHub Pages deployment success. Compare the live `catalog.json` and
   `viewer-manifest.json` revisions, source SHA, model and snapshot counts with the build.
   Open a new model in the map, click **P2I로 보기**, and verify the real module/operation
   viewer loads under `/apps/p2imap/`.
5. Before publishing, recheck KG main. If newer changes landed, rebuild against that
   SHA. A failure keeps the previously deployed site intact; report the concrete
   blocker and retry on the next extension run, even if no new models were added.

Snapshot factories may use smaller configurations and random weights. Preserve each
model's runtime_note so the web viewer describes its actual scope accurately.

The publication manifest records the source commit, catalog revision, model count,
all runtime snapshot IDs and SHA-256 checksums for generated files. Cache versions
change with the catalog and source commit. The static viewer is read-only; actual
execution and editing remain available in local P2I sessions.

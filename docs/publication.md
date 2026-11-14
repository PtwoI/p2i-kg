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

## Event-driven publisher and safe update plan

The twice-daily extension task researches and adds architectures. A separate
GitHub PR-merge webhook runs publication for merged `PtwoI/p2i-kg` PRs targeting
`main`. Direct pushes are not covered by that webhook. The extension task must
still reconcile publication at the end of every run, including retries with no
new models. GitHub CI builds artifacts; it does not itself write another repo.

The publisher uses the existing authorized GitHub connection, never a new PAT.
After checking successful CI for the pinned KG main SHA, prepare a read-only plan
against a clean, latest homepage checkout:

```sh
python -m p2i_kg publication-plan ./new-p2imap ./homepage \
  --expected-source "$KG_MAIN_SHA" --expected-home "$HOMEPAGE_MAIN_SHA" > plan.json
```

This command neither modifies the checkout nor publishes. It rejects a dirty or
changed homepage, a wrong source SHA, incomplete checksums, symlinks, path
traversal, unchecksummed build files, overwritten customizations, and collisions
with unmanaged files. A repeated source/catalog event is a no-op. The plan lists
only managed `apps/p2imap/` writes/deletes with before/after SHA-256 checksums.
Never silently overwrite a modified managed UI file: port that customization
back to the generator, test it, then regenerate.

The publisher must enforce the remaining transactional requirements:

- Serialize publication per destination repository. Do not merge concurrent or
  older publication PRs. Recheck KG main before preparing and again before merge;
  if it changed, rebuild. Use a branch keyed by source SHA and reuse its open PR
  for duplicate deliveries rather than creating another one.
- Read current homepage main again before creating the commit. Use
  `expected_homepage_revision` as the parent and a non-forced compare-and-swap
  update. On a changed parent, refresh the checkout and regenerate the plan.
  Recheck every `before_sha256` and use only listed paths. Do not stage `git add .`
  or delete an entire directory. Other homepage commits must remain intact.
- Verify the exact PR head, checks and main revisions before merge. A merge or
  push response alone is not deployment success. Wait for Pages and compare the
  live manifest source SHA, catalog revision, exact runtime ID set and counts.
  Open at least one new runtime in the browser. Record and report the verified
  source SHA and Pages run; on failure keep/retry the pending publication.
- A webhook can arrive before main CI finishes. Pending CI is not success or a
  permanent failure: resume after CI, and leave a durable retry when interrupted.

Do not replace this flow with a cross-repository `GITHUB_TOKEN` push. That token
is repository-scoped. Even a same-repository token push does not trigger a Pages
branch build, so a future all-Actions implementation needs an explicitly reviewed
publication/authentication design rather than an assumed push side effect.

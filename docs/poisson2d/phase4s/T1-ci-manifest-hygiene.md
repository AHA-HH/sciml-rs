# T1: CI and manifest hygiene

Phase 4s, task 1. Branch `phase4s/T1-ci-manifest-hygiene`. Depends on nothing. Every other
task starts after this one merges: it fixes the CI that every later PR runs through.

## Read first
- Review §5 S5 ("CI and manifest"), §6 T6-7, §7 step A7.
- `CLAUDE.md` "Checks" and "Build environment".
- `.github/workflows/run-tests.yml`, `.github/workflows/run-examples.yml`, `Cargo.toml`,
  `src/lib.rs:1-55`.

## Goal
Make CI, the manifest and CLAUDE.md say the same thing, and remove CI steps that do nothing.
No Rust behaviour changes.

## Deliverables
1. **Delete `.github/workflows/run-examples.yml`.** Its only job installs
   `cargo-templated-examples` (:26-27) and the one real step is commented out (:29-30). It
   costs a cold `cargo install` on every push. `cargo clippy --examples` in `run-tests.yml`
   already proves the examples compile.
2. **`.github/workflows/run-tests.yml`:**
   - Delete the commented apt blocks in `run-tests-rust` (:30-49) and `style-checks-rust`
     (:71-90).
   - Fix the extra two-space indent of `style-checks-rust`, the whitespace-only lines and the
     trailing space in `cargo test ` (:52).
   - Leave `run-tests-chebyshev` (:100-123) as it is: T15/T16 change its apt step and feature
     name.
3. **`Cargo.toml`:**
   - `homepage` / `repository` (:8-9) point at Codeberg. Point them at the GitHub repository
     (take the URL from `git remote -v`).
   - `categories` (:12): drop `"scientific machine learning"`, which is not a crates.io
     category, and keep `science` and `algorithms`. Check the slugs against
     https://crates.io/category_slugs.
   - Finish the truncated feature comment at :70-73 and remove its trailing whitespace. Say
     which backend burn-signal does not support, rather than "it".
   - Delete the two empty tables `[target.'cfg(target_os= "macos")'.dev-dependencies]`
     (:127) and the Linux one (:129).
4. **`src/lib.rs:22-32`:** the no-backend `compile_error!` message lists `rocm, vulkan`,
   which are commented out in `Cargo.toml` (:79-80). List only the enabled features (`flex`,
   `metal`, `cuda`, `wgpu`), and drop the disabled ones from the `cfg` if they are there.
5. **`CLAUDE.md` "Checks":**
   - Replace "CI … runs exactly" with what the three jobs run, including `run-tests-chebyshev`
     (`clippy --features chebyshev --all-targets`, `cargo test --features chebyshev`).
   - Keep the "Before finishing any task, also run …" line.

## Tests
None added or removed. The test count is unchanged (record the "running N tests" lines).

## Acceptance
- Every CI job on the PR is green, and `run-examples` no longer appears.
- `cargo metadata --no-deps --format-version 1` parses, and `cargo package --list
  --allow-dirty` raises no category warning. If `cargo package` can't run offline, say so.
- `git grep -n codeberg -- Cargo.toml` finds nothing.

## Checks to run and report
The six in the README ("Checks for every task"). The chebyshev pair is not needed: the
`run-tests-chebyshev` job runs it on the PR.

## Do not
- Change the `REVIEW.md` citations. They are part of T24, despite review step A7 listing them.
- Touch `lib.rs:48`'s blanket `#[allow]` (T19), the crate docs (T24) or the chebyshev apt step
  (T15/T16).
- Add new CI jobs or caching.

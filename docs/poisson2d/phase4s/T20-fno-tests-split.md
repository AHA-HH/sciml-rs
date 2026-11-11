# T20: `models/fno.rs` tests split

Phase 4s, task 20. Branch `phase4s/T20-fno-tests-split`. Depends on T5. Can run in parallel
with T21 and anything outside `models/`.

> **Refresh at start:** re-read `fno.rs`'s test helpers after T5, which inlined the grid
> oracle. Re-count the tests (`cargo test neural_operators::models::fno` → "running N
> tests").

## Read first
- Review §6 T5-1, T7-5, §7 Wave E1.
- `models/fno.rs` at `6eb3212`: 1,569 lines, production code :1-306, tests :307-1569.
  - `reference_forward` :522-548, which is exactly `reference_forward_padded(.., Pad::Off)`
  - `reference_forward_padded` :718-771
  - `pre_padding_forward` :774-795
  - the grid helpers `old_grid_f64` / `with_grid` / `with_old_grid` (T5 rewired them)
  - the `perm_in` / `perm_out` closures in `forward` :271-275 and :298-302, repeated in the
    test references

## Goal
Production code is cohesive; the tests are 1,263 lines in one block, with three reference
forwards. Split them by the CONVENTIONS section they verify, and remove the duplicate
reference.

## Deliverables
1. **Module layout:** `models/fno.rs` becomes `models/fno/mod.rs` (production) and
   `models/fno/tests/{mod.rs, reference.rs, padding.rs, grid.rs, checkpoint.rs}`, under
   `#[cfg(test)] mod tests;`.
   - Group the tests by CONVENTIONS section: reference forward / identities (§7),
     padding (§3), grid (§2), checkpoint/record.
   - No test file over about 450 lines.
   - Shared helpers live in `tests/mod.rs`.
2. **Delete `reference_forward`.** Its callers use `reference_forward_padded(.., Pad::Off)`.
3. **Named permutations:**
   - `to_channels_first` / `to_channels_last` become small `pub(crate)` helpers in
     `fno/mod.rs`.
   - `forward` and the test references use them instead of inline closures.
   - The permutation order is unchanged (CONVENTIONS §1).
4. **Grid shims:** retire `with_old_grid` / `old_grid_f64` only if the grid tests can
   assert directly against T5's oracle. Otherwise keep them and say why.
5. **Exact test filters:**
   - Paths change from `models::fno::tests::<name>` to `models::fno::tests::<group>::<name>`.
   - Update every doc or brief that quotes an exact filter (CLAUDE.md's example uses a
     generic `<name>`; check `docs/` for concrete ones).
   - Update the CONVENTIONS "Verification" lists only if test **names** change. They
     shouldn't.

## Tests
**The test count is unchanged.** Compare "running N tests" for
`cargo test neural_operators::models::fno` before and after, and give both numbers in the
PR. The only test-code deletion is the `reference_forward` helper, not a test.

## Acceptance
- `wc -l src/neural_operators/models/fno/**/*.rs`: every file ≤ about 450 lines.
- `cargo test --features metal neural_operators::models::fno` runs the Metal regression
  test (the channels-last grid). Say whether Metal was run.

## Checks to run and report
The six from the README, plus the Metal run above. The chebyshev pair is not needed.

## Do not
- Change production behaviour. `forward` must be bit-identical, which the existing
  bitwise tests check.
- Rename tests, or move the CONVENTIONS prose (T24).

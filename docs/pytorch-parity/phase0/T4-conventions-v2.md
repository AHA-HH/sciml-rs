# pytorch-parity phase 0 / T4 - CONVENTIONS version 2 (C0.4)

Can start at once. **This task needs the user's sign-off in its PR**, before any Phase 1
task starts (design §7, §9.3).

Read first:
- `docs/CONVENTIONS.md`, the whole file;
- `docs/design/pytorch-parity.md` §2.6, the diff to apply;
- the design sections each line refers to: §2.1–§2.5 and §5.4;
- `CLAUDE.md` "Source of truth": a convention changes only through a proposed diff and a
  `CONVENTION_VERSION` bump.

This task delivers `docs/CONVENTIONS.md` at version 2. It contains every convention line
the later phases encode: the defaults (equal to today's behaviour) and the opt-in
alternatives for neuraloperator parity. It lands as one diff with one bump. CONVENTIONS'
preamble requires that a change to §1–§6 bumps the version, and that a convention lands
before the code that relies on it. Every Phase 1–4 task cites these lines instead of
editing the file.

Do:
- `docs/CONVENTIONS.md`: apply the design §2.6 diff, section by section.
  - **Preamble.**
    - `CONVENTION_VERSION = 2`.
    - Add the sentence on what version 2 adds, and the statement that a config with
      every new field `None` is bit-identical to version 1, including its RNG draw
      sequence.
    - Cite `tests/fno_default_params.rs` (T3) as the test that enforces it.
  - **§1.** The `out_channels` line, as in the diff, marked "takes effect in Phase 1
    (C1.4); until then `FNOConfig::init` asserts 1 (`fno.rs:75-78`)".
  - **§2.** The `GridKind::Periodic` option.
  - **§3.** The `domain_padding` option, mutually exclusive with `padding`.
  - **§4.**
    - Replace the "Not yet fixed by a test" bullet with the resolved statement, citing
      the test `irfft_ignores_dc_nyquist_imag` (T2) by that exact name. Wording:
      "`irfft` ignores them: exactly for power-of-two n, to f32 round-off for other n
      (Bluestein)".
    - Add the `output_shape` resize line, marked "**Pending Q7** (design §9.2); decided
      in Phase 4".
  - **§5.** The neuraloperator `n_modes` mapping; the bias, `clip_modes`,
    `max_modes`/`set_modes`, separable and factorised options.
  - **§6.** The `NeuralopNormal` row, with the σ²/2 per-part variance marked "to be
    confirmed in Phase 1 (C1.1)".
  - **§8.** The `LpLoss` `eps`/`measure`/`rel_nd` options and `H1Loss`, citing design
    §2.4.
  - **Every new line names its `FNOConfig` or `SpectralConv` field, and says that
    `None` is the version-1 behaviour.** Use the field names of design §5.2. Design §5.2
    names no `SpectralConv` field for some options: write "field name settled in C1.1"
    for `clip_modes` and separable, and "settled in C2.2" for `max_modes`/`set_modes`.
    Do not invent names.
  - **Ordering with T3.** The preamble cites `tests/fno_default_params.rs`, which T3
    adds. Merge T4 after T3, or say in the PR that the file arrives with T3.
  - Keep the file's style: short bullets, `CONVENTIONS §n` cross-references, no
    derivations. Cite "design `pytorch-parity` §n" for detail rather than copying it.
- **PR description.**
  - Paste the diff.
  - List every line that changes behaviour only behind an `Option`.
  - State that no default changes, and that no code changes in this PR.
  - Ask the user explicitly for sign-off, and list the two pending items (Q7, and the
    σ²/2 variance).

Tests that define done (a documentation task: these are checks, not unit tests):
- `grep -n "CONVENTION_VERSION" docs/CONVENTIONS.md` shows `= 2`.
- **Every option is covered.** Every option in design §2.6 appears in the file, and no
  option outside §2.6 does. Check the PR diff against §2.6 line by line, and list any
  wording changes.
- **No version-1 statement is deleted or weakened**, except:
  - the §4 "Not yet fixed by a test" bullet, which is replaced;
  - the §1 `out_channels` line, which is generalised.
- **The test name** `irfft_ignores_dc_nyquist_imag` appears exactly once in the file.
- **`CLAUDE.md` still agrees** with the conventions file. It cites CONVENTIONS only by
  section number, so no `CLAUDE.md` change is expected. If one is needed, report it
  instead of making it.

Must pass:
- `cargo fmt -- --check` and `cargo doc --no-deps`. No Rust changes, so this confirms
  nothing else was touched.
- `git diff --stat main` shows only `docs/CONVENTIONS.md`.

Do not:
- Change any code or test.
- Add conventions that are not in design §2.6. If you find one missing, list it in the
  PR for the user to decide.
- Resolve Q7 or the σ²/2 variance. Keep them marked as pending.
- Edit the design document. If §2.6 and this file need to differ, describe it in the
  PR; CONVENTIONS wins.

# T21: `SpectralConv` polish

Phase 4s, task 21. Branch `phase4s/T21-spectral-conv-polish`. Depends on T1. Can run fully
in parallel with any task: it touches only `layers/spectral_convolution.rs`.

## Read first
- Review §6 T7-4, §7 Wave E2. CONVENTIONS §5 (spectral convolution layout) and §6
  (initialisation).
- `src/neural_operators/layers/spectral_convolution.rs` (853 lines).

## Goal
Small legibility fixes in a cohesive hot file: one helper for a repeated idiom, one fewer
piece of stored state, and a doc formula that matches the code.

## Deliverables
1. **A helper** `fn with_bc<T: Copy, const R: usize>(a: T, b: T, rest: &[T]) -> [T; R]`
   (name it to taste). It replaces the five `vec![a, b]; extend(..); try_into::<[_;
   R]>().unwrap()` sites:
   - :179-181 (weight shape)
   - :281-283 (`out_shape` in `complex_multiplication`)
   - :356-358 (`out_shape`)
   - :368-370 and :384-386 (input and output ranges)

   Document its panic (`rest.len() != R − 2`).
2. **`flat_modes`:**
   - Today it is computed (:176), stored as a field (:91, :206) and passed back in as `flat`
     (:263, :381).
   - Compute it locally where it's needed and drop the field.
   - **Removing it is expected to be safe for old runs.** It is a plain `usize` field
     (:91) of a `#[derive(Module)]` struct. Burn treats such fields as constants, not saved
     tensors, so the `.bpk` record should not contain it. Confirm with the `load_fno`
     round-trip test and by loading an existing run. If an old run no longer loads, keep the
     field and say so in the PR.
3. **The doc formula at :251** states the four-multiply product
   (a + bi)(c + di) = (ac − bd) + (ad + bc)i. The code uses Gauss's 3-matmul form
   (:274-279): `ab_cd = (x_re + x_im)(w_re + w_im)`, `out_re = ac − bd`,
   `out_im = ab_cd − ac − bd`. Document that form and why it is used (3 matmuls instead
   of 4).

## Tests
- Every `spectral_convolution` test passes unchanged, including the bitwise ones.
- Add a tiny test for the helper.
- The test count rises by at most 1.

## Acceptance
- A saved run still loads: `predict_poisson` on `runs/poisson_fno_p4_1791922918`, or the
  `load_fno` round-trip test. This matters if `flat_modes` was touched.
- No behaviour change.

## Checks to run and report
The six from the README. Also run `--features metal` on the spectral tests, since the layer
is backend-sensitive, and say whether it ran.

## Do not
- Change the FFT path, the corner layout or the initialisation.
- Change `#[allow(clippy::derivable_impls)]` (:54) unless it is no longer needed.

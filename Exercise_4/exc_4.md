# Exercise 4: Massive Dataset Generation for SHC-PWM

## Objective

Scale from single operating point optimization to the automated generation
of a 10,000-point dataset for the SHC-PWM problem. The exercise implements
a homotopy continuation strategy: the solution of each operating point is
used as the seed for the next, with the HDE solver acting as a fallback
when the trajectory breaks.

The resulting CSV file is the mandatory input for the neural network
training exercises in Day 2.

## Problem Setup

- **N = 17** switching angles, Half-Wave Symmetry (HWS).
- **Fixed:** m₁ = 0.85.
- **Swept:** m₅ ∈ [0.01, 0.15] with 100 steps.
- **Swept:** φ₅ ∈ [−π, π] with 100 steps.
- **Total:** 100 × 100 = **10,000 operating points**.

## Prerequisite

`hybrid_differential_evolution` must be installed in SciPy.
See `exercise_3/exc_3.md` for instructions.

## Instructions

Open `exercise_4_base.py` and implement the logic inside `process_grid`.

### Task: Homotopy Continuation and Fallback

For each `(m_ref, phi_ref)` pair in the grid, implement the following
sequential logic:

1. **Primary attempt:** call `solve_sqp` using `alpha_prev` as the
   initial guess `x0`.
2. **Evaluate:** if the returned error exceeds `config["FITNESS_LIMIT"]`,
   the homotopy trajectory broke. The SQP diverged or converged to a
   local minimum.
3. **Fallback:** call `solve_hde` to recover the global optimum
   from scratch.
4. **Update and save:** if the final solution is valid
   (error ≤ FITNESS_LIMIT and angles is not None):
   - Update `alpha_prev` with the new angles.
   - Build the row: `m_ref + [phi_ref[1]] + list(angles)`.
   - Append the row to `dataset`.

## Analysis

Monitor the console progress bar during execution:

1. Observe that SQP resolves hundreds of points per second by inheriting
   highly accurate seeds.
2. Identify the temporary slowdowns — these correspond to topological
   breaks in the solution trajectory where the HDE fallback was invoked.
3. Verify the final count: how many of the 10,000 points were solved
   successfully? What fraction required HDE rescue?

## Output

The script generates `dataset_shcpwm_10000.csv` in the current directory
with the following columns:

| Column | Description |
|---|---|
| `m1`, `m5`, `m7`, `m11`, `m13`, `m17`, `m19` | Harmonic magnitude references |
| `phi5` | Phase reference for the 5th harmonic (radians) |
| `alpha_1` ... `alpha_17` | Optimal switching angles (radians) |

This file must be placed in the `data/` folder of the Day 2 directory
before running any Day 2 exercise.

## Files

- `exercise_4_base.py` — incomplete script for in-class work.
- `exercise_4_solution.py` — complete reference solution.
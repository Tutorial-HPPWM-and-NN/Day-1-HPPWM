# Exercise 2: Global Solver (Differential Evolution)

## Objective

Demonstrate how a metaheuristic algorithm can solve the HPPWM problem
autonomously, without any initial guess, overcoming the local minimum
trapping observed in Exercise 1. The exercise also highlights the
computational cost of pure global search.

## Problem Setup

Identical to Exercise 1:
- **N = 17** switching angles, Half-Wave Symmetry (HWS).
- **Harmonics:** h = {1, 5, 7, 11, 13, 17, 19}.
- **Target:** m₁ = 0.85, all other harmonics eliminated.

## Instructions

Open `exercise_2_base.py` and complete the following tasks.

### Task 1: Pre-conditioned Initial Population

Before calling the solver, generate a list of 100 individuals:
- For each individual, generate a random array of `N_ANGLES` values
  between 0.01 and π.
- Apply `enforce_constraints()` to sort and bound each individual.
- Convert the list to a numpy array.

This pre-conditioning ensures every individual respects the monotonicity
constraint from the first generation, avoiding wasted evaluations on
physically invalid angle sequences.

### Task 2: Configure the DE Solver

Call `scipy.optimize.differential_evolution` with:
- `bounds`, `init=initial_population`
- `maxiter=150`, `popsize=100`, `tol=1e-5`
- `mutation=(0.5, 0.6)`, `recombination=0.9`
- `polish=True` — applies L-BFGS-B refinement after convergence
- `disp=True` — shows generation-by-generation progress

## Analysis

After running the script, compare with Exercise 1:

1. Did the solver find a valid solution (`Final Error < 1e-4`) without
   any initial guess?
2. What is the total execution time? How does it compare to the
   milliseconds required by SLSQP in Exercise 1?
3. How many function evaluations (`nfev`) were required?
4. Consider: is it viable to use pure DE to generate a dataset of one
   million operating points?

## Output

No files are generated. Analysis is based on console output.

## Files

- `exercise_2_base.py` — incomplete script for in-class work.
- `exercise_2_solution.py` — complete reference solution.
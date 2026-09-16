# Exercise 3: Hybrid Differential Evolution (HDE-SQP)

## Objective

Implement a true Hybrid Differential Evolution (HDE) by injecting an SQP
local refinement step directly inside the evolutionary loop. This exercise
demonstrates that hybridization achieves both global robustness and
quadratic terminal convergence, using a fraction of the population and
iterations required by pure DE.

## Prerequisite: Installing the Hybrid Solver

This exercise requires modifying the SciPy source code in your Python
environment. **Using a virtual environment is strongly recommended.**

Follow these steps before running the script:

1. Find your SciPy installation path:
   ```python
   import scipy; print(scipy.__file__)
   ```

2. Navigate to the `scipy/optimize/` directory within that path.

3. Copy `_hybrid_differential_evolution.py` (provided at the root of the
   Day 1 repository) into that directory.

4. Open `__init__.py` in the same `scipy/optimize/` directory with a
   text editor.

5. Find the line:
   ```python
   from ._differentialevolution import differential_evolution
   ```

6. Add the following line immediately below it:
   ```python
   from ._hybrid_differential_evolution import hybrid_differential_evolution
   ```

7. Save `__init__.py`.

8. Verify the installation by running:
   ```python
   from scipy.optimize import hybrid_differential_evolution
   print("Installation successful.")
   ```

## Instructions

Open `exercise_3_base.py` and complete the following task.

### Task 1: Call the Hybrid Solver

Locate `run_hde_solver` and call `hybrid_differential_evolution` with:
- `bounds`, `init=init_pop`
- `maxiter=100`, `popsize=20`
- `mutation=(0.5, 0.6)`, `recombination=0.9`
- `g_hybrid=5` — SQP is injected every 5 generations

Note the drastically reduced `popsize` (20 vs 100 in Exercise 2) and
`maxiter` (100 vs 150). The hybrid approach compensates with the SQP
refinement step.

## Analysis

1. Compare the execution time with Exercise 2. The HDE should converge
   in a fraction of the time despite using a much smaller population.
2. Experiment with different values of `g_hybrid` (3, 5, 10, 20).
   How does the hybridization frequency affect convergence speed and
   solution quality?
3. Try reducing `popsize` to 10 or even 5. At what point does the HDE
   start failing to find the global optimum?

## Output

No files are generated. Analysis is based on console output.

## Files

- `exercise_3_base.py` — incomplete script for in-class work.
- `exercise_3_solution.py` — complete reference solution.
- `_hybrid_differential_evolution.py` — modified SciPy module (root of Day 1).
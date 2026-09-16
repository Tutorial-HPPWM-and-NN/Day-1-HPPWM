import numpy as np
import pandas as pd
from scipy.optimize import minimize, hybrid_differential_evolution
from tqdm import tqdm
import time

# =============================================================================
# EXERCISE 4: MASSIVE DATASET GENERATION (SHC-PWM) WITH HOMOTOPY TRACKING
#             — SOLUTION
#
# PREREQUISITE: hybrid_differential_evolution must be installed in SciPy.
# See exercise_3/exc_3.md for instructions.
# =============================================================================

OUTPUT_FILE = "dataset_shcpwm_10000.csv"


def get_config():
    """Configuration for the 2D grid sweep (10,000 operating points)."""
    return {
        "N_ANGLES": 17,
        "H_VALUES": [1, 5, 7, 11, 13, 17, 19],
        "ERROR_WEIGHTS": [1, 1, 1, 1, 1, 1, 1],
        "MIN_ANGLE_SEP": 0.001,
        "FITNESS_LIMIT": 1e-4,
        "M1_FIXED": 0.85,
        "M5_SWEEP": np.round(np.linspace(0.01, 0.15, 100), 4),
        "PHI5_SWEEP": np.round(np.linspace(-np.pi, np.pi, 100), 4),
    }


def calculate_fourier_coefficients(alpha, h):
    """
    Calculates the Fourier coefficients (a_h, b_h) for harmonic order h
    under Half-Wave Symmetry (HWS) with a fixed alternating switching
    polarity, q_k = (-1)**k. This is the classical HWS formula, NOT the
    virtual-angles technique: here polarity is never a free variable
    (it is fixed by index parity), so there is no q_k to eliminate.
    Virtual angles are a different, more general technique used when
    polarity itself must be optimized (see the Day 1 book chapter,
    "Tratamiento de la Variable Discreta").
    """
    a_h = -(2.0 / (h * np.pi)) * np.sum(
        [(-1) ** (k + 1) * np.sin(h * alpha[k]) for k in range(len(alpha))]
    )
    b_h = (2.0 / (h * np.pi)) * np.sum(
        [(-1) ** k * np.cos(h * alpha[k]) for k in range(len(alpha))]
    )
    return a_h, b_h


def enforce_constraints(alpha):
    """Ensures angles are sorted and strictly bounded within (0, pi)."""
    return np.clip(np.sort(alpha), 0.001, np.pi - 0.001)


def cost_function(alpha, m_ref, phi_ref, config):
    """
    Evaluates the SHC-PWM cost using Cartesian coordinates.
    """
    alpha = np.sort(alpha)
    total_error = 0.0

    for i, h in enumerate(config["H_VALUES"]):
        a_h, b_h = calculate_fourier_coefficients(alpha, h)
        a_ref = (m_ref[i] / 2.0) * np.cos(phi_ref[i])
        b_ref = (m_ref[i] / 2.0) * np.sin(phi_ref[i])
        error_h = (a_h - a_ref) ** 2 + (b_h - b_ref) ** 2
        total_error += config["ERROR_WEIGHTS"][i] * error_h

    if np.any(np.diff(alpha) < config["MIN_ANGLE_SEP"]):
        total_error += 1e6

    return total_error


def solve_sqp(x0, m_ref, phi_ref, config):
    """Fast local solver used as the primary tracker in homotopy continuation."""
    bounds = [(0.01, np.pi)] * config["N_ANGLES"]
    result = minimize(
        lambda x: cost_function(x, m_ref, phi_ref, config),
        x0,
        method="SLSQP",
        bounds=bounds,
        tol=1e-5,
    )
    angles = enforce_constraints(result.x)
    return angles, cost_function(angles, m_ref, phi_ref, config)


def solve_hde(m_ref, phi_ref, config):
    """
    Fallback global solver invoked when the homotopy trajectory breaks.
    Uses a small population since it is only called for recovery.
    """
    bounds = [(0.01, np.pi)] * config["N_ANGLES"]
    popsize = 20
    init_pop = np.array([
        enforce_constraints(np.random.uniform(0.01, np.pi, config["N_ANGLES"]))
        for _ in range(popsize)
    ])
    try:
        result = hybrid_differential_evolution(
            lambda x: cost_function(x, m_ref, phi_ref, config),
            bounds=bounds,
            maxiter=100,
            popsize=popsize,
            mutation=(0.5, 0.9),
            recombination=0.9,
            init=init_pop,
            g_hybrid=10,
            disp=False,
        )
        angles = enforce_constraints(result.x)
        return angles, cost_function(angles, m_ref, phi_ref, config)
    except Exception:
        return None, 1e6


def build_grid(config):
    """Builds the list of all (m_ref, phi_ref) operating points to solve."""
    grid = []
    for m5 in config["M5_SWEEP"]:
        for phi5 in config["PHI5_SWEEP"]:
            m_ref   = [config["M1_FIXED"], m5, 0, 0, 0, 0, 0]
            phi_ref = [0, phi5, 0, 0, 0, 0, 0]
            grid.append((m_ref, phi_ref))
    return grid


def process_grid(grid, config, initial_seed):
    """
    Sweeps the operating point grid using homotopy continuation.

    Strategy:
        1. Try the fast SQP solver using the previous solution as seed.
        2. If SQP fails (error > FITNESS_LIMIT), the trajectory broke.
           Invoke the HDE fallback to recover the global optimum.
        3. Update the tracking seed with the new valid solution.

    Args:
        grid: list of (m_ref, phi_ref) tuples.
        config: configuration dictionary.
        initial_seed: starting angle vector for the first operating point.

    Returns:
        List of dataset rows (inputs + outputs).
    """
    dataset = []
    alpha_prev = initial_seed.copy()
    hde_rescues = 0

    print(f"--- Processing {len(grid)} operating points ---")

    for m_ref, phi_ref in tqdm(grid, desc="Generating Dataset"):
        # Primary: fast homotopy tracking via SQP
        angles, error = solve_sqp(alpha_prev, m_ref, phi_ref, config)

        # Fallback: global recovery via HDE if trajectory broke
        if error > config["FITNESS_LIMIT"]:
            angles, error = solve_hde(m_ref, phi_ref, config)
            hde_rescues += 1

        # Save and update seed only if the solution is valid
        if error <= config["FITNESS_LIMIT"] and angles is not None:
            alpha_prev = angles
            row = m_ref + [phi_ref[1]] + list(angles)
            dataset.append(row)

    print(f"Valid points : {len(dataset)} / {len(grid)}")
    print(f"HDE rescues  : {hde_rescues}")
    return dataset


if __name__ == "__main__":
    config = get_config()
    grid   = build_grid(config)

    # Seed from a known valid solution at m1=0.85, pure SHE
    initial_seed = np.array([
        0.2579, 0.3283, 0.5190, 0.5821, 0.7649, 0.8342, 0.9928, 1.0723,
        1.1922, 1.2500, 1.4671, 1.5514, 1.6984, 1.8220, 1.9347, 2.0878, 2.1765,
    ])

    start_time = time.time()
    dataset    = process_grid(grid, config, initial_seed)

    if dataset:
        columns = (
            [f"m{h}" for h in config["H_VALUES"]]
            + ["phi5"]
            + [f"alpha_{i+1}" for i in range(config["N_ANGLES"])]
        )
        df = pd.DataFrame(dataset, columns=columns)
        df.to_csv(OUTPUT_FILE, index=False)
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f}s.")
        print(f"Saved {len(df)} points to {OUTPUT_FILE}.")
    else:
        print("\nFailed: no valid points were generated.")

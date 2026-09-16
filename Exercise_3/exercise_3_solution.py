import numpy as np
import time
from scipy.optimize import minimize, hybrid_differential_evolution

# =============================================================================
# EXERCISE 3: HYBRID DIFFERENTIAL EVOLUTION (HDE-SQP) — SOLUTION
#
# PREREQUISITE: Before running this script, the hybrid solver must be
# installed into your SciPy environment. Follow these steps:
#
#   1. Find your SciPy installation path:
#      >>> import scipy; print(scipy.__file__)
#
#   2. Navigate to the scipy/optimize/ directory.
#
#   3. Copy _hybrid_differential_evolution.py (provided in this repository)
#      into that directory.
#
#   4. Open __init__.py in the same directory and add the following line
#      immediately after the differential_evolution import:
#        from ._hybrid_differential_evolution import hybrid_differential_evolution
#
#   5. Save __init__.py.
#
# Using a virtual environment is strongly recommended to isolate this change.
# =============================================================================


def get_config():
    """Configuration for a single SHE-PWM operating point."""
    return {
        "N_ANGLES": 17,
        "H_VALUES": [1, 5, 7, 11, 13, 17, 19],
        "ERROR_WEIGHTS": [1, 1, 1, 1, 1, 1, 1],
        "MIN_ANGLE_SEP": 0.001,
        "FITNESS_LIMIT": 1e-4,
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


def run_hde_solver(m_ref, phi_ref, config):
    """
    Executes the Hybrid Differential Evolution (HDE-SQP) solver.

    The HDE injects an SQP local refinement step every g_hybrid generations.
    This allows the algorithm to escape local minima stochastically while
    converging quadratically in the terminal phase, using a much smaller
    population and fewer iterations than pure DE.

    Args:
        m_ref: list of target harmonic magnitudes.
        phi_ref: list of target harmonic phases.
        config: configuration dictionary.

    Returns:
        Tuple (final_angles, final_error).
    """
    bounds = [(0.01, np.pi)] * config["N_ANGLES"]
    popsize = 20

    print("\n--- Running Hybrid Solver (HDE-SQP) ---")
    start_time = time.time()

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
            mutation=(0.5, 0.6),
            recombination=0.9,
            init=init_pop,
            g_hybrid=5,   # SQP is injected every 5 generations
            disp=True,
        )

        elapsed = time.time() - start_time
        final_angles = enforce_constraints(result.x)
        final_error  = cost_function(final_angles, m_ref, phi_ref, config)

        print(f"Success Flag : {result.success}")
        print(f"Exec. Time   : {elapsed:.3f} seconds")
        print(f"Final Error  : {final_error:.2e}")

        if final_error <= config["FITNESS_LIMIT"]:
            print("Result       : VALID SOLUTION FOUND")
        else:
            print("Result       : FAILED TO CONVERGE")

        return final_angles, final_error

    except ImportError:
        print("ERROR: hybrid_differential_evolution not found in scipy.optimize.")
        print("Please follow the installation steps at the top of this file.")
        return None, None


if __name__ == "__main__":
    config = get_config()

    m_ref   = [0.85, 0, 0, 0, 0, 0, 0]
    phi_ref = [0, 0, 0, 0, 0, 0, 0]

    run_hde_solver(m_ref, phi_ref, config)

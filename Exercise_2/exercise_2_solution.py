import numpy as np
from scipy.optimize import differential_evolution
import time

# =============================================================================
# EXERCISE 2: GLOBAL SOLVER (DIFFERENTIAL EVOLUTION) — SOLUTION
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
    using the Virtual Angles formulation for Half-Wave Symmetry (HWS).
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


def run_de_solver(m_ref, phi_ref, config):
    """
    Executes the Differential Evolution global solver.

    The initial population is pre-conditioned: each individual is generated
    randomly and then sorted via enforce_constraints to ensure physical
    validity from the first generation. This avoids wasting evaluations on
    infeasible angle sequences.

    Args:
        m_ref: list of target harmonic magnitudes.
        phi_ref: list of target harmonic phases.
        config: configuration dictionary.

    Returns:
        Tuple (final_angles, final_error).
    """
    bounds = [(0.01, np.pi)] * config["N_ANGLES"]
    print("\n--- Running Global Solver (Differential Evolution) ---")

    start_time = time.time()

    # Pre-conditioned initial population: each individual respects
    # the monotonicity constraint from the start.
    popsize = 100
    initial_population = np.array([
        enforce_constraints(np.random.uniform(0.01, np.pi, config["N_ANGLES"]))
        for _ in range(popsize)
    ])

    result = differential_evolution(
        lambda x: cost_function(x, m_ref, phi_ref, config),
        bounds=bounds,
        maxiter=150,
        popsize=popsize,
        tol=1e-5,
        mutation=(0.5, 0.6),
        recombination=0.9,
        init=initial_population,
        polish=True,
        disp=True,
    )

    elapsed = time.time() - start_time
    final_angles = enforce_constraints(result.x)
    final_error  = cost_function(final_angles, m_ref, phi_ref, config)

    print(f"Success Flag : {result.success}")
    print(f"Generations  : {result.nit}")
    print(f"Evaluations  : {result.nfev}")
    print(f"Exec. Time   : {elapsed:.2f} seconds")
    print(f"Final Error  : {final_error:.2e}")

    if final_error <= config["FITNESS_LIMIT"]:
        print("Result       : VALID SOLUTION FOUND")
    else:
        print("Result       : FAILED TO CONVERGE")

    return final_angles, final_error


if __name__ == "__main__":
    config = get_config()

    # Same target as Exercise 1 — pure SHE-PWM, no initial guess required
    m_ref   = [0.85, 0, 0, 0, 0, 0, 0]
    phi_ref = [0, 0, 0, 0, 0, 0, 0]

    run_de_solver(m_ref, phi_ref, config)

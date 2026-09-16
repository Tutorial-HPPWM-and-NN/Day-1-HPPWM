import numpy as np
from scipy.optimize import differential_evolution
import time

# =============================================================================
# EXERCISE 2: GLOBAL SOLVER (DIFFERENTIAL EVOLUTION)
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


def run_de_solver(m_ref, phi_ref, config):
    """
    Executes the Differential Evolution global solver.
    """
    bounds = [(0.01, np.pi)] * config["N_ANGLES"]
    print("\n--- Running Global Solver (Differential Evolution) ---")

    start_time = time.time()

    # TODO 1: Generate a pre-conditioned initial population of 100 individuals.
    # Each individual must be a random array of N_ANGLES values passed through
    # enforce_constraints() to guarantee physical validity from the start.
    # Store the result as a numpy array called initial_population.

    # initial_population = ...  # Complete here

    # TODO 2: Call scipy.optimize.differential_evolution.
    # - Pass cost_function via lambda and bounds.
    # - Set init=initial_population.
    # - Set maxiter=150, popsize=100, tol=1e-5.
    # - Set mutation=(0.5, 0.6), recombination=0.9.
    # - Enable polish=True and disp=True.

    # result = differential_evolution(...)  # Uncomment and complete

    # NOTE: Keep these lines commented until TODOs are complete.
    """
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
    """
    return None, None


if __name__ == "__main__":
    config = get_config()

    m_ref   = [0.85, 0, 0, 0, 0, 0, 0]
    phi_ref = [0, 0, 0, 0, 0, 0, 0]

    run_de_solver(m_ref, phi_ref, config)

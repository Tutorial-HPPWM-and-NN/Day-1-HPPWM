import numpy as np
import time
from scipy.optimize import minimize, hybrid_differential_evolution

# =============================================================================
# EXERCISE 3: HYBRID DIFFERENTIAL EVOLUTION (HDE-SQP)
#
# PREREQUISITE: Before running this script, install the hybrid solver into
# your SciPy environment. See exc_3.md for step-by-step instructions.
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


def run_hde_solver(m_ref, phi_ref, config):
    """
    Executes the Hybrid Differential Evolution (HDE-SQP) solver.
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
        # TODO 1: Call hybrid_differential_evolution.
        # - Pass cost_function via lambda and bounds.
        # - Set maxiter=100, popsize=popsize.
        # - Set mutation=(0.5, 0.6), recombination=0.9.
        # - Set init=init_pop.
        # - Activate hybridization with g_hybrid=5.

        # result = hybrid_differential_evolution(...)  # Uncomment and complete

        # NOTE: Keep these lines commented until TODO 1 is complete.
        """
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
        """
        pass

    except ImportError:
        print("ERROR: hybrid_differential_evolution not found in scipy.optimize.")
        print("Please follow the installation steps in exc_3.md.")

    return None, None


if __name__ == "__main__":
    config = get_config()

    m_ref   = [0.85, 0, 0, 0, 0, 0, 0]
    phi_ref = [0, 0, 0, 0, 0, 0, 0]

    run_hde_solver(m_ref, phi_ref, config)

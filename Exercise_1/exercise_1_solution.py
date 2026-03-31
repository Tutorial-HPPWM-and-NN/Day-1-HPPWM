import numpy as np
from scipy.optimize import minimize

# =============================================================================
# EXERCISE 1: LOCAL SOLVER (SQP) AND INITIAL GUESS DEPENDENCE — SOLUTION
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

    Args:
        alpha: array of switching angles (radians).
        h: harmonic order.

    Returns:
        Tuple (a_h, b_h) of Fourier coefficients.
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

    Cartesian projection avoids the atan2 discontinuities that arise
    when working directly with magnitude and phase. The error for each
    harmonic is the squared Euclidean distance between the computed and
    reference Fourier coefficients.

    Args:
        alpha: candidate switching angle vector.
        m_ref: list of target harmonic magnitudes.
        phi_ref: list of target harmonic phases (radians).
        config: configuration dictionary.

    Returns:
        Scalar total error.
    """
    alpha = np.sort(alpha)
    total_error = 0.0

    for i, h in enumerate(config["H_VALUES"]):
        a_h, b_h = calculate_fourier_coefficients(alpha, h)

        # Project polar reference to Cartesian coordinates
        a_ref = (m_ref[i] / 2.0) * np.cos(phi_ref[i])
        b_ref = (m_ref[i] / 2.0) * np.sin(phi_ref[i])

        # Squared Euclidean distance
        error_h = (a_h - a_ref) ** 2 + (b_h - b_ref) ** 2
        total_error += config["ERROR_WEIGHTS"][i] * error_h

    # Dead-time physical constraint: penalty for overlapping angles
    if np.any(np.diff(alpha) < config["MIN_ANGLE_SEP"]):
        total_error += 1e6

    return total_error


def run_sqp_solver(x0, m_ref, phi_ref, config, label=""):
    """
    Executes the SQP local solver (SLSQP) from a given initial guess.

    Args:
        x0: initial guess vector for the switching angles.
        m_ref: list of target harmonic magnitudes.
        phi_ref: list of target harmonic phases.
        config: configuration dictionary.
        label: descriptive label for console output.

    Returns:
        Tuple (final_angles, final_error).
    """
    bounds = [(0.01, np.pi)] * config["N_ANGLES"]
    print(f"\n--- Running SQP Solver: {label} ---")

    result = minimize(
        lambda x: cost_function(x, m_ref, phi_ref, config),
        x0,
        method="SLSQP",
        bounds=bounds,
        tol=1e-5,
    )

    final_angles = enforce_constraints(result.x)
    final_error = cost_function(final_angles, m_ref, phi_ref, config)

    print(f"Success Flag : {result.success}")
    print(f"Iterations   : {result.nit}")
    print(f"Final Error  : {final_error:.2e}")

    if final_error <= config["FITNESS_LIMIT"]:
        print("Result       : VALID SOLUTION FOUND")
    else:
        print("Result       : STUCK IN LOCAL MINIMUM")

    return final_angles, final_error


if __name__ == "__main__":
    config = get_config()

    # Target: m_1 = 0.85, all other harmonics eliminated (pure SHE-PWM)
    m_ref   = [0.85, 0, 0, 0, 0, 0, 0]
    phi_ref = [0, 0, 0, 0, 0, 0, 0]

    # Scenario A: informed initial guess (from pre-computed offline data)
    x0_good = np.array([
        0.2579, 0.3283, 0.5190, 0.5821, 0.7649, 0.8342, 0.9928, 1.0723,
        1.1922, 1.2500, 1.4671, 1.5514, 1.6984, 1.8220, 1.9347, 2.0878, 2.1765,
    ])

    # Scenario B: biased initial guess (angles clustered in the first quarter)
    # This forces the solver to navigate blindly and demonstrates local trapping.
    x0_bad = np.linspace(0.05, np.pi / 2, config["N_ANGLES"])

    run_sqp_solver(x0_good, m_ref, phi_ref, config, label="GOOD INITIAL GUESS")
    run_sqp_solver(x0_bad,  m_ref, phi_ref, config, label="BAD INITIAL GUESS (TRAPPED)")

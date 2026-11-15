import numpy as np
from numpy import float64 as f64


def beta(Gamma: f64) -> f64:
    """Calculate the beta factor from the Lorentz factor Gamma.

    Args:
        Gamma (f64): The Lorentz factor.

    Returns:
        beta (f64): The beta factor, which is the ratio of the particle's velocity to the speed of light.
    """
    return np.sqrt(1 - 1 / Gamma**2)

def Gamma(beta: f64) -> f64:
    """Calculate the Lorentz factor Gamma from the beta factor.

    Args:
        beta (f64): The beta factor, which is the ratio of the particle's velocity to the speed of light.

    Returns:
        Gamma (f64): The Lorentz factor.
    """
    return 1 / np.sqrt(1 - beta**2)

def m_decleration(M0:f64, Gamma:f64) -> f64:
    """Calculate the decelerated mass of a particle given its rest mass and Lorentz factor.

    Args:
        M0 (f64): The rest mass of the particle.
        Gamma (f64): The Lorentz factor.

    Returns:
        M (f64): The decelerated mass of the particle.
    """
    return 3*M0/2/Gamma


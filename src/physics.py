# src/physics.py

import numpy as np

k_B = 1.38e-16  # Boltzmann constant in erg/K
mH = 1.67e-24   # Hydrogen mass in g

def cooling_function(T):
    """
    Cooling function Lambda(T) in erg cm^3 s^-1.
    """
    logT = np.log10(T / 1e5)
    Theta = 0.4 * logT - 3 + 5.2 / (np.exp(logT + 0.08) + np.exp(-1.5 * (logT + 0.08)))
    return 1.1e-21 * 10**Theta


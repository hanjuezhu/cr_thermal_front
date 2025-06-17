# src/solver.py

import numpy as np
from scipy.integrate import solve_ivp
from src.physics import k_B, mH, cooling_function


def system_of_equations(x, state, **params):
    """
    Second-order ODE for the number density n(x) across a cosmic-ray mediated thermal front.
    Implements Eq. 7 in the paper,
    written in terms of n(x), T(n), and dT/dn.

    Parameters
    ----------
    x : float or ndarray
        Spatial coordinate [cm]
    state : list or ndarray
        [n(x), dn/dx(x)] where n is number density [cm^-3], dn/dx is density gradient [cm^-4]
    params : dict
        Physical parameters such as v0, T0, B, alpha, n0

    Returns
    -------
    [dn/dx, d²n/dx²] : list
        First and second derivative of number density
    """
    n, dn_dx = state

    # --- Physical parameters (Table 1) ---
    v0    = params.get('v0', 4e4)         # Bulk flow speed [cm/s]
    T0    = params.get('T0', 1e4)         # Reference temperature [K]
    n0    = params.get('n0', 0.1)         # Total number density (protons + electrons) [cm^-3]
    alpha = params.get('alpha', 3.0)      # Pressure ratio: (P_cr + P_gas)/P_gas
    B     = params.get('B', 3e-6)         # Magnetic field strength [G]
    Gamma = 1e-26                         # Background heating [erg/s]

    n_p = n0 / 2                          # Proton density [cm^-3]
    P_c0 = (alpha - 1) * n0 * k_B * T0    # Cosmic ray pressure [erg/cm^3]

    # --- Temperature as a function of density, T(n) [K] ---
    T = (
        (n_p * v0**2 * mH / k_B + alpha * n0 * T0) * n**(-1)
        - P_c0 * n0**(-2/3) / k_B * n**(-1/3)
        - 2 * n_p**2 * v0**2 * mH / k_B * n**(-2)
    )

    # --- First and second derivatives of T with respect to n ---
    dT_dn = (
        - (n_p * v0**2 * mH / k_B + alpha * n0 * T0) * n**(-2)
        + (1/3) * P_c0 * n0**(-2/3) / k_B * n**(-4/3)
        + 4 * n_p**2 * v0**2 * mH / k_B * n**(-3)
    )

    d2T_dn2 = (
        2 * (n_p * v0**2 * mH / k_B + alpha * n0 * T0) * n**(-3)
        - (4/9) * P_c0 * n0**(-2/3) / k_B * n**(-7/3)
        - 12 * n_p**2 * v0**2 * mH / k_B * n**(-4)
    )

    # --- Conductive coefficient κ(T) and denominator of Eq. 7 ---
    kappa = 5.6e-7 * T**(5/2)             # Thermal conductivity [erg s⁻¹ cm⁻¹ K⁻¹]
    denom = -kappa * dT_dn               # Denominator term for d²n/dx² equation

    # --- Coefficients in Eq. 7 ---
    O_n = (kappa * d2T_dn2 + (5/2) * 5.6e-7 * T**(3/2) * dT_dn**2) / denom

    P_n = (
        - (5/2) * k_B * v0 * n0 * dT_dn
        + 0.5 * (v0 * n0)**3 * mH * n**(-3)
        - B / np.sqrt(4 * np.pi * (n / 2) * mH) * P_c0 * n**(-1/3) * n0**(-2/3) * (2/3)
    ) / denom

    # --- Cooling function and CR heating terms ---
    Lambda = cooling_function(T)
    R_n = ((n / 2) * Gamma - (n / 2)**2 * Lambda) / denom

    # Return dn/dx and d²n/dx²
    return [dn_dx, O_n * dn_dx**2 + P_n * dn_dx + R_n]



def run_solve_ivp(n_initial=1e-3, dn_dx_initial=-1e-14, x_span=(0, 3e20),
                  atol=1e-6, rtol=0, method='RK45', **params):
    """
    Solve the initial value problem for cosmic-ray mediated thermal fronts.

    Parameters
    ----------
    n_initial : float
        Initial number density n(x=0) [cm^-3]
    dn_dx_initial : float
        Initial density gradient dn/dx(x=0) [cm^-4]
    x_span : tuple of float
        Integration domain [cm]
    atol : float
        Absolute error tolerance
    rtol : float
        Relative error tolerance
    method : str
        Integration method for solve_ivp
    params : dict
        Physical parameters to pass into system_of_equations()

    Returns
    -------
    n_values : ndarray
        Density profile n(x)
    dn_dx_values : ndarray
        Density gradient profile dn/dx(x)
    x_values : ndarray
        Spatial coordinate x
    d2n_dx2_values : ndarray
        Second derivative profile d²n/dx²(x)
    """
    sol = solve_ivp(
        fun=lambda x, y: system_of_equations(x, y, **params),
        t_span=x_span,
        y0=[n_initial, dn_dx_initial],
        method=method,
        atol=atol,
        rtol=rtol
    )

    # Unpack solution
    x_values = sol.t
    y_values = sol.y[0]  # n(x)
    z_values = sol.y[1]  # dn/dx(x)

    # Compute d²n/dx² analytically using the same ODE definition
    d2n_dx2_values = np.array([
        system_of_equations(xi, [ni, zi], **params)[1]
        for xi, ni, zi in zip(x_values, y_values, z_values)
    ])

    return y_values, z_values, x_values, d2n_dx2_values

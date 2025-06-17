# scripts/plot_profiles.py

import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
from src.solver import run_solve_ivp
from src.physics import k_B, mH


def plot_style():
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic('config', 'InlineBackend.figure_format = "retina"')

    plt.rcParams.update({
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.labelsize': 'xx-large',
        'xtick.labelsize': 'xx-large',
        'ytick.labelsize': 'xx-large',
        'legend.fontsize': 13,
        'axes.titlesize': 'xx-large',
        'axes.linewidth': 1.2,
        'lines.linewidth': 2,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.top': True,
        'ytick.right': True,
        'figure.figsize': (6.4, 9.6),    # reduced height
        'figure.dpi': 100,               # shrink display size
        'savefig.dpi': 300,              # keep high-res for PDF
    })

plot_style()

def plot_all_profiles(z_initial_list=None, T0=1e4, B=3e-6, alpha=3, v0=0, n0=0.1):
    """
    Generate plots of n(x), T(x), and pressure components across CR-mediated thermal fronts
    for multiple initial density gradients.

    Parameters
    ----------
    z_initial_list : list of float
        List of initial dn/dx values [cm^-4]
    T0 : float
        Upstream temperature [K]
    B : float
        Magnetic field [G]
    alpha : float
        Pressure ratio: P_total / P_gas
    v0 : float
        Upstream outflow velocity [cm/s]
    n0 : float
        Upstream number density [cm^-3]

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object for further customization/saving
    axs : ndarray of Axes
        Array of axes corresponding to subplots
    """
    if z_initial_list is None:
        z_initial_list = [-1e-14, -1e-15, -1e-16, -1e-17]

    n_p = n0 / 2                              # Proton density [cm^-3]
    P_c0 = (alpha - 1) * n0 * k_B * T0        # Initial CR pressure [erg/cm^3]

    # Color palette for different initial conditions
    palette = ['#19489C', '#548ACA', '#71C193', '#E72519']

    fig, axs = plt.subplots(3, 1, figsize=(6.4, 14.4), sharex=True)
    plt.subplots_adjust(hspace=0)

    for i, z_init in enumerate(z_initial_list):
        label = r'$\rm \frac{{dn}}{{dx}}\vert_0=10^{{{}}} \, \mathrm{{cm}}^{{-4}}$'.format(int(np.log10(abs(z_init))))

        # Solve the ODE
        n_vals, dn_dx_vals, x_vals,_ = run_solve_ivp(
            n_initial=n0, dn_dx_initial=z_init,
            T0=T0, v0=v0, alpha=alpha, n0=n0, B=B,
            rtol=1e-6, atol=0, x_span=(0, 3e20)
        )

        # Recompute physical quantities from n(x)
        T_vals = (
            (n_p * v0**2 * mH / k_B + alpha * n0 * T0) * n_vals**(-1)
            - P_c0 * n0**(-2/3) / k_B * n_vals**(-1/3)
            - 2 * n_p**2 * v0**2 * mH / k_B * n_vals**(-2)
        )
        P_c_vals = P_c0 * (n_vals / n0)**(2 / 3)             # CR pressure
        P_g_vals = n_vals * k_B * T_vals                     # Gas pressure
        v_vals   = n0 * v0 / n_vals                          # Mass flux conservation
        P_dyn    = n_vals * mH * v_vals**2                   # Dynamical pressure

        x_pc = x_vals / 3e18  # Convert from [cm] to [pc]

        # Plot density
        axs[0].loglog(x_pc, n_vals, label=label, color=palette[i])
        # Plot temperature
        axs[1].loglog(x_pc, T_vals, label=label, color=palette[i])
        # Plot pressure components
        axs[2].loglog(x_pc, P_c_vals / k_B,      label='CR' if i == 0 else "", linestyle='-',  color=palette[i])
        axs[2].loglog(x_pc, P_g_vals / k_B,      label='Gas' if i == 0 else "", linestyle='--', color=palette[i])
        axs[2].loglog(x_pc, P_dyn    / k_B,      label='Dynamical' if i == 0 else "", linestyle=':',  color=palette[i])

    # --- Axis labeling ---
    axs[0].set_ylabel("Number Density [cm$^{-3}$]")
    axs[1].set_ylabel("Temperature [K]")
    axs[2].set_ylabel("Pressure / $k_B$ [cm$^{-3}$ K]")
    axs[2].set_xlabel("x [pc]")

    # --- Legends and tick marks ---
    axs[0].legend(fontsize=12)
    axs[2].legend(fontsize=12, loc="lower left")
    axs[0].set_xlim(8e-7, 2e2)
    axs[0].set_xticks([1e-6, 1e-4, 1e-2, 1, 100])

    plt.tight_layout()
    return fig, axs

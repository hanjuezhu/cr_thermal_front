# scripts/plot_line_ratios.py

"""
Generate a diagnostic plot of line ratio predictions from CR-mediated thermal fronts,
compared with Wakker et al. absorption line observations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate

from src.solver import run_solve_ivp
from src.physics import k_B, mH

# Load ion fraction table
ion_table = np.loadtxt('./data/tab2.txt', skiprows=125)
Temp_tab = ion_table[:, 0]
SiIV_tab = ion_table[:, 58]
CIV_tab  = ion_table[:, 10]
OVI_tab  = ion_table[:, 27]
NV_tab   = ion_table[:, 18]

# Solar abundances (Asplund et al. 2009)
Si_abund = 10**(7.51 - 12)
C_abund  = 10**(8.43 - 12)
O_abund  = 10**(8.69 - 12)
N_abund  = 10**(7.83 - 12)

# Load observational data (Wakker et al.)
wakker = pd.read_csv('./data/wakker.txt', skiprows=85, delim_whitespace=True, header=None, on_bad_lines='warn')
SiIV_CIV = wakker[wakker[22] == 'SiIV/CIV']
CIV_OVI  = wakker[wakker[22] == 'CIV/OVI']
NV_OVI   = wakker[wakker[22] == 'NV/OVI']

# Merge SiIV/CIV and CIV/OVI
obs = pd.merge(SiIV_CIV, CIV_OVI, on=[0, 1, 2], suffixes=('_SiIV_CIV', '_CIV_OVI'))
obs = obs[obs['24_SiIV_CIV'] != 0]  # Remove invalid entries

# Merge CIV/OVI and NV/OVI
obs_2 = pd.merge(CIV_OVI, NV_OVI, on=[0, 1, 2], suffixes=('_CIV_OVI', '_NV_OVI'))
obs_2 = obs_2[obs_2['24_NV_OVI'] != 0]

# --- Model parameters ---
colors = ['#19489C', '#548ACA', '#71C193', '#E72519']
z_initial_list = [-1e-14, -1e-15, -1e-16, -1e-17]
T0, v0, n0 = 1e4, 4e4, 0.1  # [K], [cm/s], [cm^-3]
np0 = n0 / 2                # Proton density assuming full ionization

def compute_column_densities(n_vals, T_vals, x_vals):
    """Interpolate ion fractions and compute integrated column densities."""
    OVI  = np.interp(T_vals, Temp_tab, OVI_tab)
    CIV  = np.interp(T_vals, Temp_tab, CIV_tab)
    SiIV = np.interp(T_vals, Temp_tab, SiIV_tab)
    NV   = np.interp(T_vals, Temp_tab, NV_tab)

    N_OVI  = integrate.simpson(O_abund * OVI  * n_vals, x_vals)
    N_CIV  = integrate.simpson(C_abund * CIV  * n_vals, x_vals)
    N_SiIV = integrate.simpson(Si_abund * SiIV * n_vals, x_vals)
    N_NV   = integrate.simpson(N_abund * NV   * n_vals, x_vals)

    return N_CIV, N_OVI, N_SiIV, N_NV

def run_model(ax1, ax2, B, alpha, marker):
    """
    Run front solver and compute multiple line ratios for a set of initial gradients.
    Plot on both SiIV/CIV and NV/OVI panels.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        Top panel: SiIV/CIV vs CIV/OVI
    ax2 : matplotlib.axes.Axes
        Bottom panel: NV/OVI vs CIV/OVI
    B : float
        Magnetic field [G]
    alpha : float
        Pressure ratio: P_total / P_gas
    marker : str
        Matplotlib marker style
    """
    P_c0 = (alpha - 1) * n0 * k_B * T0

    for i, z0 in enumerate(z_initial_list):
        color = colors[i]
        label = rf"$\left.\frac{{dn}}{{dx}}\right|_{{x=0}} = 10^{{{int(np.log10(abs(z0)))}}}$"

        n_vals, dn_dx_vals, x_vals, _ = run_solve_ivp(
            n_initial=n0, dn_dx_initial=z0,
            T0=T0, v0=v0, alpha=alpha, n0=n0, B=B,
            rtol=1e-6, atol=0, x_span=(0, 3e20)
        )

        T_vals = (
            (np0 * v0**2 * mH / k_B + alpha * n0 * T0) * n_vals**(-1)
            - P_c0 * n0**(-2/3) / k_B * n_vals**(-1/3)
            - 2 * np0**2 * v0**2 * mH / k_B * n_vals**(-2)
        )

        OVI = np.interp(T_vals, Temp_tab, OVI_tab)
        CIV = np.interp(T_vals, Temp_tab, CIV_tab)
        SiIV = np.interp(T_vals, Temp_tab, SiIV_tab)
        NV  = np.interp(T_vals, Temp_tab, NV_tab)

        N_OVI  = integrate.simpson(O_abund * OVI  * n_vals, x_vals)
        N_CIV  = integrate.simpson(C_abund * CIV  * n_vals, x_vals)
        N_SiIV = integrate.simpson(Si_abund * SiIV * n_vals, x_vals)
        N_NV   = integrate.simpson(N_abund * NV   * n_vals, x_vals)

        ax1.plot(np.log10(N_CIV / N_OVI), np.log10(N_SiIV / N_CIV),
                 marker, ms=10, label=label, color=color, zorder=20,
                 markeredgecolor='white', markeredgewidth=0.3)

        ax2.plot(np.log10(N_CIV / N_OVI), np.log10(N_NV / N_OVI),
                 marker, ms=10, label=label, color=color, zorder=20,
                 markeredgecolor='white', markeredgewidth=0.3)


def plot_line_ratios():
    """
    Plot line ratio diagrams in two stacked panels:
    - Top: log[N(SiIV)/N(CIV)] vs. log[N(CIV)/N(OVI)]
    - Bottom: log[N(NV)/N(OVI)] vs. log[N(CIV)/N(OVI)]

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : np.ndarray of Axes
    """
    fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=True)
    ax1, ax2 = axs

    for B, alpha, marker in [
        (3e-6, 1,  'P'),
        (3e-6, 3,  's'),
        (3e-6, 10, 'o'),
        (2e-5, 3,  'X'),
        (3e-5, 3,  '^'),
    ]:
        run_model(ax1, ax2, B=B, alpha=alpha, marker=marker)

    # Top panel: SiIV/CIV vs CIV/OVI
    ax1.errorbar(obs['24_CIV_OVI'], obs['24_SiIV_CIV'],
                 xerr=obs['25_CIV_OVI'], yerr=obs['25_SiIV_CIV'],
                 fmt='o', capsize=0, ms=8, alpha=1, elinewidth=2,
                 markeredgecolor='white', markeredgewidth=0.2,
                 c='#40494b', ecolor='#D9D4D0')
    ax1.set_ylabel(r'$\log\,\mathrm{[N(SiIV)/N(CIV)]}$', fontsize=16)

    # Bottom panel: NV/OVI vs CIV/OVI
    ax2.errorbar(obs_2['24_CIV_OVI'], obs_2['24_NV_OVI'],
                 xerr=obs_2['25_CIV_OVI'], yerr=obs_2['25_NV_OVI'],
                 fmt='o', capsize=0, ms=8, alpha=1, elinewidth=2,
                 markeredgecolor='white', markeredgewidth=0.2,
                 c='#40494b', ecolor='#D9D4D0')
    ax2.set_xlabel(r'$\log\,\mathrm{[N(CIV)/N(OVI)]}$', fontsize=16)
    ax2.set_ylabel(r'$\log\,\mathrm{[N(NV)/N(OVI)]}$', fontsize=16)

    # Shared legend
    legend_elements = [
        plt.Line2D([0], [0], marker='P', color='w', label=r'$\alpha = 1$ (No CR)', markerfacecolor='k', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label=r'$B = 3\,\mu$G, $\alpha = 3$', markerfacecolor='k', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=r'$B = 3\,\mu$G, $\alpha = 10$', markerfacecolor='k', markersize=10),
        plt.Line2D([0], [0], marker='X', color='w', label=r'$B = 20\,\mu$G, $\alpha = 3$', markerfacecolor='k', markersize=10),
        plt.Line2D([0], [0], marker='^', color='w', label=r'$B = 30\,\mu$G, $\alpha = 3$', markerfacecolor='k', markersize=10)
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=12)

    for ax in axs:
        ax.grid(False)

    plt.tight_layout()
    return fig, axs

# scripts/plot_components.py


import numpy as np
import matplotlib.pyplot as plt
from src.solver import run_solve_ivp
from src.physics import k_B, mH, cooling_function

def plot_components(z_initial_list=None, T0=1e4, B=3e-6, alpha=3, v0=4e4, n0=0.1, Gamma=1e-26):
    """
    Plot individual energy balance components across cosmic-ray-mediated thermal fronts.

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
        Upstream velocity [cm/s]
    n0 : float
        Upstream number density [cm^-3]
    Gamma : float
        Uniform heating rate [erg cm^-3 s^-1]

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : ndarray of Axes
    """
    if z_initial_list is None:
        z_initial_list = [-1e-14, -1e-15, -1e-16, -1e-17]

    np0 = n0 / 2
    P_c0 = (alpha - 1) * n0 * k_B * T0
    palette = ['#19489C', '#548ACA', '#71C193', '#E72519']

    fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(21, 12))
    num_cols = 2

    for i, z0 in enumerate(z_initial_list):
        n, dn_dx, x, d2n_dx2 = run_solve_ivp(
            n_initial=n0, dn_dx_initial=z0, x_span=(0, 3e20),
            atol=0, rtol=1e-6, v0=v0, T0=T0, n0=n0, alpha=alpha, B=B
        )

        # Temperature and derivatives
        T = ((np0 * v0**2 * mH / k_B + alpha * n0 * T0) * n**(-1)
             - P_c0 * n0**(-2/3) / k_B * n**(-1/3)
             - 2 * np0**2 * v0**2 * mH / k_B * n**(-2))

        dT_dn = (-(np0 * v0**2 * mH / k_B + alpha * n0 * T0) * n**(-2)
                 + (1/3) * P_c0 * n0**(-2/3) / k_B * n**(-4/3)
                 + 4 * np0**2 * v0**2 * mH / k_B * n**(-3))

        dT_dx = dT_dn * dn_dx

        d2T_dx2 = dT_dn * d2n_dx2 + (
            2 * (np0 * v0**2 * mH / k_B + alpha * n0 * T0) * n**(-3)
            - (4/9) * P_c0 * n0**(-2/3) / k_B * n**(-7/3)
            - 12 * np0**2 * v0**2 * mH / k_B * n**(-4)
        ) * dn_dx**2

        # Energy balance components
        conductive_flux_div = 5.6e-7 * (T**(5/2) * d2T_dx2 + 5/2 * T**(3/2) * dT_dx**2)
        enthalpy_flux_div = 5/2 * k_B * v0 * n0 * dT_dx
        v_A = B / np.sqrt(4 * np.pi * (n / 2) * mH)
        dP_c_dx = P_c0 * n**(-1/3) * n0**(-2/3) * (2/3) * dn_dx
        cr_heating = -v_A * dP_c_dx
        Lambda = cooling_function(T)
        net_cooling = (n / 2) * Gamma - (n / 2)**2 * Lambda
        kinetic_flux_div = -mH * (v0 * n0)**3 * n**(-3) * dn_dx

        # Select subplot
        row, col = divmod(i, num_cols)
        ax = axs[row, col]
        x_pc = x / 3e18
        label = r'$\rm \frac{{dn}}{{dx}}\vert_0=10^{{{}}} \, \mathrm{{cm}}^{{-4}}$'.format(int(np.log10(abs(z0))))

        # Plot each component
        ax.loglog(x_pc, enthalpy_flux_div, label=r'$\frac{\partial}{\partial x}(\frac{\gamma_g}{\gamma_{g}-1} k_B v nT)$', color='red', lw=3)
        ax.loglog(x_pc, conductive_flux_div, label=r'$\frac{\partial}{\partial x} (\kappa_T \frac{dT}{dx})$', color='blue', lw=3)
        ax.loglog(x_pc, -conductive_flux_div, linestyle='--', label=r'$-\frac{\partial}{\partial x} (\kappa_T \frac{dT}{dx})$',color='blue', lw=3)
        ax.loglog(x_pc, cr_heating, label=r'$-v_A \frac{dP_c}{dx}$', color='orange', lw=5)
        ax.loglog(x_pc, -net_cooling, label=r'$n^2\Lambda - n\Gamma$', color='green', lw=3)
        ax.loglog(x_pc, kinetic_flux_div, label=r'$\frac{\partial}{\partial x}(\frac{1}{2}\rho v^2 v)$', color='black', lw=3)

        
        ax.text(5e-2, 1e-21, label, fontsize=25, weight='bold')

        ax.set_xlim(8e-7, 2e2)
        ax.set_ylim(1e-31, 5e-20)
        ax.set_xticks([1e-6, 1e-4, 1e-2, 1, 1e2])
        ax.set_yticks([1e-28, 1e-24, 1e-20])
        ax.tick_params(axis='both', which='major', labelsize=30, pad=10)
        ax.xaxis.get_offset_text().set_size(30)

    legend = axs[0, 1].legend(bbox_to_anchor=(1, 1), fontsize=30)
    for handle in legend.legendHandles:
        handle.set_markersize(3)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('x (pc)', fontsize=40, labelpad=30)
    plt.ylabel(r'Flux Gradient (erg cm$^{-3}$ s$^{-1}$)', fontsize=40, labelpad=70)

    return fig, axs
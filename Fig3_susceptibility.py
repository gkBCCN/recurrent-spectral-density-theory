import matplotlib.pyplot as plt
import joblib
import theory
from param_file import *


plt.figure(figsize=(5, 3.5))

# -------------------------------------------------------------------------------------- #
#                                   Sim Results                                          #
# -------------------------------------------------------------------------------------- #
ks_results = joblib.load(dp + f'SXs_ave_over_1000_trials')
X_sim = ks_results['Sas'] / ks_results['Ss']
X_sim_mag = np.sqrt(X_sim.real ** 2 + X_sim.imag ** 2)
plt.plot(ks_results['freqs'][1:], X_sim_mag[1:], lw=2, c=sim_col, label='sim', zorder=1, alpha=.8)

r0 = joblib.load(dp + f'ISI_ave_over_1000_trials')['r0']


# -------------------------------------------------------------------------------------- #
#                                 Theory Results                                         #
# -------------------------------------------------------------------------------------- #
theory_freq = stat_params['theory_freq']
X_curr, X_noise, X_rm = theory.load_susceptibilities(sim_params, stat_params, dp)
plt.plot(theory_freq, X_rm['mag'], c=rm_col, lw=3, ls='-', label=r'$\chi_{RM}$-with $\chi_D$', alpha=1, zorder=5)

if J==0.01:
    pd = theory.delay_characteristic_func(sim_params, stat_params['theory_freq'])
    R_curr, R_noise = theory.input_mean_and_intensity(sim_params)
    X_no_D = X_curr['complex'] / (1 - X_curr['complex'] * R_curr * pd)
    X_no_D_mag = np.sqrt(X_no_D.real**2 + X_no_D.imag**2)  # this is more accurate than np.abs()
    plt.plot(theory_freq, X_no_D_mag, c='C9', lw=3, ls='-', label=r'$\chi_{RM}$-no $\chi_D$', alpha=1, zorder=5)


# -------------------------------------------------------------------------------------- #
#                                   Formatting                                           #
# -------------------------------------------------------------------------------------- #
plt.axvline(r0, c='k', ls=':', lw=1, zorder=0)

plt.xscale('log')
plt.xlabel('f')
xtick_arr = [0.01, 0.1, 1, r0, 10, fc]
xtick_str = ['$10^{-2}$', '$10^{-1}$', '$10^{0}$', r'$r_0$', '$10^{1}$', '']
plt.xticks(xtick_arr, xtick_str)
plt.xlim(0.05, fc)

plt.ylabel(r'$|\chi|$', rotation=0, labelpad=20)
plt.ylim(0, 3)
plt.yticks([1, 3])

plt.legend(fontsize=16, frameon=0)
plt.gca().spines['right'].set_visible(0)
plt.gca().spines['top'].set_visible(0)
plt.subplots_adjust(top=0.95, right=0.97, left=0.16, bottom=0.2)
plt.show()

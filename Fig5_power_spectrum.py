import matplotlib.pyplot as plt
import joblib
import theory
from param_file import *


plt.figure(figsize=(5, 4.7))

# -------------------------------------------------------------------------------------- #
#                                   Sim Results                                          #
# -------------------------------------------------------------------------------------- #
ks_results = joblib.load(dp + f'SXs_ave_over_1000_trials')
SX, freqs = ks_results['Sa'], ks_results['freqs']
assert np.all(SX.imag == 0)
plt.plot(freqs, stat_params['N_obs']**2 * SX.real, lw=2, c=sim_col, alpha=0.8)

ks_results = joblib.load(dp + f'Sas_Sphis_ave_over_1000_trials')
SA = ks_results['Sa']
assert np.all(SA.imag == 0)
plt.plot(freqs, SA.real, lw=2, c='C0', alpha=0.8)

SPhi = ks_results['Sy']
assert np.all(SPhi.imag == 0)
plt.plot(freqs, SPhi.real, lw=2, c='C4', alpha=0.8)

r0 = joblib.load(dp + f'ISI_ave_over_1000_trials')['r0']


# -------------------------------------------------------------------------------------- #
#                                 Theory Results                                         #
# -------------------------------------------------------------------------------------- #
theory_freq = stat_params['theory_freq']

S_X = theory.allspike_power_spectrum(sim_params, stat_params, dp)
plt.plot(theory_freq, S_X, label=r'$S_{X}$', c=pop_th_col, lw=th_lw, alpha=0.8)

S_A = theory.activity_power_spectrum(sim_params, stat_params, dp)
plt.plot(theory_freq, S_A, label=r'$S_{A}$', c='C9', lw=th_lw)

S_Phi_results = theory.load_sync_power_spectrum(sim_params, stat_params, dp)

plt.plot(S_Phi_results['freqs'], S_Phi_results['S_Phi'], label=r'$S_\Phi$', c=rm_col, lw=th_lw)


# -------------------------------------------------------------------------------------- #
#                                   Formatting                                           #
# -------------------------------------------------------------------------------------- #
plt.axvline(r0, c='k', ls=':', lw=1, zorder=0)
plt.axvline(1 / act_delta, c='k', ls=':', lw=1, zorder=0)

plt.xscale('log')
plt.xlabel('f')
xtick_arr = [0.01, 0.1, 1, r0, 1/act_delta]
xtick_str = ['$10^{-2}$', '$10^{-1}$', '$10^{0}$', r'$r_0$', r'$\Delta^{-1}$']
plt.xticks(xtick_arr, xtick_str)
plt.xlim(0.05, 2 / act_delta)

plt.yscale('log')
plt.ylabel('power spectrum')
plt.ylim(1e-5, 2e4)

plt.legend(fontsize=16, frameon=0)
plt.gca().spines['right'].set_visible(0)
plt.gca().spines['top'].set_visible(0)
plt.subplots_adjust(top=0.99, right=0.99, left=0.24, bottom=0.15)
plt.show()

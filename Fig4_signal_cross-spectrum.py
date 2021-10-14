import matplotlib.pyplot as plt
import joblib
import theory
from param_file import *


plt.figure(figsize=(5, 4.7))

# -------------------------------------------------------------------------------------- #
#                                   Sim Results                                          #
# -------------------------------------------------------------------------------------- #
ks_results = joblib.load(dp + f'SXs_ave_over_1000_trials')
SXs, freqs = ks_results['Sas'], ks_results['freqs']
SXs_mag = stat_params['N_obs'] * np.sqrt(SXs.real**2 + SXs.imag**2)
plt.plot(freqs, SXs_mag, lw=2, c=sim_col, alpha=0.8)

ks_results = joblib.load(dp + f'Sas_Sphis_ave_over_1000_trials')
SAs = ks_results['Sas']
SAs_mag = np.sqrt(SAs.real**2 + SAs.imag**2)
plt.plot(freqs, SAs_mag, lw=2, c='C0', alpha=0.8)

SPhis = ks_results['Sys']
SPhis_mag = np.sqrt(SPhis.real**2 + SPhis.imag**2)
plt.plot(freqs, SPhis_mag, lw=2, c='C4', alpha=0.8)

r0 = joblib.load(dp + f'ISI_ave_over_1000_trials')['r0']


# -------------------------------------------------------------------------------------- #
#                                 Theory Results                                         #
# -------------------------------------------------------------------------------------- #
theory_freq = stat_params['theory_freq']

results = theory.load_signal_cross_spectra(sim_params, stat_params, dp)
plt.plot(theory_freq, np.abs(results['SXs']), label=r'$S_{Xs}$', c=pop_th_col, lw=th_lw, alpha=1)
plt.plot(theory_freq, np.abs(results['SAs']), label=r'$S_{As}$', c='C9', lw=th_lw)
plt.plot(theory_freq, np.abs(results['Sphis']), label=r'$S_{\Phi s}$', c=rm_col, lw=th_lw, ls='-', alpha=0.8)


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
plt.ylabel('cross-spectrum', rotation=90)
plt.ylim(3e-5, 3)

plt.legend(fontsize=16, frameon=0)
plt.gca().spines['right'].set_visible(0)
plt.gca().spines['top'].set_visible(0)
plt.subplots_adjust(top=0.99, right=0.99, left=0.24, bottom=0.15)
plt.show()

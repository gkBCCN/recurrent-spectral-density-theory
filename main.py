from mpmath import mp
mp.dps = 20
from param_file import sim_params, stat_params, dp
import theory

# ------------------------------------------------------------------------------------------------------------------- #
SUSC   = 0

Sxs    = 0
SXs    = 0
SAs    = 0
Sphis  = 0

Sij    = 0
Sii    = 0
Cyy    = 0

S0     = 0
Sx     = 0
SX     = 0
SA     = 0
Sphi   = 0

COH_X   = 0
COH_Phi = 0

# ------------------------------------------------------------------------------------------------------------------- #
if SUSC:
    print('\nRecurrence-modulated susceptibility.')
    theory.calc_exact_diff_approx_susceptibility(sim_params, stat_params, dp)
# --------------------------------------------------------------------------------------------------------------- #
if Sxs:
    from IFy.scripts.theory.cross_spectrum import *
    calc_linear_approx_of_signal_cross_spectrum_from_exact_stats(sim_params, stat_params, dp)
# --------------------------------------------------------------------------------------------------------------- #
if Sij:
    from IFy.scripts.theory.cross_spectrum import *
    print('Sij')
    calc_network_cross_spectra(sim_params, stat_params, dp)
if Sii:
    from IFy.scripts.theory.cross_spectrum import *
    print('Sii')
    calc_network_auto_spectra(sim_params, stat_params, dp)
if Cyy:
    print(f"Cyy: thresh={stat_params['act_thresh']}")
    from IFy.lib.theory.gauss_activity_synchrony_approximations import calc_sync_autocorrelation
    calc_sync_autocorrelation(sim_params, stat_params, dp)
# --------------------------------------------------------------------------------------------------------------- #
if S0:
    from IFy.scripts.theory.power_spectrum import *
    print('\nUnperturbed power spectrum.')
    calc_exact_white_noise_power_spectrum(sim_params, stat_params, dp)
# --------------------------------------------------------------------------------------------------------------- #
if COH_X:
    from IFy.scripts.theory.cross_spectrum import *
    print('\nExact COH')
    calc_exact_subpop_coherence(sim_params, stat_params, dp)


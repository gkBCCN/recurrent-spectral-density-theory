from mpmath import mp
mp.dps = 20
from param_file import sim_params, stat_params, dp
import theory

# ------------------------------------------------------------------------------------------------------------------- #
SUSC   = 0
S0     = 0

Sij    = 0
Sii    = 0

SYs    = 0

Cyy    = 0

SX     = 0
SA     = 0
Sphi   = 0

COH_X   = 0
COH_Phi = 0

# ------------------------------------------------------------------------------------------------------------------- #
if SUSC:
    print('\nSingle-neuron current- and noise-modulation susceptibilities.')
    theory.calc_single_neuron_susceptibilities(sim_params, stat_params, dp)
    print('\nRecurrence-modulated susceptibility.')
    theory.calc_recurrence_modulated_susceptibility(sim_params, stat_params, dp)
# --------------------------------------------------------------------------------------------------------------- #
if S0:
    print('\nUnperturbed power spectrum.')
    theory.calc_unperturbed_power_spectrum(sim_params, stat_params, dp)
# --------------------------------------------------------------------------------------------------------------- #
if Sij:
    print('Sij')
    theory.calc_network_cross_spectra(sim_params, stat_params, dp)
if Sii:
    print('Sii')
    theory.calc_network_auto_spectra(sim_params, stat_params, dp)
# --------------------------------------------------------------------------------------------------------------- #
if SYs:
    theory.calc_signal_cross_spectra(sim_params, stat_params, dp)
# --------------------------------------------------------------------------------------------------------------- #
if Cyy:
    print(f"Cyy: thresh={stat_params['act_thresh']}")
    calc_sync_autocorrelation(sim_params, stat_params, dp)
if COH_X:
    print('\nExact COH')
    calc_exact_subpop_coherence(sim_params, stat_params, dp)


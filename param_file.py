import numpy as np
import os
from matplotlib import rc

# ------------------------------------------------------------------------ #
#                       Model Parameters                                   #
# ------------------------------------------------------------------------ #
sim_params = dict()
sim_params['T'] = T = 1000                  # trial length (recording time)
sim_params['dt'] = dt = 1e-3                # time step
sim_params['burn_in'] = burn_in = 200       # period before measurement to negate initial cond. artifacts

net_params = dict()
net_params['N_e'] = N_e = int(10e3)         # excitatory population size
net_params['gamma'] = gamma = .25           # ratio of inhib:excit
net_params['N_i'] = N_i = int(N_e * gamma)  # inhibitory population size
sim_params['net'] = net_params

neuron_params = dict()
neuron_params['vt'] = vt = +1.              # threshold
neuron_params['vr'] = vr = 0.               # reset
neuron_params['vl'] = vl = 0.               # leak
neuron_params['taum'] = taum = 1.           # membrane time constant
neuron_params['tref'] = tref = 0.1          # refractory period
sim_params['neuron'] = neuron_params

signal_params = dict()
signal_params['f_c'] = fc = 15.             # signal cutoff frequency
signal_params['sig_s'] = sig_s = 0.3        # external signal standard deviation
signal_params['D_s'] = Ds = 1 / (4 * fc)    # external noise intensity -> gives var=1
signal_params['mu'] = mu = 1.1 * (vt - vr)  # bias
signal_params['D'] = D = 1.0e-2             # intrinsic noise intensity
sim_params['signal'] = signal_params

synapse_params = dict()
synapse_params['J'] = J = 1.0e-2            # (excitatory) synaptic strength
synapse_params['g'] = g = 5.                # relative inhibitory strength (4=balanced if gamma=0.25)
synapse_params['Ci'] = Ci = 200             # fixed number of inhibitory inputs
synapse_params['Ce'] = Ce = 800             # fixed number of excitatory inputs
synapse_params['p_conn'] = pc = Ce / N_e    # connection probability (sparseness)
synapse_params['delay'] = [0.05, 0.2]       # synaptic delay
sim_params['synapse'] = synapse_params


# ------------------------------------------------------------------------ #
#                       Measurement Parameters                             #
# ------------------------------------------------------------------------ #
stat_params = dict()
stat_params['N_obs'] = N_obs = 250          # size of subpopulation (number of observed neurons)
stat_params['act_delta'] = act_delta = 0.5  # Note: Use 'None' to set this dynamically
stat_params['R0'] = None                    # Average activity rate (only used if act_delta=None)
stat_params['act_thresh'] = 0.2             # synchrony threshold applied to activity
stat_params['Caa_df'] = 1/100               # subsampling when calculating the activity autocorrelation
stat_params['theory_freq'] = np.around(np.logspace(-3, 3, 1000), 3)  # frequencies at which to calculate theory

# ------------------------------------------------------------------------ #
#                              Save Directory                              #
# ------------------------------------------------------------------------ #
dp = f'data/J={J}/'
if not os.path.exists(dp):
    os.makedirs(dp)


# ------------------------------------------------------------------------ #
#                           Plot Variables                                 #
# ------------------------------------------------------------------------ #
th_lw = 3
rm_col = '#DF1EA8E3'
sim_col = 'k'
pop_th_col = '#00CBAD'
rc('font', **{'size': 22, })

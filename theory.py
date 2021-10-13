import numpy as np
from mpmath import sqrt, exp, quad, power, pi, erfc, j, fabs, inf, mpc as complex_number, pcfd as D_parab, sinc


def setup_erf_fr(tref, taum, vt, vr):
    """
      Stationary or Mean Firing Rate of single LIF neuron
      with Gaussian white noise
    """
    def stationary_fr_of_lif_neuron_receiving_white_noise_erf(mu, D):
        """
        Siegert (1951) or Ricciardi (1977)
        Single neuron transfer function (f-mu curve).
        """
        sig = sqrt(2 * D)
        upper_bound = (mu - vr) / sig
        lower_bound = (mu - vt) / sig

        def integr(z):
            return exp(power(z, 2.)) * erfc(z)

        _integral = quad(integr, [lower_bound, upper_bound], verbose=False)
        return (tref + taum * sqrt(pi) * _integral) ** (-1)

    return stationary_fr_of_lif_neuron_receiving_white_noise_erf
def diffusion_mean_field_approximation(nu, taum, Ce, J, g, gamma):
    """
    Mean-field approximations of synaptic input from networks with rates nu.
    Brunel (2000).

    :param nu: stationary firing rate of population (asynchronous state)
    :param taum: membrane time constant of postsynaptic neuron
    :param Ce: size of excitatory population
    :param J: synaptic weight
    :param g: relative weight of inhibitory synapse
    :param gamma: relative size of inhibitory population
    """
    """ 
        mean from equation (4)
    """
    mu = Ce * J * (1 - gamma * g) * nu * taum

    """
        standard devation from equation (5) of Brunel (2000)
        sigma = sqrt(2D) => D = sigma^2 / 2.  See also Dummer et al. (2014) eq. (14)
    """
    sigma_squared = power(J, 2) * Ce * (1 + gamma * power(g, 2)) * nu * taum
    D = sigma_squared / 2

    return mu, D


def self_consistent_rate(par, start=1., tol=0.1):
    mu = par['signal']['mu']
    Ds = par['signal']['sig_s']**2 * par['signal']['D_s']
    D = par['signal']['D'] + Ds
    # if D == 0:
    #     D = 1e-9  # to avoid division by 0 errors
    taum = par['neuron']['taum']
    tref = par['neuron']['tref']
    vr = par['neuron']['vr']
    vt = par['neuron']['vt']

    gamma = par['net']['gamma']

    Ce = par['synapse']['Ce']
    J = par['synapse']['J']
    g = par['synapse']['g']
    _fr_ = setup_erf_fr(tref=tref, taum=taum, vt=vt, vr=vr)

    v_in_list = [start]
    v_out_list = []


    mean_recurr, recurr_intensity = diffusion_mean_field_approximation(v_in_list[-1], taum, Ce, J, g, gamma)
    D_tot = D + recurr_intensity
    mu_tot = mu + mean_recurr
    v_out_list.append(_fr_(mu_tot, D_tot))

    think_list = ['/', '--', '\\', '|']
    think_iter = 0
    while abs(v_out_list[-1] - v_in_list[-1]) > tol:
        print('\rFinding self-consistent rate and input: {}'.format(think_list[think_iter]), end='', flush=True)

        v_in_list.append(v_out_list[-1])
        mean_recurr, recurr_intensity = diffusion_mean_field_approximation(v_in_list[-1], taum, Ce, J, g, gamma)
        mu_tot = mu + mean_recurr
        D_tot = D + recurr_intensity
        nu_new = _fr_(mu_tot, D_tot)
        v_out_list.append(nu_new)

        if g >= 5:
            v_out_list[-1] = np.array(v_out_list)[-2:].mean()
        think_iter += 1
        if think_iter == 4:
            think_iter = 0

    print('\rFinding self-consistent rate and input: ', end='', flush=True)
    nu = float(v_out_list[-1])
    mu_sc = float(mu_tot)
    D_sc = float(D + recurr_intensity)
    print('ν={:.4f}, mu={:.4f}, D={:.4f}'.format(nu, mu_sc, D_sc), flush=True)
    return nu, mu_sc, D_sc

def get_current_and_noise_susceptibilities(sim_params):
    vt = sim_params['neuron']['vt']
    vr = sim_params['neuron']['vr']
    taum = sim_params['neuron']['taum']
    taur = sim_params['neuron']['tref']
    nu, mu_sc, D_sc = self_consistent_rate_brian(sim_params, tol=1e-4)

    X_mu = setup_susceptibility(r0=nu, mu=mu_sc, D=D_sc, vt=vt, vr=vr, taum=taum, tref=taur)
    X_D = setup_noise_coded_suscept(r0=nu, mu=mu_sc, D=D_sc, vt=vt, vr=vr, taum=taum, tref=taur)
    return X_mu, X_D


def calc_exact_diff_approx_susceptibility(sim_params, stat_params, save_dir):
    X_mu_of, X_D_of = get_current_and_noise_susceptibilities(sim_params)
    susceptibility_helper(X_mu_of, X_D_of, stat_params, save_dir + DA_EXACT_STR)
import numpy as np
import joblib
from mpmath import sqrt, exp, quad, power, pi, erfc, j, fabs, inf, mpc as complex_number, pcfd as D_parab, sinc


# ------------------------------------------------------------------------- #
#                   Stationary Firing Rate                                  #
# ------------------------------------------------------------------------- #
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


def self_consistent_rate(sim_params, start=1., tol=0.1):
    mu = sim_params['signal']['mu']
    Ds_normed = sim_params['signal']['D_s']
    Ds = sim_params['signal']['sig_s']**2 * Ds_normed
    D = sim_params['signal']['D'] + Ds
    # if D == 0:
    #     D = 1e-9  # to avoid division by 0 errors
    vt = sim_params['neuron']['vt']
    vr = sim_params['neuron']['vr']
    taum = sim_params['neuron']['taum']
    tref = sim_params['neuron']['tref']

    gamma = sim_params['net']['gamma']

    Ce = sim_params['synapse']['Ce']
    J = sim_params['synapse']['J']
    g = sim_params['synapse']['g']
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
# ------------------------------------------------------------------------- #


# ------------------------------------------------------------------------- #
#          Firing Rate Response or Susceptibility                           #
# ------------------------------------------------------------------------- #
def setup_susceptibility(r0, mu, D, vt, vr, taum, tref):
    sqr_D = sqrt(D)
    prefactor_base = r0 / sqr_D
    e_min = exp((vr**2 - vt**2 + 2 * mu * (vt - vr)) / (4 * D))
    up_lim = (mu - vt) / sqr_D
    low_lim = (mu - vr) / sqr_D

    def susceptibility_of_white_noise_driven_lif(f):
        iw = complex_number(2 * pi * f * taum * j)
        e_tau = exp(iw * tref)
        prefactor = prefactor_base * iw / (iw - 1)
        numerator = D_parab(iw - 1, up_lim) - e_min * D_parab(iw - 1, low_lim)
        denom = D_parab(iw, up_lim) - e_min * e_tau * D_parab(iw, low_lim)
        _X_ = prefactor * numerator / denom
        X_r, X_i = float(_X_.real), float(_X_.imag)  # convert to numpy
        X_mag = np.sqrt(X_r**2 + X_i**2)  # this is more accurate than np.abs()
        X_ang = np.arctan2(X_i, X_r)
        return X_r + X_i*1j, X_mag, X_ang

    return susceptibility_of_white_noise_driven_lif


def setup_noise_coded_suscept(r0, mu, D, vt, vr, taum, tref):
    sqr_D = sqrt(D)
    prefactor_base = r0 / D
    e_min = exp((vr**2 - vt**2 + 2 * mu * (vt - vr)) / (4 * D))
    up_lim = (mu - vt) / sqr_D
    low_lim = (mu - vr) / sqr_D

    def susceptibility_of_white_noise_driven_lif_to_noise_modulation(f):
        iw = complex_number(2 * pi * f * taum * j)
        e_tau = exp(iw * tref)
        prefactor = prefactor_base * iw * (iw - 1) / (2 - iw)
        numerator = D_parab(iw - 2, up_lim) - e_min * D_parab(iw - 2, low_lim)
        denom = D_parab(iw, up_lim) - e_min * e_tau * D_parab(iw, low_lim)
        _X_ = prefactor * numerator / denom
        X_r, X_i = float(_X_.real), float(_X_.imag)  # convert to numpy
        X_mag = np.sqrt(X_r**2 + X_i**2)  # this is more accurate than np.abs()
        X_ang = np.arctan2(X_i, X_r)
        return X_r + X_i*1j, X_mag, X_ang

    return susceptibility_of_white_noise_driven_lif_to_noise_modulation


def get_current_and_noise_susceptibilities(sim_params):
    nu, mu_sc, D_sc = self_consistent_rate(sim_params, tol=1e-4)
    vt = sim_params['neuron']['vt']
    vr = sim_params['neuron']['vr']
    taum = sim_params['neuron']['taum']
    tref = sim_params['neuron']['tref']
    X_mu = setup_susceptibility(r0=nu, mu=mu_sc, D=D_sc, vt=vt, vr=vr, taum=taum, tref=tref)
    X_D = setup_noise_coded_suscept(r0=nu, mu=mu_sc, D=D_sc, vt=vt, vr=vr, taum=taum, tref=tref)
    return X_mu, X_D


def freq_str(params):
    freq = params['theory_freq']
    return '_log_f=[{}-{},{}]'.format(freq[0], freq[-1], freq.size)


def calc_single_neuron_susceptibilities(sim_params, stat_params, save_dir):
    freq = stat_params['theory_freq']

    X_mu_of, X_D_of = get_current_and_noise_susceptibilities(sim_params)

    # data structures for the current-modulated susceptibility
    X_mu = np.zeros(freq.size, dtype='complex128')
    X_mu_mag = np.zeros(freq.size)
    X_mu_ang = np.zeros(freq.size)
    # data structures for the noise-modulated susceptibility
    X_D = np.zeros(freq.size, dtype='complex128')
    X_D_mag = np.zeros(freq.size)
    X_D_ang = np.zeros(freq.size)

    for f_i, f in enumerate(freq):
        print(f'\r{f_i + 1} / {freq.size}', end='', flush=True)
        X_mu[f_i], X_mu_mag[f_i], X_mu_ang[f_i] = X_mu_of(f)
        X_D[f_i], X_D_mag[f_i], X_D_ang[f_i] = X_D_of(f)

    current_mod_results = {'complex': X_mu, 'mag': X_mu_mag, 'ang': X_mu_ang}
    noise_mod_results = {'complex': X_D, 'mag': X_D_mag, 'ang': X_D_ang}
    results = {'current': current_mod_results, 'noise': noise_mod_results}
    joblib.dump(results, save_dir + 'X_DA_exact' + freq_str(stat_params))


def delay_characteristic_func(sim_params, freq):
    taum = sim_params['neuron']['taum']
    d_min, d_max = sim_params['synapse']['delay']

    d_bar = 0.5 * (d_max + d_min)
    d_del = d_max - d_bar

    pd = np.zeros(freq.size, dtype='complex128')
    for f_i, f in enumerate(freq):
        w = 2 * pi * f * taum
        pd[f_i] = exp(w * j * d_bar) * sinc(w * d_del)
    return pd


def input_mean_and_intensity(sim_params):
    J = sim_params['synapse']['J']
    g = sim_params['synapse']['g']
    Ce = sim_params['synapse']['Ce']
    taum = sim_params['neuron']['taum']
    gamma = sim_params['net']['gamma']
    R_curr = taum * J * Ce * (1 - gamma * g)
    R_noise = 0.5 * taum * J**2 * Ce * (1 + gamma * g**2)
    return R_curr, R_noise


def calc_recurrence_modulated_susceptibility(sim_params, stat_params, save_dir, use_noise_mod=True):
    results = joblib.load(save_dir + 'X_DA_exact' + freq_str(stat_params))
    X_curr, X_noise = results['current'], results['noise']
    pd = delay_characteristic_func(sim_params, stat_params['theory_freq'])
    R_curr, R_noise = input_mean_and_intensity(sim_params)
    X_complex = X_curr['complex'] / (1 - (X_curr['complex'] * R_curr + X_noise['complex'] * R_noise) * pd)
    X_mag = np.sqrt(X_complex.real ** 2 + X_complex.imag ** 2)  # this is more accurate than np.abs()
    X_ang = np.arctan2(X_complex.imag, X_complex.real)
    results = {'complex': X_complex, 'mag': X_mag, 'ang': X_ang}
    joblib.dump(results, save_dir + 'X_rm_' + freq_str(stat_params))


def load_susceptibilities(sim_params, stat_params, save_dir):
    single_results = joblib.load(save_dir + 'X_DA_exact' + freq_str(stat_params))
    X_curr, X_noise = single_results['current'], single_results['noise']
    X_rm = joblib.load(save_dir + 'X_rm_' + freq_str(stat_params))
    return X_curr, X_noise, X_rm
# ------------------------------------------------------------------------- #


# ------------------------------------------------------------------------- #
#                         Power Spectrum                                    #
# ------------------------------------------------------------------------- #
def setup_Sxx(r0, mu, D, vt, vr, taum, tref):
    delta_exponent = (vr**2 - vt**2 + 2 * mu * (vt - vr)) / (4 * D)
    e_d = exp(delta_exponent)
    e_2d = exp(2 * delta_exponent)
    up_lim = (mu - vt) / sqrt(D)
    low_lim = (mu - vr) / sqrt(D)

    def power_spectrum_of_white_noise_driven_lif(f):
        iw = complex_number(2 * pi * f * taum * j)
        e_ref = exp(iw * tref)
        D_t = D_parab(iw, up_lim)
        D_r = D_parab(iw, low_lim)
        return r0 * (fabs(D_t)**2 - e_2d * fabs(D_r)**2) / fabs(D_t - e_d * e_ref * D_r)**2

    return power_spectrum_of_white_noise_driven_lif
# ------------------------------------------------------------------------- #

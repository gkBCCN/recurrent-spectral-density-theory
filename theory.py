import numpy as np
import joblib
from mpmath import sqrt, exp, quad, power, pi, erfc, j, fabs, inf, mpc as complex_number, pcfd as D_parab, sinc


# ------------------------------------------------------------------------- #
#                         Helpers                                           #
# ------------------------------------------------------------------------- #
def freq_str(params):
    freq = params['theory_freq']
    return '_log_f=[{}-{},{}]'.format(freq[0], freq[-1], freq.size)


def get_delta_and_R0(sim_params, stat_params):
    nu, mu_sc, D_sc = self_consistent_rate(sim_params, tol=1e-4)
    delta = stat_params['act_delta']
    R0 = stat_params['R0']
    if not delta:
        delta = R0 / nu
    else:
        R0 = nu * delta
    return delta, R0


# -------Stochastic Mean-field---------- #
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


def delay_characteristic_func(sim_params, freq):
    taum = sim_params['neuron']['taum']
    d_min, d_max = sim_params['synapse']['delay']

    d_bar = 0.5 * (d_max + d_min)
    d_del = d_max - d_bar

    pd = np.zeros(freq.size, dtype='complex128')
    for f_i, f in enumerate(freq):
        w = 2 * pi * f * taum
        pd[f_i] = exp(w * j * d_bar) * sinc(w * d_del)  # mpmath sinc (without pi)
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


# ------------Activity------------------ #
def box_filter(sim_params, stat_params):
    delta, R0 = get_delta_and_R0(sim_params, stat_params)
    """Note: np.sinc(x) = sin(pi x) / (pi x), so I don't need to pass pi in the argument for sinc(delta pi f)."""
    return delta * np.sinc(delta * stat_params['theory_freq'])


# ------------Synchrony----------------- #
def sync_beta(sim_params, stat_params, sig_A):
    phi = stat_params['act_thresh']
    delta, R0 = get_delta_and_R0(sim_params, stat_params)
    N_obs = stat_params['N_obs']
    return (phi - R0 - 1/(2 * N_obs)) / sig_A


def sync_alpha(sim_params, stat_params, sig_A):
    beta = float(sync_beta(sim_params, stat_params, sig_A))
    return np.exp(-0.5 * beta**2) / (np.sqrt(2 * np.pi) * float(sig_A))


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
def setup_current_susceptibility(r0, mu, D, vt, vr, taum, tref):
    """
    Current-modulated susceptibility, X_mu
    :param r0:
    :param mu:
    :param D:
    :param vt:
    :param vr:
    :param taum:
    :param tref:
    :return:
    """
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


def setup_noise_susceptibility(r0, mu, D, vt, vr, taum, tref):
    """
    Noise-modulated susceptibility, X_D
    :param r0:
    :param mu:
    :param D:
    :param vt:
    :param vr:
    :param taum:
    :param tref:
    :return:
    """
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
    X_mu = setup_current_susceptibility(r0=nu, mu=mu_sc, D=D_sc, vt=vt, vr=vr, taum=taum, tref=tref)
    X_D = setup_noise_susceptibility(r0=nu, mu=mu_sc, D=D_sc, vt=vt, vr=vr, taum=taum, tref=tref)
    return X_mu, X_D


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


def calc_recurrence_modulated_susceptibility(sim_params, stat_params, save_dir, use_noise_mod=True):
    """
    Recurrence-modulated susceptibility from eq. (22)
    :param sim_params:
    :param stat_params:
    :param save_dir:
    :param use_noise_mod:
    :return:
    """
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
#                     Network Correlations                                  #
# ------------------------------------------------------------------------- #
def get_ave_input(sim_params, stat_params):
    pd = delay_characteristic_func(sim_params, stat_params['theory_freq'])
    Ce, Ci = sim_params['synapse']['Ce'], sim_params['synapse']['Ci']
    Ne, Ni = sim_params['net']['N_e'], sim_params['net']['N_i']
    Je = sim_params['synapse']['J']
    g = sim_params['synapse']['g']
    Ji = - g * Je
    taum = sim_params['neuron']['taum']
    mu_exc = Ce * Je * taum * pd
    mu_inh = Ci * Ji * taum * pd
    mu_tot = mu_exc + mu_inh
    mu_c = np.abs(mu_exc) ** 2 / Ne + np.abs(mu_inh) ** 2 / Ni

    return mu_exc, mu_inh, mu_tot, mu_c


def calc_network_cross_spectra(sim_params, stat_params, data_path):
    mu_exc, mu_inh, mu_tot, mu_c = get_ave_input(sim_params, stat_params)
    X_curr, X_noise, X_rm = load_susceptibilities(sim_params, stat_params, data_path)
    S_lif = load_exact_diff_approx_power_spectrum(stat_params, data_path)
    Ne, Ni = sim_params['net']['N_e'], sim_params['net']['N_i']

    def Snet(mu_1, N_1, mu_2, N_2):
        return S_lif * (X_rm['complex'] * mu_2 / N_2 + np.conj(X_rm['complex']) * np.conj(mu_1) / N_1 + X_rm['mag']**2 * mu_c)

    results = {'ee': Snet(mu_exc, Ne, mu_exc, Ne), 'ii': Snet(mu_inh, Ni, mu_inh, Ni),
               'ei': Snet(mu_exc, Ne, mu_inh, Ni), 'ie': Snet(mu_inh, Ni, mu_exc, Ne)}
    joblib.dump(results, data_path + 'Sij_net_populations' + freq_str(stat_params))
    return results


def calc_average_network_cross_spectrum(sim_params, stat_params, data_path):
    Snet = calc_network_cross_spectra(sim_params, stat_params, data_path)
    Ne, Ni = sim_params['net']['N_e'], sim_params['net']['N_i']
    N_tot = Ne + Ni
    N_conn = N_tot**2
    p_ee = Ne**2 / N_conn
    p_ii = Ni**2 / N_conn
    p_ei = p_ie = (Ne * Ni) / N_conn
    assert p_ee + p_ii + p_ei + p_ie == 1
    Sij_ave = Snet['ee'] * p_ee + Snet['ii'] * p_ii + Snet['ei'] * p_ei + Snet['ie'] * p_ie
    joblib.dump(Sij_ave, data_path + 'Sij_net_ave' + freq_str(stat_params))
    return Sij_ave


def load_average_network_cross_spectrum(sim_params, stat_params, data_path):
    return joblib.load(data_path + 'Sij_net_ave' + freq_str(stat_params))


def calc_spiketrain_cross_correlations(sim_params, stat_params, data_path):
    """
    The total average spike-train cross-spectrum, Sij, used in eq. (25).
    :param sim_params:
    :param stat_params:
    :param data_path:
    :return:
    """
    X_curr, X_noise, X_rm = load_susceptibilities(sim_params, stat_params, data_path)
    Ss = signal_power_spectrum(sim_params)
    Sij_net = calc_average_network_cross_spectrum(sim_params, stat_params, data_path)
    Sij_total = Sij_net + X_rm['mag']**2 * Ss
    joblib.dump(Sij_total, data_path + 'Sij_total' + freq_str(stat_params))


def load_spiketrain_cross_correlations(sim_params, stat_params, data_path):
    return joblib.load(data_path + 'Sij_total' + freq_str(stat_params))


def calc_network_auto_spectra(sim_params, stat_params, data_path):
    mu_exc, mu_inh, mu_tot, mu_c = get_ave_input(sim_params, stat_params)
    X_curr, X_noise, X_rm = load_susceptibilities(sim_params, stat_params, data_path)
    pc = sim_params['synapse']['p_conn']

    S_lif = load_exact_diff_approx_power_spectrum(stat_params, data_path)
    Ne, Ni = sim_params['net']['N_e'], sim_params['net']['N_i']

    def Snet(mu_1, N_1):
        return S_lif * (X_rm['complex'] * mu_1 / N_1
                        +
                        np.conj(X_rm['complex']) * np.conj(mu_1) / N_1
                        +
                        X_curr['mag']**2 * mu_c * (1 - pc) / pc
                        +
                        X_rm['mag']**2 * mu_c)

    results = {'ee': Snet(mu_exc, Ne), 'ii': Snet(mu_inh, Ni)}
    joblib.dump(results, data_path + 'Sii_net_populations' + freq_str(stat_params))
    return results


def calc_average_network_auto_spectrum(sim_params, stat_params, data_path):
    Snet = calc_network_auto_spectra(sim_params, stat_params, data_path)
    Ne, Ni = sim_params['net']['N_e'], sim_params['net']['N_i']
    N_tot = Ne + Ni
    Sii_ave = Snet['ee'] * Ne/N_tot + Snet['ii'] * Ni/N_tot
    joblib.dump(Sii_ave, data_path + 'Sii_net_ave' + freq_str(stat_params))
    return Sii_ave


def load_average_network_auto_spectrum(sim_params, stat_params, data_path):
    return joblib.load(data_path + 'Sii_net_ave' + freq_str(stat_params))


def calc_spiketrain_auto_correlations(sim_params, stat_params, data_path):
    X_curr, X_noise, X_rm = load_susceptibilities(sim_params, stat_params, data_path)
    Ss = signal_power_spectrum(sim_params)
    Sii_net = calc_average_network_auto_spectrum(sim_params, stat_params, data_path)
    S0 = load_exact_diff_approx_power_spectrum(stat_params, data_path)
    Sii_total = S0 + Sii_net + X_rm['mag']**2 * Ss
    assert np.all(Sii_total.imag == 0)
    joblib.dump(Sii_total, data_path + 'Sii_total' + freq_str(stat_params))


def load_spiketrain_auto_correlations(sim_params, stat_params, data_path):
    return joblib.load(data_path + 'Sii_total' + freq_str(stat_params))


# ------------------------------------------------------------------------- #
#                        Signal Cross-spectra                               #
# ------------------------------------------------------------------------- #
def calc_signal_cross_spectra(sim_params, stat_params, save_dir):
    X_curr, X_noise, X_rm = load_susceptibilities(sim_params, stat_params, save_dir)
    Ss = signal_power_spectrum(sim_params)

    Sxs = X_rm['complex'] * Ss

    SXs = stat_params['N_obs'] * Sxs

    B = box_filter(sim_params, stat_params)
    SAs = B * Sxs

    sig_A = np.sqrt(activity_variance(sim_params, stat_params, save_dir))
    alpha = sync_alpha(sim_params, stat_params, sig_A)
    Sphis = alpha * SAs

    results = {'Sxs': Sxs, 'SXs': SXs, 'SAs': SAs, 'Sphis':Sphis}
    joblib.dump(results, save_dir + 'S_Ys_' + freq_str(stat_params))


def load_signal_cross_spectra(sim_params, stat_params, save_dir):
    return joblib.load(save_dir + 'S_Ys_' + freq_str(stat_params))


# ------------------------------------------------------------------------- #
#                         Power Spectrum                                    #
# ------------------------------------------------------------------------- #
def signal_power_spectrum(sim_params):
    return sim_params['signal']['sig_s']**2 / (2 * sim_params['signal']['f_c'])


# --------------------------Unperturbed------------------------------------ #
def load_exact_diff_approx_power_spectrum(stat_params, save_dir):
    return joblib.load(save_dir + 'PS_DA_exact' + freq_str(stat_params))


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


def calc_unperturbed_power_spectrum(params, stat_params, save_dir):
    nrn = params['neuron']
    r0_sc, mu_sc, D_sc = self_consistent_rate(params, tol=1e-4)
    S_lif = setup_Sxx(r0=r0_sc, mu=mu_sc, D=D_sc, vt=nrn['vt'], vr=nrn['vr'], taum=nrn['taum'], tref=nrn['tref'])

    freq = stat_params['theory_freq']
    ps = np.zeros(freq.size)
    for f_i, f in enumerate(freq):
        print(f'\r{f_i + 1} / {freq.size}', end='', flush=True)
        ps[f_i] = float(S_lif(f))
    joblib.dump(ps, save_dir + 'PS_DA_exact' + freq_str(stat_params))
# ------------------------------------------------------------------------- #


# ---------------------------All-spike------------------------------------- #
def allspike_power_spectrum(sim_params, stat_params, save_dir):
    """
    All-spike power spectrum from eq. (25)
    :param sim_params: model parameters
    :param stat_params: statistical parameters for measures
    :param save_dir: directory where data is located
    :return: the all-spike power spectrum, S_X(f)
    """
    S_ij = load_spiketrain_cross_correlations(sim_params, stat_params, save_dir)
    S_ii = load_spiketrain_auto_correlations(sim_params, stat_params, save_dir)
    N_A = stat_params['N_obs']
    S_pop = N_A * S_ii  + N_A * (N_A - 1) * S_ij
    return S_pop


# ---------------------------Activity-------------------------------------- #
def activity_power_spectrum(sim_params, stat_params, save_dir):
    """
    Activity power spectrum from eq. (29)
    :param sim_params: model parameters
    :param stat_params: statistical parameters for measures
    :param save_dir: directory where data is located
    :return: the activity power spectrum, S_A(f)
    """
    N_A = stat_params['N_obs']
    B = box_filter(sim_params, stat_params)
    return (np.abs(B)/N_A)**2 * allspike_power_spectrum(sim_params, stat_params, save_dir)


def activity_autocorrelation(sim_params, stat_params, save_dir):
    dt = sim_params['dt']
    f_nyq = 0.5 / dt
    df = stat_params['Caa_df']  # subsample the frequency spectrum for faster results
    subsampled_freq = np.arange(0, f_nyq + df, df)
    Sa_log = activity_power_spectrum(sim_params, stat_params, save_dir)
    Sa_lin = np.interp(subsampled_freq, stat_params['theory_freq'], Sa_log)

    """The n value is the size of the OUTPUT.  Because it is larger than m, the size of Sa_lin, Sa_lin is padded with
       zeros, or is doubled with the second half (negative frequency terms) padded with zeros.
       Then when taking the forward transform, the output will be n//2 + 1 = 2 (m - 1)//2 + 1 == m.  
       Therefore, Caa and Sa_lin will have the same length.
       The norm is set to None (which is the default 'backward'), meaning the ifft is normalized by 1/m_fft
       note that m_fft * df ~ 1/dt

       See notes: tangents_and_isolated_examples: FFT_scaling.ipynb
    """
    m_fft = f_nyq / df + 1
    assert m_fft == Sa_lin.size
    Caa = np.fft.irfft(Sa_lin) / dt
    assert Caa.size == 2 * (m_fft - 1)
    return Caa, subsampled_freq


def activity_variance(sim_params, stat_params, save_dir):
    Caa, _ = activity_autocorrelation(sim_params, stat_params, save_dir)
    return Caa[0]
# ------------------------------------------------------------------------- #


# ---------------------------Synchrony------------------------------------- #
def integrate_Mehler_formula(rho_A_of_tau, beta_sq):
    def integr(a):
        return exp(-beta_sq / (1 + a)) / sqrt(1 - a**2)
    result = quad(integr, [0, rho_A_of_tau], verbose=False)
    return result


def get_var_s_hat_save_str(stat_params):
    return f"delta={stat_params['act_delta']}" if stat_params['act_delta'] else f"R0={stat_params['R0']}"


def calc_sync_power_spectrum(sim_params, stat_params, save_dir):
    var_A = activity_variance(sim_params, stat_params, save_dir)
    beta_sq = float(sync_beta(sim_params, stat_params, np.sqrt(var_A))) ** 2

    Caa, subsampled_freq = activity_autocorrelation(sim_params, stat_params, save_dir)
    rho_A = Caa / var_A
    Cyy = np.zeros(rho_A.shape, dtype='float')  # @Note: autocorrelation functions are always real (symmetric).

    for rho_i, rho in enumerate(rho_A.real):
        print(f'\r{rho_i + 1}/{rho_A.size}', end='', flush=True)
        Cyy[rho_i] = integrate_Mehler_formula(rho, beta_sq)

    Cyy /= 2 * np.pi
    dt = sim_params['dt']
    results = {'S_Phi': np.fft.rfft(Cyy) * dt, 'freqs': subsampled_freq}
    joblib.dump(results, save_dir + f"Sy_phi={stat_params['act_thresh']}_df={stat_params['Caa_df']}_"
                + get_var_s_hat_save_str(stat_params))


def load_sync_power_spectrum(sim_params, stat_params, save_dir):
    return joblib.load(save_dir + f"Sy_phi={stat_params['act_thresh']}_df={stat_params['Caa_df']}_"
                       + get_var_s_hat_save_str(stat_params))



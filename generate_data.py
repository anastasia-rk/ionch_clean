# this code runs the generation of synthetic data for the HH and Kemp models
# imports

from load_protocols import *
import load_protocols
# defnitions
def hh_model(t, x, theta):
    a, r = x[:2]
    *p, g = theta[:9]
    v = V(t)
    k1 = p[0] * np.exp(p[1] * v)
    k2 = p[2] * np.exp(-p[3] * v)
    k3 = p[4] * np.exp(p[5] * v)
    k4 = p[6] * np.exp(-p[7] * v)
    tau_a = 1 / (k1 + k2)
    a_inf = tau_a * k1
    tau_r = 1 / (k3 + k4)
    r_inf = tau_r * k4
    da = (a_inf - a) / tau_a
    dr = (r_inf - r) / tau_r
    return [da,dr]

def hh_model_markov(t, x, theta):
    C, s_I, s_O = x[:3]
    *p, g = theta[:9]
    v = V(t)
    k1 = p[0] * np.exp(p[1] * v)
    k2 = p[2] * np.exp(-p[3] * v)
    k3 = p[4] * np.exp(p[5] * v)
    k4 = p[6] * np.exp(-p[7] * v)
    IC = 1 - C - s_I - s_O
    dC = C * (-k1 - k3) + IC * k4 + k2 * s_O
    ds_I = IC * k1 + k3 * s_O + s_I * (-k2 - k4)
    ds_O = C * k1 + k4 * s_I + s_O * (-k2 - k3)
    return [dC, ds_I, ds_O]

def hh_model_markov_ss(t, x, theta, v):
    C, s_I, s_O = x[:3]
    *p, g = theta[:9]
    k1 = p[0] * np.exp(p[1] * v)
    k2 = p[2] * np.exp(-p[3] * v)
    k3 = p[4] * np.exp(p[5] * v)
    k4 = p[6] * np.exp(-p[7] * v)
    IC = 1 - C - s_I - s_O
    dC = C * (-k1 - k3) + IC * k4 + k2 * s_O
    ds_I = IC * k1 + k3 * s_O + s_I * (-k2 - k4)
    ds_O = C * k1 + k4 * s_I + s_O * (-k2 - k3)
    return [dC, ds_I, ds_O]

def hh_model_ss_analytical(theta, v):
    *p, g = theta[:9]
    k1 = p[0] * np.exp(p[1] * v)
    k2 = p[2] * np.exp(-p[3] * v)
    k3 = p[4] * np.exp(p[5] * v)
    k4 = p[6] * np.exp(-p[7] * v)
    tau_a = 1 / (k1 + k2)
    a_inf = tau_a * k1
    tau_r = 1 / (k3 + k4)
    r_inf = tau_r * k4
    return [a_inf, r_inf]

def kemp_model(t, x, theta):
    # this function computes the ODE for the Kemp model: independent gating
    # activation: 3 state Markov C2 - C1 - O
    # inactivation: 2 state HH I - O
    op, c1, h = x[:3]
    *p, g = theta[:13]
    v = V(t)
    a1 = p[0] * np.exp(p[1] * v)
    b1 = p[2] * np.exp(-p[3] * v)

    ah = p[6] * np.exp(-p[7] * v)
    bh = p[4] * np.exp(p[5] * v)

    a2 = p[8] * np.exp(p[9] * v)
    b2 = p[10] * np.exp(-p[11] * v)
    dop = a2*c1 - b2*op
    dc1 = b2*op + a1*(1 - op - c1) - (a2 + b1)*c1
    # inactivation moodel is indepen
    tau_h = 1 / (ah + bh)
    h_inf = tau_h * ah
    dh = (h_inf - h)/tau_h
    return [dop,dc1,dh]

def kemp_model_markov(t, x, theta):
    # this function computes the ODE for the Kemp model: independent gating
    # activation: 3 state Markov C2 - C1 - O
    # inactivation: 2 state HH I - O
    c1, c2, i, ic1, ic2, o = x[:6]
    *p, g = theta[:13]
    v = V(t)
    a1 = p[0] * np.exp(p[1] * v)
    b1 = p[2] * np.exp(-p[3] * v)

    ah = p[6] * np.exp(-p[7] * v)
    bh = p[4] * np.exp(p[5] * v)

    a2 = p[8] * np.exp(p[9] * v)
    b2 = p[10] * np.exp(-p[11] * v)

    dc1 = a1 * c2 + ah * ic1 + b2 * o - (b1 + bh + a2) * c1
    dc2 = b1 * c1 + ah * ic2 - (a1 + bh) * c2
    di = a2 * ic1 + bh * o - (b2 + ah) * i
    dic1 = a1 * ic2 + bh * c1 + b2 * i - (b1 + ah + a2) * ic1
    dic2 = b1 * ic1 + bh * c2 - (ah + a1) * ic2
    do = a2 * c1 + ah * i - (b2 + bh) * o

    return [dc1, dc2, di, dic1, dic2, do]

def kemp_model_markov_ss(t, x, theta, v):
    # Markovian representation of the Kemp model
    c1, c2, i, ic1, ic2, o = x[:6]
    *p, g = theta[:13]
    a1 = p[0] * np.exp(p[1] * v)
    b1 = p[2] * np.exp(-p[3] * v)

    ah = p[6] * np.exp(-p[7] * v)
    bh = p[4] * np.exp(p[5] * v)

    a2 = p[8] * np.exp(p[9] * v)
    b2 = p[10] * np.exp(-p[11] * v)

    dc1 = a1 * c2 + ah * ic1 + b2 * o - (b1 + bh + a2) * c1
    dc2 = b1 * c1 + ah * ic2 - (a1 + bh) * c2
    di = a2 * ic1 + bh * o - (b2 + ah) * i
    dic1 = a1 * ic2 + bh * c1 + b2 * i - (b1 + ah + a2) * ic1
    dic2 = b1 * ic1 + bh * c2 - (ah + a1) * ic2
    do = a2 * c1 + ah * i - (b2 + bh) * o

    return [dc1, dc2, di, dic1, dic2, do]

def kemp_model_ss(t, x, theta, v):
    # this function computes the steady state of the Kemp model at voltage v
    op, c1, h = x[:3]
    *p, g = theta[:13]
    a1 = p[0] * np.exp(p[1] * v)
    b1 = p[2] * np.exp(-p[3] * v)

    ah = p[6] * np.exp(-p[7] * v)
    bh = p[4] * np.exp(p[5] * v)

    a2 = p[8] * np.exp(p[9] * v)
    b2 = p[10] * np.exp(-p[11] * v)
    dop = a2*c1 - b2*op
    dc1 = b2*op + a1*(1 - op - c1) - (a2 + b1)*c1
    tau_h = 1/(ah + bh)
    h_inf = tau_h * ah
    dh = (h_inf - h)/tau_h
    return [dop,dc1,dh]

def wang_model(t, x, theta):
    # Wang model with MM model of joint gating
    # C3 <- (b_a0, a_a0) -> C2 <-(k_b, k_f)-> C1 <-(b_a1, a_a1)-> O <-(b_1, a_1)-> I
    # activation: 4 state MM
    # inactivation" 2 state MM
    op, c1, c2, c3, i_h = x[:5]
    *p, g = theta[:15]
    v = V(t)
    a_a0 = p[0]*np.exp(p[1] * v)
    b_a0 = p[2]*np.exp(-p[3] * v)
    a_a1 = p[4] * np.exp(p[5] * v)
    b_a1 = p[6] * np.exp(-p[7] * v)
    k_f = p[8]
    k_b = p[9]
    # inactivation model
    a_1 = p[10] * np.exp(p[11] * v)
    b_1 = p[12] * np.exp(-p[13] * v)

    dop = a_a1 * c1 + b_1 * i_h - (a_1 + b_a1) * op
    dc1 = k_f * c2 + b_a1 * op - (a_a1 + k_b) * c1
    dc2 = a_a0 * c3 + k_b * c1 - (k_f + b_a0) * c2
    dc3 = b_a0 * c2 - a_a0 * c3
    dih = a_1 * op - b_1 * i_h
    return [dop, dc1, dc2, dc3, dih]

def wang_model_ss(t, x, theta, v):
    # Wang model to get the steady state
    # with MM model of joint gating
    # C3 <- (b_a0, a_a0) -> C2 <-(k_b, k_f)-> C1 <-(b_a1, a_a1)-> O <-(b_1, a_1)-> I
    # activation: 4 state MM
    # inactivation" 2 state MM
    op, c1, c2, c3, i_h = x[:5]
    *p, g = theta[:15]
    a_a0 = p[0]*np.exp(p[1] * v)
    b_a0 = p[2]*np.exp(-p[3] * v)
    a_a1 = p[4] * np.exp(p[5] * v)
    b_a1 = p[6] * np.exp(-p[7] * v)
    k_f = p[8]
    k_b = p[9]
    # inactivation model
    a_1 = p[10] * np.exp(p[1] * v)
    b_1 = p[2] * np.exp(-p[3] * v)

    dop = a_a1 * c1 + b_1 * i_h - (a_1 + b_a1) * op
    dc1 = k_f * c2 + b_a1 * op - (a_a1 + k_b) * c1
    dc2 = a_a0 * c3 + k_b * c1 - (k_f + b_a0) * c2
    dc3 = b_a0 * c2 - a_a0 * c3
    dih = a_1 * op - b_1 * i_h
    return [dop, dc1, dc2, dc3, dih]

def two_state_model_log(t, x, theta):
    # this function computes the ODE for the HH dynamics with parameters given in log space
    a, r = x[:2]
    p = theta[:8]
    v = V(t)
    k1 = np.exp(p[0] + np.exp(p[1]) * v)
    k2 = np.exp(p[2] - np.exp(p[3]) * v)
    k3 = np.exp(p[4] + np.exp(p[5]) * v)
    k4 = np.exp(p[6] - np.exp(p[7]) * v)
    tau_a = 1 / (k1 + k2)
    a_inf = tau_a * k1
    tau_r = 1 / (k3 + k4)
    r_inf = tau_r * k4
    da = (a_inf - a) / tau_a
    dr = (r_inf - r) / tau_r
    return [da,dr]

def observation_kemp(t, ode_solution, theta, snr=None):
    # this function computes the observation from the ODE solution using Kemp dynamics
    x = ode_solution.sol(t)
    op, c1, h = x[:3]
    *p, g = theta[:13]
    noiseless_output = g * op * h * (V(t) - EK)
    # compute power of noiseless output
    signal_power = np.mean(noiseless_output ** 2)
    if snr is None:
        return noiseless_output
    else:
        sigma_noise = np.sqrt(signal_power / snr)
        return  noiseless_output + np.random.normal(0, sigma_noise, size=len(t))

def observation_kemp_markov(t, ode_solution, theta, snr=None):
    # this function computes the observation from the ODE solution using Kemp dynamics
    x = ode_solution.sol(t)
    c1, c2, i, ic1, ic2, o = x[:6]
    *p, g = theta[:13]
    noiseless_output = g * o * (V(t) - EK)
    # compute power of noiseless output
    signal_power = np.mean(noiseless_output ** 2)
    if snr is None:
        return noiseless_output
    else:
        sigma_noise = np.sqrt(signal_power / snr)
        return  noiseless_output + np.random.normal(0, sigma_noise, size=len(t))

def observation_wang(t, ode_solution, theta, snr=None):
    # this function computes the observation from the ODE solution using Kemp dynamics
    x = ode_solution.sol(t)
    op, c1, c2, c3, i_h = x[:5]
    *p, g = theta[:15]
    noiseless_output = g * op * (V(t) - EK)
    # compute power of noiseless output
    signal_power = np.mean(noiseless_output ** 2)
    if snr is None:
        return noiseless_output
    else:
        sigma_noise = np.sqrt(signal_power / snr)
        return  noiseless_output + np.random.normal(0, sigma_noise, size=len(t))

def observation_hh(t, ode_solution, theta, snr=None):
    # this function computes the observation from the ODE solution using HH dynamics
    x = ode_solution.sol(t)
    a, r = x[:2]
    *ps, g = theta[:9]
    # generate noise from normal distr to match dimension of the output
    noiseless_output = g * a * r * (V(t) - EK)
    # compute power of noiseless output
    signal_power = np.mean(noiseless_output ** 2)
    if snr is None:
        return noiseless_output
    else:
        sigma_noise = np.sqrt(signal_power / snr)
        return  noiseless_output + np.random.normal(0, sigma_noise, size=len(t))

def observation_hh_markov(t, ode_solution, theta, snr=None):
    # observation model to match the HH model with Markovian gating
    x = ode_solution.sol(t)
    C, s_I, s_O = x[:3]
    *ps, g = theta[:9]
    noiseless_output = g * s_O * (V(t) - EK)
    # compute power of noiseless output
    signal_power = np.mean(noiseless_output ** 2)
    if snr is None:
        return noiseless_output
    else:
        sigma_noise = np.sqrt(signal_power / snr)
        return  noiseless_output + np.random.normal(0, sigma_noise, size=len(t))

def observation_markovian_convention(t, ode_solution, theta, snr=None):
    x = ode_solution.sol(t)
    open_state = x[-1,:]
    *ps, g = theta[:9]
    noiseless_output = g * open_state * (V(t) - EK)
    # compute power of noiseless output
    signal_power = np.mean(noiseless_output ** 2)
    if snr is None:
        return noiseless_output
    else:
        sigma_noise = np.sqrt(signal_power / snr)
        return  noiseless_output + np.random.normal(0, sigma_noise, size=len(t))

def observation_direct_input(x, v, theta):
    # this function computes the observation from the ODE solution and the voltage passed as time series
    a, r = x[:2]
    *ps, g = theta[:9]
    # generate noise from normal distr to match dimension of the output
    noiseless_output = g * a * r * (v - EK)
    return noiseless_output

def observation_log(t, ode_solution, theta):
    # this function computes the observation from the approximate ODE solution as array of states
    x = ode_solution
    *ps_log, g_log = theta[:9]
    # generate noise from normal distr to match dimension of the output
    noiseless_output = np.exp(g_log) * np.prod(x, axis=0) * (V(t) - EK)
    return  noiseless_output

def generate_synthetic_data(model_name, thetas_true, times):
    # this function generates synthetic data for the model specified by model_name given the true parameters thetas_true
    # check if model name is within list of available models
    if model_name.lower() not in available_models:
        raise ValueError(f'Unknown model name: {model_name}. Available models are: {available_models}')
    elif model_name.lower() == 'hh':
        # initialise and solve ODE
        x0 = [0, 1, 0]
        t_end = 10e3  # run for a long time to make sure that the steady state is reached for the slow gating variables
        v_ss = V(times[0])
        solution_ss = sp.integrate.solve_ivp(hh_model_markov_ss, [0, t_end], x0, args=[thetas_true, v_ss], dense_output=True, method='LSODA',rtol=1e-8, atol=1e-8)
        ss = solution_ss.sol(t_end)
        # ss = hh_model_ss_analytical(thetas_true, v_ss)
        solution = sp.integrate.solve_ivp(hh_model_markov, [0, times[-1]], ss, args=[thetas_true], dense_output=True, method='LSODA',rtol=1e-8, atol=1e-8)
        current_model = observation_hh_markov
    elif model_name.lower() == 'kemp':
        # find steady state at -80mV to use as initial condition
        x0_init = [0, 0.85, 0, 0, 0.15, 0]
        # run for a long time for the slow rate states to settle
        t_end = 10e3 # run for a long time to make sure that the steady state is reached for the slow gating variables
        v_ss = V(times[0]) # the voltage for which we wish to compute the steady state
        solution_ss = sp.integrate.solve_ivp(kemp_model_markov_ss, [0, t_end], x0_init, args=[thetas_true, v_ss], dense_output=True,
                                             method='LSODA', rtol=1e-8, atol=1e-8)
        ss = solution_ss.sol(t_end)
        # obtain solution initialised at SS
        # ss = x0_init
        solution = sp.integrate.solve_ivp(kemp_model_markov, [0, times[-1]], ss, args=[thetas_true], dense_output=True,
                                              method='LSODA', rtol=1e-8, atol=1e-8)
        current_model = observation_kemp_markov
    elif model_name.lower() == 'wang':
        # find steady state at -80mV to use as initial condition
        x0_init = [0.2, 0.2, 0.2, 0.2, 0.2]
        # run for a long time for the slow rate states to settle
        t_end = 10e3  # run for a long time to make sure that the steady state is reached for the slow gating variables
        v_ss = V(times[0])  # the voltage for which we wish to compute the steady state
        solution_ss = sp.integrate.solve_ivp(wang_model_ss, [0, t_end], x0_init, args=[thetas_true, v_ss],
                                             dense_output=True,
                                             method='LSODA', rtol=1e-8, atol=1e-8)
        ss = solution_ss.sol(t_end)
        # obtain solution initialised at SS
        solution = sp.integrate.solve_ivp(wang_model, [0, times[-1]], ss, args=[thetas_true], dense_output=True,
                                          method='LSODA', rtol=1e-8, atol=1e-8)
        current_model = observation_wang
    return solution, current_model

def split_generated_data_into_segments(solution, modelled_current, jump_indeces, times):
    states_roi = []
    states_known_roi = []
    current_roi = []
    for iJump, jump in enumerate(jump_indeces[:-1]):  # loop oversegments (nJumps - )
        # define a region of interest - we will need this to preserve the
        # trajectories of states given the full clamp and initial position, while
        ROI_start = jump
        ROI_end = jump_indeces[iJump + 1] + 1  # add one to ensure that t_end equals to t_start of the following segment
        ROI = times[ROI_start:ROI_end]
        states_model = solution.sol(ROI)
        # save states
        states_roi.append(states_model)
        states_known_roi.append([1] * len(ROI))  # adding ones in case we have situation where one of the known states is involved in output fn
        # save current
        current_roi.append(modelled_current[ROI_start:ROI_end])
        # initial values for the betas
    # finish loop over segments
    return states_roi, states_known_roi, current_roi

# available models to generate synthetic data
available_models = ['hh', 'kemp','wang']
# set resting potential
EK = -80
# possible parameter combinations
thetas_hh_baseline = [2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524]
# From Chon's paper on temperature dependence of HH model
thetas_hh_25 = [7.65e-5, 9.05e-2, 2.84e-5, 4.74e-2, 1.03e-1, 2.13e-2, 8.01e-3, 2.96e-2, 3.1e-2]
thetas_hh_37 = [2.07e-3, 7.17e-2, 3.44e-5, 6.18e-2, 4.18e-1, 2.58e-2, 4.57e-2, 2.51e-2, 3.33e-2]
# Changed conductances
# thetas_hh_25 = [7.65e-5, 9.05e-2, 2.84e-5, 4.74e-2, 1.03e-1, 2.13e-2, 8.01e-3, 2.96e-2, 8.471005e-02]
# thetas_hh_37 = [2.07e-3, 7.17e-2, 3.44e-5, 6.18e-2, 4.18e-1, 2.58e-2, 4.57e-2, 2.51e-2, 8.471005e-02]
# Kemp model:
thetas_kemp = [8.5318e-03, 8.3176e-02, 1.2628e-02, 1.03628e-07, 2.702763e-01, 1.580004e-02, 7.6669948e-02,
                  2.2457500e-02, 1.490338e-01, 2.431569e-02, 5.58072e-04, 4.06619e-02, 8.471005e-02]
# Wang model
thetas_wang = [0.022348, 0.01176, 0.047002, 0.0631, 0.013733, 0.038198, 0.0000689, 0.04178, 0.023761, 0.036778,
               0.090821, 0.023391, 0.006497, 0.03268, 8.471005e-02]
####################################################################################################################
if __name__ == '__main__':
    # test generating data for multiple segments
    # select the model that we will use to generate the synthetic data
    load_protocols # this module will load the voltage protocol ad give times of interest so we dont have to generate it again
    ## define the time interval on which the fitting will be done
    tlim = [0, 14899]
    times = np.linspace(*tlim, tlim[-1] - tlim[0], endpoint=False)
    del tlim
    # generate the segments with B-spline knots and intialise the betas for splines
    jump_indeces, times_roi, voltage_roi, knots_roi, collocation_roi, spline_order = generate_knots(times)
    nSegments = len(jump_indeces[:-1])
    print('Inner optimisation is split into ' + str(nSegments) + ' segments based on protocol steps.')
    state_names = ['a', 'r'] # how many states we have in the model that we are fitting
    nBsplineCoeffs = (len(knots_roi[0]) - spline_order - 1) * len(state_names)
    init_betas_roi = nSegments * [0.5 * np.ones(nBsplineCoeffs)]
    print('Number of B-spline coeffs per segment: ' + str(nBsplineCoeffs) +'.')
    ####################################################################################################################
    # parameters needed to generate synthetic data
    model_name = 'kemp'
    if model_name.lower() not in available_models:
        raise ValueError(f'Unknown model name: {model_name}. Available models are: {available_models}.')
    elif model_name.lower() == 'hh':
        y_labels8 = ['C', 's_I', 's_O', 'Current, nA', 'Voltage, mV']
        thetas_true = thetas_hh_37
    elif model_name.lower() == 'kemp':
        y_labels8 = ['c1', 'c2', 'i', 'ic1', 'ic2', 'o', 'Current, nA', 'Voltage, mV']
        thetas_true = thetas_kemp
    elif model_name.lower() == 'wang':
        thetas_true = thetas_wang
        y_labels8 = ['op', 'c1', 'c2', 'c3', 'ih', 'Current, nA', 'Voltage, mV']

    solution, current_model = generate_synthetic_data(model_name, thetas_true, times)
    true_states = solution.sol(times)
    # set signal to noise ratio in decibels
    snr_db = 30
    snr = 10 ** (snr_db / 10)
    current_test = current_model(times, solution, thetas_true, snr=snr)
    # # test generating niseless data
    # current_test_noiseless = current_model(times, solution, thetas_true)
    ####################################################################################################################
    ## create multiple segments limited by time instances of jumps
    states_roi, states_known_roi, current_roi = split_generated_data_into_segments(solution, current_test, jump_indeces, times)
    # create a current_true list out of current_roi - we will need it for the outer optimisation
    # use list comprehension but remember to remove the end point from each segment - we needed them for continuity in segments
    current_true = [item for sublist in current_roi for item in sublist[:-1]]
    # add the last point from the last segment
    current_true += [current_roi[-1][-1]]
    ####################################################################################################################
    ## plots to test model outputs
    # plot the current generated at the entire protocol length and the current from segments
    ylabels = ['Current, pA', ' error in Current, pA']
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(12, 6))
    axes = axes.ravel()
    axes[0].plot(times, current_test, '--k', linewidth=1, label='Current generated at the entire segment')
    axes[0].plot(times, current_true,':m', linewidth=1, label='Current from segments')
    axes[1].plot(times, current_test - current_true, '--k', linewidth=1, label='Error in current')
    for iAx, ax in enumerate(axes):
        ax.set_xlabel('Time, ms')
        ax.set_ylabel(ylabels[iAx])
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
        ax.legend(loc='best')
    plt.tight_layout()
    fig.savefig('Figures/current_true_test_model_'+model_name+'.png', dpi=400)

    fig, axes = plt.subplots(2, 2)
    axes = axes.ravel()
    axes[0].plot(times, solution.sol(times)[0, :], '--k', label=model_name)
    axes[0].set_ylabel('a/O gating variable')
    axes[1].plot(times, solution.sol(times)[-1, :], '--k', label=model_name)
    axes[1].set_ylabel('r/h gating variable')
    volts = V(times)
    axes[2].plot(times, V(times), 'k--', label='Input voltage')
    axes[2].plot(times[jump_indeces], volts[jump_indeces], 'm.')
    axes[2].set_ylabel('voltage, mV')
    axes[3].plot(times, current_true, '--k', label=model_name)
    axes[3].set_ylabel('Current, A')
    for ax in axes:
        ax.legend(fontsize=14, loc='best')
        ax.set_xlabel('times, ms')
        ax = pretty_axis(ax)
    plt.tight_layout()
    fig.savefig('Figures/generated_data_test_model_'+model_name+'.png',dpi=400)

    # plot output for poster
    fig, ax = plt.subplots(figsize=(5,2))
    ax.plot(times, current_true,'-k', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time, ms')
    ax.set_ylabel('Current, nA')
    ax = pretty_axis(ax, legendFlag=False)
    plt.tight_layout()
    fig.savefig('Figures/current_true_gen_model_'+model_name+'.png', dpi=400)

    # test running things for the AP protocol
    load_protocols.volts_interpolated = volts_interpolated_ap
    voltage_ap = V(times_ap)
    solution, current_model = generate_synthetic_data(model_name, thetas_true, times_ap)
    current_ap_noiseless = current_model(times_ap, solution, thetas_true)
    # plot the states at times_ap, voltage and current of the true model
    fig8, axes8 = plt.subplots(len(y_labels8), 1, figsize=(10, 15), sharex=True)
    for iAx, ax in enumerate(axes8):
        ax.set_ylabel(y_labels8[iAx], fontsize=12)
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
    axes8[-1].set_xlabel('Time [ms]')
    axes8[-1].plot(times_ap, voltage_ap, '-k', label='Voltage', linewidth=1.5, alpha=0.27)
    axes8[-2].plot(times_ap, current_ap_noiseless, '-k', label='True current', linewidth=1.5, alpha=0.27)
    for iState in range(len(y_labels8)-2):
        axes8[iState].plot(times_ap, solution.sol(times_ap)[iState, :], '-k', label='True state', linewidth=1.5, alpha=0.27)
    # save
    fig8.tight_layout()
    fig8.savefig('Figures/' + model_name.lower() +  '_check_on_ap_protocol.png', dpi=400)

    # test complete
    print('Produced synthetic data for model training based on the pre-loaded voltage protocol.')

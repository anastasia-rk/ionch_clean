# imports
from generate_data import *
import time as tm
from functools import partial

# definitions
debug = False
def simulate_segment(parameters, times, support, first_spline_coeffs = None):
    # given times and return the simulated values
    coeffs_a, coeffs_r = np.split(parameters, 2)
    if first_spline_coeffs is not None:
        # if we have the parameters put in for first coeffs, we have to insert them into the array
        first_spline_coeff_a, first_spline_coeff_r = first_spline_coeffs
        coeffs_a = np.insert(coeffs_a, 0, first_spline_coeff_a)
        coeffs_r = np.insert(coeffs_r, 0, first_spline_coeff_r)
    tck_a = (support, coeffs_a, spline_order)
    tck_r = (support, coeffs_r, spline_order)
    dot_a = sp.interpolate.splev(times, tck_a, der=1)
    dot_r = sp.interpolate.splev(times, tck_r, der=1)
    fun_a = sp.interpolate.splev(times, tck_a, der=0)
    fun_r = sp.interpolate.splev(times, tck_r, der=0)
    # the RHS must be put into an array
    dadr = fitted_model(times, [fun_a, fun_r], Thetas_ODE)
    rhs_theta = np.array(dadr)
    spline_surface = np.array([fun_a, fun_r])
    spline_deriv = np.array([dot_a, dot_r])
    # pack all required variables into the same array - will be the wrong orientation from pints preferred nTimes x nOutputs
    packed_output = np.concatenate((spline_surface, spline_deriv, rhs_theta), axis=0)
    return np.transpose(packed_output)

def multiobjective_for_segment(betas, segment, volts_segment, current_segment, support_segment, model):
    model_output = model(betas, segment, support_segment)
    x, x_dot, rhs = np.split(model_output, 3, axis=1)  # we split the array into states, state derivs, and RHSs
    # compute the data fit
    # *ps, g = Thetas_ODE[:9]
    # mini_tic = tm.time()
    # d_y = g * np.prod(x, axis=1) * (volts_segment - EK) - current_segment
    # test if it is faster to call normal observation function
    # mini_tac = tm.time()
    d_y = observation_direct_input(x.transpose(),volts_segment,Thetas_ODE) - current_segment
    # mini_toc = tm.time()
    # print ('Time without the observation function: ' + str(mini_tac - mini_tic) + ' s.')
    # print ('Time for the observation function: ' + str(mini_toc - mini_tac) + ' s.')
    data_fit_cost = np.transpose(d_y) @ d_y
    # compute the gradient matching cost
    d_deriv = (x_dot - rhs) ** 2
    axis_of_time = np.argmax(d_deriv.shape)
    integral_quad = sp.integrate.simpson(y=d_deriv, even='avg', axis=axis_of_time)
    gradient_match_cost = np.sum(integral_quad, axis=0)
    # not the most elegant implementation because it just grabs global lambda
    return data_fit_cost + lambd * gradient_match_cost

def optimise_segment(roi,input_roi,output_roi,support_roi,first_spline_coeffs=None):
    # global nBsplineCoeffs
    if first_spline_coeffs is None:
        init_betas = 0.5 * np.ones(nBsplineCoeffs)
    else:
        init_betas = 0.5 * np.ones(nBsplineCoeffs - len(state_names))  # initial values of B-spline coefficients
    bounds_rect = tuple(len(init_betas) * [(0, 0.99)])
    #define partial object that takes first_spline_coeffs as a fixed arguement
    model_bsplines = partial(simulate_segment, first_spline_coeffs=first_spline_coeffs)
    res = sp.optimize.minimize(multiobjective_for_segment, init_betas,
                               args=(roi, input_roi, output_roi, support_roi, model_bsplines),
                               method=method_name, bounds=bounds_rect, tol=tolerance,
                               options={'disp': False, 'maxiter': max_iter_inner})
    betas_roi = res.x
    cost_roi = res.fun
    nEvaluations = res.nfev
    if debug:
      print('Optimisation success: ' + str(res.success) + '.')
    # if we are optimising the first segment, we just output the optimisation results
    if first_spline_coeffs is None:
        return betas_roi, cost_roi, nEvaluations
    # otherwise, we add the first b-spline coeficient to the front
    first_spline_coeff_a, first_spline_coeff_r = first_spline_coeffs
    coeffs_a, coeffs_r = np.split(betas_roi, 2)
    coeffs_a = np.insert(coeffs_a,0,first_spline_coeff_a)
    coeffs_r = np.insert(coeffs_r,0,first_spline_coeff_r)
    betas_roi_with_first_coeff = np.concatenate((coeffs_a,coeffs_r))
    return betas_roi_with_first_coeff, cost_roi, nEvaluations

def inner_optimisation_test(theta, weight, times_roi, voltage_roi, current_roi, knots_roi, collocation_roi):
    # extract settings
    global Thetas_ODE, lambd
    # nBsplineCoeffs, spline_order, nSegments, state_names = bspline_settings
    Thetas_ODE = theta.copy()
    lambd = weight
    # set placeholders for betas and other variables we wish to store over segments
    all_betas = []  # store the bspline coefficients
    all_costs = []  # store the inner costs
    end_of_roi = []  # store the final values of the states to ensure continuity
    # states, derivatives and RHSs
    state_of_roi = {key: [] for key in state_names}
    rhs_of_roi = {key: [] for key in state_names}
    deriv_of_roi = {key: [] for key in state_names}
    # loop over segments
    # check the costs for segments
    if debug:
        grad_cost_all_segments = 0
        data_cost_all_segments = 0
    # loop over segments
    for iSegment in range(nSegments):
        segment = times_roi[iSegment]
        input_segment = voltage_roi[iSegment]
        output_segment = current_roi[iSegment]
        support_segment = knots_roi[iSegment]
        collocation_segment = collocation_roi[iSegment]
        # note that segment 1 is different because we do not need to ensure continuity
        if iSegment == 0:
            first_spline_coeffs = None
            # betas, cost, nEvals = optimise_segment(segment, input_segment, output_segment, support_segment,
            #                                        first_spline_coeffs)
            index_start = 0  # from which timepoint to store the states
        else:
            # find the scaling coeff of the first height by matiching its height at t0 of the segment to the final value of the previous segment
            first_spline_coeff_a = end_of_roi[-1][0] / collocation_segment[0, 0]
            first_spline_coeff_r = end_of_roi[-1][1] / collocation_segment[0, 0]
            first_spline_coeffs = [first_spline_coeff_a, first_spline_coeff_r]
            index_start = 1  # from which timepoint to store the states
        betas, cost, nEvals = optimise_segment(segment, input_segment, output_segment, support_segment,
                                                   first_spline_coeffs)
        all_betas.append(betas)
        all_costs.append(cost)
        model_output_fit = simulate_segment(betas, segment, support_segment, first_spline_coeffs=None)
        state_at_sample, state_deriv_at_sample, rhs_at_sample = np.split(model_output_fit, 3, axis=1)
        if debug:
            index_start_segment = 0 #index_start
            current_at_sample = observation_direct_input(state_at_sample[index_start_segment:].transpose(), input_segment[index_start_segment:], Thetas_ODE)
            d_deriv = (np.array(state_deriv_at_sample) - np.array(rhs_at_sample)) ** 2
            # find the longer axis - note that our calculation here is inverted
            axis_of_time = np.argmax(d_deriv.shape)
            integral_quad = sp.integrate.simpson(y=d_deriv[index_start_segment:], even='avg', axis=axis_of_time)
            gradient_matching_cost = np.sum(integral_quad, axis=0)
            data_matching_cost = np.sum((current_at_sample - output_segment[index_start_segment:]) ** 2)
            print('Data matching cost: ' + str(data_matching_cost) + ' +  Gradient matching cost: ' + str(
                gradient_matching_cost) + '. = ' + str(data_matching_cost + lambd * gradient_matching_cost) + '.')
            # the sum of costs should be the same as the sum of cost of the segment returned by the optimiser
            print('Cost of segment from optimiser: ' + str(cost) + '.')
            grad_cost_all_segments += gradient_matching_cost
            data_cost_all_segments += data_matching_cost
        # save the final value of the segment
        end_of_roi.append(state_at_sample[-1, :])
        for iState, stateName in enumerate(state_names):
            state_of_roi[stateName] += list(state_at_sample[index_start:, iState])
            deriv_of_roi[stateName] += list(state_deriv_at_sample[index_start:, iState])
            rhs_of_roi[stateName] += list(rhs_at_sample[index_start:, iState])
    states_of_segments = [v for k, v in state_of_roi.items()]
    deriv_of_segments = [v for k, v in deriv_of_roi.items()]
    rhs_of_segments = [v for k, v in rhs_of_roi.items()]
    if debug:
        current_at_sample = observation_direct_input(np.array(states_of_segments), voltage, Thetas_ODE)
        d_deriv = (np.array(deriv_of_segments) - np.array(rhs_of_segments)) ** 2
        axis_of_time = np.argmax(d_deriv.shape)
        integral_quad = sp.integrate.simpson(y=d_deriv, even='avg', axis=axis_of_time)
        gradient_matching_cost = np.sum(integral_quad, axis=0)
        data_matching_cost = np.sum((current_at_sample - current_true) ** 2)
        print('Data matching cost: ' + str(data_matching_cost) + ' +  Gradient matching cost: ' + str(gradient_matching_cost) + '. = ' + str(data_matching_cost + lambd * gradient_matching_cost) + '.')
        print('Sum of costs over segments: ' + str(sum(all_costs)) + '.')  # should be the same as the sum of the costs
        print ('Sum of gradient matching costs over segments: ' + str(grad_cost_all_segments) + '.')
        print ('Sum of data matching costs over segments: ' + str(data_cost_all_segments) + '.')
        print('Combined cost: ' + str(data_cost_all_segments + lambd * grad_cost_all_segments) + '.')  # should be the same as the sum of all costs
    result = (all_betas, all_costs, states_of_segments)
    return result

def inner_optimisation(theta, weight, times_roi, voltage_roi, current_roi, knots_roi, collocation_roi):
    # extract settings
    global Thetas_ODE, lambd
    # nBsplineCoeffs, spline_order, nSegments, state_names = bspline_settings
    Thetas_ODE = theta.copy()
    lambd = weight
    # set placeholders for betas and other variables we wish to store over segments
    all_betas = []  # store the bspline coefficients
    end_of_roi = []  # store the final values of the states to ensure continuity
    # states, derivatives and RHSs
    state_of_roi = {key: [] for key in state_names}
    rhs_of_roi = {key: [] for key in state_names}
    deriv_of_roi = {key: [] for key in state_names}
    # loop over segments
    for iSegment in range(nSegments):
        segment = times_roi[iSegment]
        input_segment = voltage_roi[iSegment]
        output_segment = current_roi[iSegment]
        support_segment = knots_roi[iSegment]
        collocation_segment = collocation_roi[iSegment]
        # note that segment 1 is different because we do not need to ensure continuity
        if iSegment == 0:
            first_spline_coeffs = None
            # betas, cost, nEvals = optimise_segment(segment, input_segment, output_segment, support_segment,
            #                                        first_spline_coeffs)
            index_start = 0  # from which timepoint to store the states
        else:
            # find the scaling coeff of the first height by matiching its height at t0 of the segment to the final value of the previous segment
            first_spline_coeff_a = end_of_roi[-1][0] / collocation_segment[0, 0]
            first_spline_coeff_r = end_of_roi[-1][1] / collocation_segment[0, 0]
            first_spline_coeffs = [first_spline_coeff_a, first_spline_coeff_r]
            index_start = 1  # from which timepoint to store the states
        betas, cost, nEvals = optimise_segment(segment, input_segment, output_segment, support_segment,
                                               first_spline_coeffs)
        all_betas.append(betas)
        model_output_fit = simulate_segment(betas, segment, support_segment, first_spline_coeffs=None)
        state_at_sample, state_deriv_at_sample, rhs_at_sample = np.split(model_output_fit, 3, axis=1)
        # save the final value of the segment
        end_of_roi.append(state_at_sample[-1, :])
        for iState, stateName in enumerate(state_names):
            state_of_roi[stateName] += list(state_at_sample[index_start:, iState])
            deriv_of_roi[stateName] += list(state_deriv_at_sample[index_start:, iState])
            rhs_of_roi[stateName] += list(rhs_at_sample[index_start:, iState])
    # end for over segments
    states_of_segments = [v for k, v in state_of_roi.items()]
    deriv_of_segments = [v for k, v in deriv_of_roi.items()]
    rhs_of_segments = [v for k, v in rhs_of_roi.items()]
    # this is problematice because we need to pass the voltage to the function
    current_at_sample = observation_direct_input(np.array(states_of_segments), voltage, Thetas_ODE)
    d_deriv = (np.array(deriv_of_segments) - np.array(rhs_of_segments)) ** 2
    axis_of_time = np.argmax(d_deriv.shape)
    integral_quad = sp.integrate.simpson(y=d_deriv, even='avg', axis=axis_of_time)
    gradient_matching_cost = np.sum(integral_quad, axis=0)
    data_matching_cost = np.sum((current_at_sample - current_true) ** 2) # current_tru is a global variable
    inner_cost_all_segments = data_matching_cost + lambd * gradient_matching_cost
    result = (all_betas, inner_cost_all_segments, data_matching_cost, gradient_matching_cost, states_of_segments)
    return result
########################################################################################################################
# choose optimisation method to plug into scipy optimiser - Make this global for now
# keep it global for now
method_name = 'L-BFGS-B'
# method_name = 'Newton-CG'
# method_name = 'SLSQP'
# method_name = 'TNC'
tolerance = 1e-12
max_iter_inner = 100000
# placeholders for the variables that will be used across functions
nBsplineCoeffs = None
spline_order = None
nSegments = None
state_names = None
fitted_model = None
voltage = None # need a placeholder to generate current in the end of inner optimisation
current_true = None # need a placeholder to generate current in the end of inner optimisation
# main
if __name__ == '__main__':
    debug = True
    # test the pints classes
    # set up variables for the simulation
    tlim = [300, 14899]
    times = np.linspace(*tlim, tlim[-1] - tlim[0], endpoint=False)
    voltage = V(times)
    del tlim
    model_name = 'HH'
    snr_db = 20
    ## set up the parameters for the fitted model
    fitted_model = hh_model
    state_names = ['a', 'r']  # how many states we have in the model that we are fitting
    inLogScale = True  # is the search of thetas in log scale
    lambd = 10e5  # gradient matching weight
    nOutputs = len(state_names) * 3
    ####################################################################################################################
    ### from this point no user changes are required
    ####################################################################################################################
    # load the protocols
    load_protocols
    # generate the segments with B-spline knots and intialise the betas for splines
    jump_indeces, times_roi, voltage_roi, knots_roi, collocation_roi, spline_order = generate_knots(times)
    jumps_odd = jump_indeces[0::2]
    jumps_even = jump_indeces[1::2]
    nSegments = len(jump_indeces[:-1])
    print('Inner optimisation is split into ' + str(nSegments) + ' segments based on protocol steps.')
    state_names = ['a', 'r']  # how many states we have in the model that we are fitting
    nBsplineCoeffs = (len(knots_roi[0]) - spline_order - 1) * len(state_names)
    init_betas_roi = nSegments * [0.5 * np.ones(nBsplineCoeffs)]
    print('Number of B-spline coeffs per segment: ' + str(nBsplineCoeffs) + '.')
    ####################################################################################################################
    # generate synthetic data
    if model_name.lower() not in available_models:
        raise ValueError(f'Unknown model name: {model_name}. Available models are: {available_models}')
    elif model_name.lower() == 'hh':
        thetas_true = thetas_hh_baseline
    elif model_name.lower() == 'kemp':
        thetas_true = thetas_kemp  # the last parameter is the conductance - get it as a separate variable just in case
    Thetas_ODE = thetas_true
    solution, current_model = generate_synthetic_data(model_name, thetas_true, times)
    states_true = solution.sol(times)
    snr = 10 ** (snr_db / 10)
    current_true = current_model(times, solution, thetas_true, snr=snr)
    states_roi, states_known_roi, current_roi = split_generated_data_into_segments(solution, current_true, jump_indeces,
                                                                                   times)
    print('Produced synthetic data for the ' + model_name + ' model based on the pre-loaded voltage protocol.')
    ####################################################################################################################
    thetas_log= np.log(thetas_true)
    # run the log models to make sure they produce the same results as decimal models
    x0= [0,1]
    solution_from_log = sp.integrate.solve_ivp(two_state_model_log, [0, times[-1]], x0, args=[thetas_log],
                                               dense_output=True,method='LSODA', rtol=1e-8, atol=1e-8)
    states_from_log = solution_from_log.sol(times)
    current_from_log = observation_log(times, states_from_log, thetas_log)
    ####################################################################################################################
    ## create figures so you can populate
    fig, axes = plt.subplot_mosaic([['a)'], ['b)'], ['c)']], layout='constrained',sharex=True)
    y_labels = ['I'] + state_names
    for _, ax in axes.items():
        for iSegment, SegmentStart in enumerate(jumps_odd):
            ax.axvspan(times[SegmentStart], times[jumps_even[iSegment]], facecolor='0.2', alpha=0.1)
    axes['a)'].plot(times, current_true, '-k', label=r'Current true',linewidth=1.5, alpha=0.5)
    axes['a)'].plot(times, current_from_log, '-r', label=r'Current from log model', linewidth=1, alpha=0.5)
    axes['b)'].plot(times, states_true[0, :], '-k', label=r'$a$ true', linewidth=1, alpha=0.5)
    axes['c)'].plot(times, states_true[1, :], '-k', label=r'$r$ true', linewidth=1, alpha=0.5)
    axes['b)'].plot(times, states_from_log[0, :], '-k', label=r'$a$ from log model', linewidth=1, alpha=0.5)
    axes['c)'].plot(times, states_from_log[1, :], '-k', label=r'$r$ from log model', linewidth=1, alpha=0.5)
    fig1, axes1 = plt.subplot_mosaic([['a)', 'a)'], ['b)', 'c)'], ['d)', 'e)']], layout='constrained')
    y_labels1 = ['I_{true} - I_{model}', 'da(C) - RHS(C)', 'dr(C) - RHS(C)',
                'a - Phi C_a', 'r - Phi C_r']
    for _, ax in axes1.items():
        for iSegment, SegmentStart in enumerate(jumps_odd):
            ax.axvspan(times[SegmentStart], times[jumps_even[iSegment]], facecolor='0.2', alpha=0.1)
    ####################################################################################################################
    for ilog in range(5,6): # loop over different orders of magnitude of lambda
        lambd = 10**ilog # set the value of lambda
        # set placeholders for betas and other variables we wish to store over segments
        all_betas = [] # store the bspline coefficients
        all_costs = [] # store the inner costs
        end_of_roi = [] # store the final values of the states to ensure continuity
        # states, derivatives and RHSs
        state_of_roi = {key: [] for key in state_names}
        rhs_of_roi = {key: [] for key in state_names}
        deriv_of_roi = {key: [] for key in state_names}
        totalEvaluations = []
        big_tic = tm.time() # time keeping
        # loop over segments
        for iSegment in range(nSegments):
            tic = tm.time()
            segment = times_roi[iSegment]
            input_segment = voltage_roi[iSegment]
            output_segment = current_roi[iSegment]
            support_segment = knots_roi[iSegment]
            collocation_segment = collocation_roi[iSegment]
            # note that segment 1 is different because we do not need to ensure continuity
            if iSegment == 0:
                first_spline_coeffs = None
                betas, cost, nEvals = optimise_segment(segment,input_segment,output_segment,support_segment,first_spline_coeffs)
                index_start = 0 # from which timepoint to store the states
            else:
                # find the scaling coeff of the first height by matiching its height at t0 of the segment to the final value of the previous segment
                first_spline_coeff_a = end_of_roi[-1][0] / collocation_segment[0, 0]
                first_spline_coeff_r = end_of_roi[-1][1] / collocation_segment[0, 0]
                first_spline_coeffs = [first_spline_coeff_a,first_spline_coeff_r]
                betas, cost, nEvals = optimise_segment(segment, input_segment, output_segment, support_segment,first_spline_coeffs)
                index_start = 1 # from which timepoint to store the states
            totalEvaluations.append(nEvals)
            all_betas.append(betas)
            all_costs.append(cost)
            toc = tm.time()
            print(
                'Iteration ' + str(iSegment) + ' is finished after '+ str(nEvals) +
                ' evaluations. Final cost value: '+ str(cost) +'. Elapsed time: ' + str(toc-tic) + 's.')
            # check collocation solution against truth
            model_output_fit = simulate_segment(betas, segment, support_segment, first_spline_coeffs=None)
            state_at_truth, state_deriv_at_truth, rhs_truth = np.split(model_output_fit, 3, axis=1)
            current_model_at_truth = observation_direct_input(state_at_truth.transpose(), input_segment, Thetas_ODE)
            # save the final value of the segment
            end_of_roi.append(state_at_truth[-1,:])
            for iState, stateName in enumerate(state_names):
                state_of_roi[stateName] += list(state_at_truth[index_start:, iState])
                deriv_of_roi[stateName] += list(state_deriv_at_truth[index_start:, iState])
                rhs_of_roi[stateName] += list(rhs_truth[index_start:, iState])
        #### end of loop
        ################################################################################################################
        big_toc = tm.time()
        total_time = big_toc - big_tic
        print('Total evaluations: ' + str(sum(totalEvaluations)) + '. Total runtime: ' + str(total_time) + ' s.')
        times_of_segments = np.hstack(times_roi[:iSegment + 1])
        states_of_segments = [v for k, v in state_of_roi.items()]
        # plot sfuff in axes
        current_modelled = observation(times, np.array(states_of_segments), thetas_true)
        axes['a)'].plot(times, current_modelled, '--',
                        label=r'$\lambda$ = ' + str(int(lambd)) + ',  runtime: {0:.3g}'.format(
                            total_time) + ' s.', linewidth=1, alpha=0.7)
        axes['a)'].set_xlim(times_of_segments[0], times_of_segments[-1])
        axes['b)'].plot(times, state_of_roi[state_names[0]], '--', label=r'$\lambda$ = ' + str(int(lambd)),
                        linewidth=1, alpha=0.7)
        axes['c)'].plot(times, state_of_roi[state_names[1]], '--', label=r'$\lambda$ = ' + str(int(lambd)),
                        linewidth=1, alpha=0.7)
        # plot errors in axes1
        axes1['a)'].plot(times, current_modelled - current_true, '--', label=r'$\lambda$ = ' + str(int(lambd)),
                         linewidth=1, alpha=0.7)
        axes1['b)'].plot(times, np.array(rhs_of_roi[state_names[0]]) - np.array(deriv_of_roi[state_names[0]]),
                         '--', label=r'$\lambda$ = ' + str(int(lambd)), linewidth=1, alpha=0.7)
        axes1['c)'].plot(times, np.array(rhs_of_roi[state_names[1]]) - np.array(deriv_of_roi[state_names[1]]),
                         '--', label=r'$\lambda$ = ' + str(int(lambd)), linewidth=1, alpha=0.7)
        axes1['d)'].plot(times, np.array(state_of_roi[state_names[0]]) - states_true[0, :], '--',
                         label=r'$\lambda$ = ' + str(int(lambd)), linewidth=1, alpha=0.7)
        axes1['e)'].plot(times, np.array(state_of_roi[state_names[1]]) - states_true[1, :], '--',
                         label=r'$\lambda$ = ' + str(int(lambd)), linewidth=1, alpha=0.7)
    ## end loop over lambdas

    ####
    # test using inner function
    weight = 10e4
    tic = tm.time()
    test_output = inner_optimisation_test(Thetas_ODE, weight, times_roi, voltage_roi, current_roi, knots_roi, collocation_roi)
    toc = tm.time()
    total_time = toc - tic
    print('Elapsed time running as function: ' + str(total_time) + ' s.')
    betas_sample, inner_costs_sample, state_fitted_at_sample = test_output
    InnerCost = sum(inner_costs_sample)
    state_all_segments = np.array(state_fitted_at_sample)
    current_all_segments = observation_direct_input(state_all_segments, V(times), Thetas_ODE)
    # get the derivative and the RHS
    rhs_of_roi = {key: [] for key in state_names}
    deriv_of_roi = {key: [] for key in state_names}
    for iSegment in range(nSegments):
        # segment = times_roi[iSegment]
        # support_segment = knots_roi[iSegment]
        # betas = betas_sample[iSegment]
        model_output_fit = simulate_segment(betas_sample[iSegment], times_roi[iSegment], knots_roi[iSegment], first_spline_coeffs=None)
        state_at_sample, state_deriv_at_sample, rhs_at_sample = np.split(model_output_fit, 3, axis=1)
        if iSegment == 0:
            index_start = 0  # from which timepoint to store the states
        else:
            index_start = 1  # from which timepoint to store the states
        for iState, stateName in enumerate(state_names):
            deriv_of_roi[stateName] += list(state_deriv_at_sample[index_start:, iState])
            rhs_of_roi[stateName] += list(rhs_at_sample[index_start:, iState])
    ## end of loop over segments


    # add test to plots
    axes['a)'].plot(times, current_modelled, ':m',
                    label=r'$\lambda$ = ' + str(int(weight)), linewidth=1, alpha=0.7)
    axes['b)'].plot(times, state_all_segments[0,:], ':m', label=r'$\lambda$ = ' + str(int(weight)),
                    linewidth=1, alpha=0.7)
    axes['c)'].plot(times, state_all_segments[1,:], ':m', label=r'$\lambda$ = ' + str(int(weight)),
                    linewidth=1, alpha=0.7)
    axes1['a)'].plot(times, current_all_segments - current_true, '--', label=r'$\lambda$ = ' + str(int(lambd)),
                     linewidth=1, alpha=0.7)
    axes1['b)'].plot(times, np.array(rhs_of_roi[state_names[0]]) - np.array(deriv_of_roi[state_names[0]]),
                     ':m', label=r'$\lambda$ = ' + str(int(weight)), linewidth=1, alpha=0.7)
    axes1['c)'].plot(times, np.array(rhs_of_roi[state_names[1]]) - np.array(deriv_of_roi[state_names[1]]),
                     ':m', label=r'$\lambda$ = ' + str(int(weight)), linewidth=1, alpha=0.7)
    axes1['d)'].plot(times, state_all_segments[0,:] - states_true[0, :], ':m',
                     label=r'$\lambda$ = ' + str(int(weight)), linewidth=1, alpha=0.7)
    axes1['e)'].plot(times, state_all_segments[1,:] - states_true[1, :], ':m',
                     label=r'$\lambda$ = ' + str(int(weight)), linewidth=1, alpha=0.7)
    ####################################################################################################################
    ## save the figures
    iAx = 0
    for _, ax in axes.items():
        ax.set_ylabel(y_labels[iAx], fontsize=12)
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
        ax.legend(fontsize=12, loc='best')
        iAx += 1
    # plt.tight_layout(pad=0.3)
    fig.savefig('Figures/states_'+method_name+'_all_lambdas.png', dpi=400)
    iAx = 0
    for _, ax in axes1.items():
        ax.set_ylabel(y_labels1[iAx], fontsize=12)
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
        ax.legend(fontsize=12, loc='best')
        iAx += 1
    # plt.tight_layout(pad=0.3)
    fig1.savefig('Figures/errors_'+method_name+'_all_lambdas.png', dpi=400)
    ####################################################################################################################
    # test plotting semgents and ODE output
    ## simulate the model using the best thetas and the ODE model used
    solution_optimised = sp.integrate.solve_ivp(fitted_model, [0, times[-1]], x0, args=[Thetas_ODE],
                                                dense_output=True, method='LSODA',rtol=1e-8, atol=1e-8)
    states_optimised_ODE = solution_optimised.sol(times)
    current_optimised_ODE = observation_direct_input(states_optimised_ODE, voltage, Thetas_ODE)
    ## create figures so you can populate them with the data
    fig, axes = plt.subplot_mosaic([['a)'], ['b)'], ['c)']], layout='constrained', sharex=True)
    y_labels = ['I'] + state_names
    for _, ax in axes.items():
        for iSegment, SegmentStart in enumerate(jumps_odd):
            ax.axvspan(times[SegmentStart], times[jumps_even[iSegment]], facecolor='0.2', alpha=0.1)
    # axes['a)'].plot(times, current_true, '-k', label=r'Current true', linewidth=1.5, alpha=0.5)
    fig1, axes1 = plt.subplot_mosaic([['a)', 'a)'], ['b)', 'c)'], ['d)', 'e)']], layout='constrained')
    y_labels1 = ['I_{true} - I_{model}', 'da(C) - RHS(C)', 'dr(C) - RHS(C)',
                 'a - Phi C_a', 'r - Phi C_r']
    for _, ax in axes1.items():
        for iSegment, SegmentStart in enumerate(jumps_odd):
            ax.axvspan(times[SegmentStart], times[jumps_even[iSegment]], facecolor='0.2', alpha=0.1)
    # add test to plots - compare the modelled current with the true current
    axes['a)'].plot(times, current_all_segments, '-c',
                    label=r'Current from B-spline approx.', linewidth=1, alpha=0.7)
    axes['a)'].plot(times, current_optimised_ODE, ':m',
                    label=r'Current from ODE solution', linewidth=1, alpha=0.7)
    # axes['a)'].set_xlim(times_of_segments[0], times_of_segments[-1])
    axes['a)'].set_xlim(1890,1920)
    axes['b)'].plot(times, state_all_segments[0, :], '-c', label=r'B-spline approx. at $\lambda$ = ' + str(int(weight)),
                    linewidth=1, alpha=0.7)
    axes['c)'].plot(times, state_all_segments[1, :], '-c', label=r'B-spline approx. at $\lambda$ = ' + str(int(weight)),
                    linewidth=1, alpha=0.7)
    axes['b)'].plot(times, states_optimised_ODE[0, :], ':m', label=r'Fitted ODE solution at $\lambda$ = ' + str(int(weight)),
                    linewidth=1, alpha=0.7)
    axes['c)'].plot(times, states_optimised_ODE[1, :], ':m', label=r'Fitted ODE solution at $\lambda$ = ' + str(int(weight)),
                    linewidth=1, alpha=0.7)
    # axes1['a)'].plot(times, current_all_segments - current_true, '--c', label=r'Current from B-spline approx.',
    #                  linewidth=1, alpha=0.7)
    # axes1['a)'].plot(times, current_optimised_ODE - current_true, ':m',
    #                  label=r'Current from ODE solution',linewidth=1, alpha=0.7)
    axes1['a)'].plot(times, current_all_segments - current_optimised_ODE, '--k', label=r'B-spline approx. - ODE solution',
                     linewidth=1, alpha=0.7)
    axes1['b)'].plot(times, np.array(rhs_of_roi[state_names[0]]) - np.array(deriv_of_roi[state_names[0]]),
                     '--k', label=r'Gradient matching error at $\lambda$ = ' + str(int(weight)), linewidth=1, alpha=0.7)
    axes1['c)'].plot(times, np.array(rhs_of_roi[state_names[1]]) - np.array(deriv_of_roi[state_names[1]]),
                     '--k', label=r'Gradient matching at $\lambda$ = ' + str(int(weight)), linewidth=1, alpha=0.7)
    axes1['d)'].plot(times, state_all_segments[0, :] - states_optimised_ODE[0, :], '--k',
                     label=r'B-spline approx. error at $\lambda$ = ' + str(int(weight)), linewidth=1, alpha=0.7)
    axes1['e)'].plot(times, state_all_segments[1, :] - states_optimised_ODE[1, :], '--k',
                     label=r'B-spline approx. error at $\lambda$ = ' + str(int(weight)), linewidth=1, alpha=0.7)
    ## save the figures
    iAx = 0
    for _, ax in axes.items():
        ax.set_ylabel(y_labels[iAx], fontsize=12)
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
        ax.legend(fontsize=12, loc='best')
        iAx += 1
    # plt.tight_layout(pad=0.3)
    fig.savefig('Figures/states_model_output_test.png', dpi=400)
    iAx = 0
    for _, ax in axes1.items():
        ax.set_ylabel(y_labels1[iAx], fontsize=12)
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
        ax.legend(fontsize=12, loc='best')
        iAx += 1
    # plt.tight_layout(pad=0.3)
    fig1.savefig('Figures/errors_model_output_test.png', dpi=400)


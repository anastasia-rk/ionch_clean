# imports
from pints_classes import *
import time as tm


# definitions
def ion_channel_model(t, x, theta):
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

def observation(t, x, theta):
    # I
    a, r = x[:2]
    *ps, g = theta[:9]
    return g * a * r * (V(t) - EK)

def optimise_first_segment(roi,input_roi,output_roi,support_roi):
    class bsplineOutput(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            # given times and return the simulated values
            coeffs_a, coeffs_r = np.split(parameters, 2)
            tck_a = (support_roi, coeffs_a, spline_order)
            tck_r = (support_roi, coeffs_r, spline_order)
            dot_a = sp.interpolate.splev(times, tck_a, der=1)
            dot_r = sp.interpolate.splev(times, tck_r, der=1)
            fun_a = sp.interpolate.splev(times, tck_a, der=0)
            fun_r = sp.interpolate.splev(times, tck_r, der=0)
            # the RHS must be put into an array
            dadr = ion_channel_model(times, [fun_a, fun_r], Thetas_ODE)
            rhs_theta = np.array(dadr)
            spline_surface = np.array([fun_a, fun_r])
            spline_deriv = np.array([dot_a, dot_r])
            # pack all required variables into the same array - will be the wrong orientation from pints preferred nTimes x nOutputs
            packed_output = np.concatenate((spline_surface,spline_deriv,rhs_theta),axis=0)
            return np.transpose(packed_output)

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs

        def n_outputs(self):
            # Return the dimension of the output vector
            return nOutputs
    # define a class that outputs only b-spline surface features
    model_bsplines = bsplineOutput()
    init_betas = 0.5 * np.ones(nBsplineCoeffs)  # initial values of B-spline coefficients
    sigma0_betas = 0.2 * np.ones(nBsplineCoeffs)
    values_to_match_output_dims = np.transpose( np.array([output_roi, input_roi, output_roi, input_roi, output_roi, input_roi]))
    problem_inner = pints.MultiOutputProblem(model=model_bsplines, times=roi, values=values_to_match_output_dims)
    error_inner = InnerCriterion(problem=problem_inner)
    boundaries_betas = pints.RectangularBoundaries(np.zeros_like(init_betas), 0.99 * np.ones_like(init_betas))
    optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, sigma0=sigma0_betas,
                                                   boundaries=boundaries_betas, method=pints.CMAES)
    optimiser_inner.set_max_iterations(100000)
    # optimiser_inner.method().set_population_size(min(5, len(init_betas)/2))
    optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-6)
    optimiser_inner.set_parallel(False)
    optimiser_inner.set_log_to_screen(False)
    betas_roi, cost_roi = optimiser_inner.run()
    nEvaluations = optimiser_inner._evaluations
    return betas_roi, cost_roi, nEvaluations

def optimise_segment(roi,input_roi,output_roi,support_roi):
    class bsplineOutputSegment(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            # given times and return the simulated values
            coeffs_a, coeffs_r = np.split(parameters, 2)
            coeffs_a = np.insert(coeffs_a, 0, first_spline_coeff_a)
            coeffs_r = np.insert(coeffs_r, 0, first_spline_coeff_r)
            tck_a = (support_roi, coeffs_a, spline_order)
            tck_r = (support_roi, coeffs_r, spline_order)
            dot_a = sp.interpolate.splev(times, tck_a, der=1)
            dot_r = sp.interpolate.splev(times, tck_r, der=1)
            fun_a = sp.interpolate.splev(times, tck_a, der=0)
            fun_r = sp.interpolate.splev(times, tck_r, der=0)
            # the RHS must be put into an array
            dadr = ion_channel_model(times, [fun_a, fun_r], Thetas_ODE)
            rhs_theta = np.array(dadr)
            spline_surface = np.array([fun_a, fun_r])
            spline_deriv = np.array([dot_a, dot_r])
            # pack all required variables into the same array - will be the wrong orientation from pints preferred nTimes x nOutputs
            packed_output = np.concatenate((spline_surface,spline_deriv,rhs_theta),axis=0)
            return np.transpose(packed_output)

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs-2

        def n_outputs(self):
            # Return the dimension of the output vector
            return nOutputs
    # define a class that outputs only b-spline surface features
    model_bsplines = bsplineOutputSegment()
    init_betas = 0.5 * np.ones(nBsplineCoeffs-2)  # initial values of B-spline coefficients
    sigma0_betas = 0.2 * np.ones(nBsplineCoeffs-2)
    values_to_match_output_dims = np.transpose( np.array([output_roi, input_roi, output_roi, input_roi, output_roi, input_roi]))
    problem_inner = pints.MultiOutputProblem(model=model_bsplines, times=roi, values=values_to_match_output_dims)
    error_inner = InnerCriterion(problem=problem_inner)
    boundaries_betas = pints.RectangularBoundaries(np.zeros_like(init_betas), 0.99 * np.ones_like(init_betas))
    optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, sigma0=sigma0_betas,
                                                   boundaries=boundaries_betas, method=pints.CMAES)
    optimiser_inner.set_max_iterations(100000)
    # optimiser_inner.method().set_population_size(min(5, len(init_betas)/2))
    optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-6)
    optimiser_inner.set_parallel(False)
    optimiser_inner.set_log_to_screen(False)
    betas_roi, cost_roi = optimiser_inner.run()
    nEvaluations = optimiser_inner._evaluations
    coeffs_a, coeffs_r = np.split(betas_roi, 2)
    coeffs_a = np.insert(coeffs_a,0,first_spline_coeff_a)
    coeffs_r = np.insert(coeffs_r,0,first_spline_coeff_r)
    betas_roi_with_first_coeff = np.concatenate((coeffs_a,coeffs_r))
    return betas_roi_with_first_coeff, cost_roi, nEvaluations

# main
if __name__ == '__main__':
    # test the pints classes
    # set up variables for the simulation
    tlim = [300, 14899]
    times = np.linspace(*tlim, tlim[-1] - tlim[0], endpoint=False)
    del tlim
    model_name = 'HH'
    snr_db = 20
    state_names = ['a', 'r']
    inLogScale = True
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
        thetas_true = thetas_kemp
    g = thetas_true[-1]  # the last parameter is the conductance - get it as a separate variable just in case
    Thetas_ODE = thetas_true
    solution, current_model = generate_synthetic_data(model_name, thetas_true, times)
    states_true = solution.sol(times)
    snr = 10 ** (snr_db / 10)
    current_true = current_model(times, solution, thetas_true, snr=snr)
    states_roi, states_known_roi, current_roi = split_generated_data_into_segments(solution, current_true, jump_indeces,
                                                                                   times)
    print('Produced synthetic data for the ' + model_name + ' model based on the pre-loaded voltage protocol.')
    ####################################################################################################################
    class bsplineOutputTest(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, support, times):
            # given times and return the simulated values
            coeffs_a, coeffs_r = np.split(parameters, 2)
            tck_a = (support, coeffs_a, spline_order)
            tck_r = (support, coeffs_r, spline_order)
            dot_a = sp.interpolate.splev(times, tck_a, der=1)
            dot_r = sp.interpolate.splev(times, tck_r, der=1)
            fun_a = sp.interpolate.splev(times, tck_a, der=0)
            fun_r = sp.interpolate.splev(times, tck_r, der=0)
            # the RHS must be put into an array
            dadr = ion_channel_model(times, [fun_a, fun_r], Thetas_ODE)
            rhs_theta = np.array(dadr)
            spline_surface = np.array([fun_a, fun_r])
            spline_deriv = np.array([dot_a, dot_r])
            # pack all required variables into the same array - will be the wrong orientation from pints preferred nTimes x nOutputs
            packed_output = np.concatenate((spline_surface,spline_deriv,rhs_theta),axis=0)
            return np.transpose(packed_output)

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs

        def n_outputs(self):
            # Return the dimension of the output vector
            return nOutputs
    #############################################################
    # define an error w.r.t B-spline parameters that assumes that it knows ODE parameters
    class InnerCriterion(pints.ProblemErrorMeasure):
        # do I need to redefine custom init or can just drop this part?
        def __init__(self, problem, weights=None):
            super(InnerCriterion, self).__init__(problem)
            if weights is None:
                weights = [1] * self._n_outputs
            elif self._n_outputs != len(weights):
                raise ValueError(
                    'Number of weights must match number of problem outputs.')
            # Check weights
            self._weights = np.asarray([float(w) for w in weights])
        # this function is the function of beta - bspline parameters
        def __call__(self, betas):
            # evaluate the integral at the value of B-spline coefficients
            model_output = self._problem.evaluate(betas)   # the output of the model with be an array of size nTimes x nOutputs
            x, x_dot, rhs = np.split(model_output, 3, axis=1) # we split the array into states, state derivs, and RHSs
            # compute the data fit
            *ps, g = Thetas_ODE[:9]
            volts_for_model = self._values[:,1] # we need to make sure that voltage is read at the times within ROI so we pass it in as part of values
            d_y = g * x[:, 0] * x[:, 1] * (volts_for_model - EK) - self._values[:,0]
            data_fit_cost = np.transpose(d_y) @ d_y
            # compute the gradient matching cost
            d_deriv = (x_dot - rhs) ** 2
            integral_quad = sp.integrate.simpson(y=d_deriv, even='avg',axis=0)
            gradient_match_cost = np.sum(integral_quad, axis=0)
            # not the most elegant implementation because it just grabs global lambda
            return data_fit_cost + lambd * gradient_match_cost
    # ####################################################################################################################
    ## create figures so you can populate
    fig, axes = plt.subplot_mosaic([['a)'], ['b)'], ['c)']], layout='constrained',sharex=True)
    y_labels = ['I', '$a$', '$r$']
    for _, ax in axes.items():
        for iSegment, SegmentStart in enumerate(jumps_odd):
            ax.axvspan(times[SegmentStart], times[jumps_even[iSegment]], facecolor='0.2', alpha=0.2)
    axes['a)'].plot(times, current_true, '-k', label=r'Current true',linewidth=2, alpha=0.7)
    axes['b)'].plot(times, solution.sol(times)[0, :], '-k', label=r'$a$ true', linewidth=2, alpha=0.7)
    axes['c)'].plot(times, solution.sol(times)[1, :], '-k', label=r'$r$ true', linewidth=2, alpha=0.7)
    fig1, axes1 = plt.subplot_mosaic([['a)', 'a)'], ['b)', 'c)'], ['d)', 'e)']], layout='constrained')
    y_labels1 = ['I_{true} - I_{model}', 'da(C) - RHS(C)', 'dr(C) - RHS(C)',
                'a - Phi C_a', 'r - Phi C_r']
    for _, ax in axes1.items():
        for iSegment, SegmentStart in enumerate(jumps_odd):
            ax.axvspan(times[SegmentStart], times[jumps_even[iSegment]], facecolor='0.2', alpha=0.2)
    ####################################################################################################################
    for ilog in range(0,6): # loop over different orders of magnitude of lambda
        lambd = 10**ilog # set the value of lambda
        # try optimising several segments
        all_betas = []
        all_costs = []
        *ps, g = thetas_true
        model_bsplines = bsplineOutputTest()
        end_of_roi = []
        state_of_roi = {key: [] for key in state_names}
        rhs_of_roi = {key: [] for key in state_names}
        deriv_of_roi = {key: [] for key in state_names}
        totalEvaluations = []
        big_tic = tm.time()
        for iSegment in range(1):
            tic = tm.time()
            segment = times_roi[iSegment]
            input_segment = voltage_roi[iSegment]
            output_segment = current_roi[iSegment]
            support_segment = knots_roi[iSegment]
            betas, cost, nEvals = optimise_first_segment(segment,input_segment,output_segment,support_segment)
            totalEvaluations.append(nEvals)
            all_betas.append(betas)
            all_costs.append(cost)
            toc = tm.time()
            print('Iteration ' + str(iSegment) + ' is finished after '+ str(nEvals) +' evaluations. Final cost value: '+ str(cost) +'. Elapsed time: ' + str(toc-tic) + 's.')
            # check collocation solution against truth
            model_output_fit_at_truth = model_bsplines.simulate(betas,knots_roi[iSegment], times_roi[iSegment])
            state_at_truth, state_deriv_at_truth, rhs_truth = np.split(model_output_fit_at_truth, 3, axis=1)
            current_model_at_truth = g * state_at_truth[:, 0] * state_at_truth[:, 1] * (voltage_roi[iSegment] - EK)
            # save the final value of the segment
            end_of_roi.append(state_at_truth[-1,:])
            for iState, stateName in enumerate(state_names):
                state_of_roi[stateName] += list(state_at_truth[:, iState])
                deriv_of_roi[stateName] += list(state_deriv_at_truth[:, iState])
                rhs_of_roi[stateName] += list(rhs_truth[:, iState])
        ####################################################################################################################
        #  optimise the following segments by matching the first B-spline height to the previous segment
        for iSegment in range(1,nSegments):
            tic = tm.time()
            segment = times_roi[iSegment]
            input_segment = voltage_roi[iSegment]
            output_segment = current_roi[iSegment]
            support_segment = knots_roi[iSegment]
            collocation_segment = collocation_roi[iSegment]
            # find the scaling coeff of the first height by matiching its height at t0 of the segment to the final value of the previous segment
            first_spline_coeff_a = end_of_roi[-1][0] / collocation_segment[0, 0]
            first_spline_coeff_r = end_of_roi[-1][1] / collocation_segment[0, 0]
            betas, cost, nEvals = optimise_segment(segment,input_segment,output_segment,support_segment)
            totalEvaluations.append(nEvals)
            all_betas.append(betas)
            all_costs.append(cost)
            toc = tm.time()
            print('Iteration ' + str(iSegment) + ' is finished after '+ str(nEvals) +' evaluations. Final cost value: '+ str(cost) +'. Elapsed time: ' + str(toc-tic) + 's.')
            # check collocation solution against truth
            model_bsplines = bsplineOutputTest()
            model_output_fit_at_truth = model_bsplines.simulate(betas,knots_roi[iSegment], times_roi[iSegment])
            state_at_truth, state_deriv_at_truth, rhs_truth = np.split(model_output_fit_at_truth, 3, axis=1)
            current_model_at_truth = g * state_at_truth[:, 0] * state_at_truth[:, 1] * (voltage_roi[iSegment] - EK)
            # store end of segment and the whole state for the
            end_of_roi.append(state_at_truth[-1, :])
            for iState, stateName in enumerate(state_names):
                state_of_roi[stateName] += list(state_at_truth[1:, iState])
                deriv_of_roi[stateName] += list(state_deriv_at_truth[1:, iState])
                rhs_of_roi[stateName] += list(rhs_truth[1:, iState])
        #### end of loop
        ################################################################################################################
        big_toc = tm.time()
        total_time = big_toc - big_tic
        print('Total evaluations: ' + str(sum(totalEvaluations)) + '. Total runtime: ' + str(total_time) + ' s.' )
        times_of_segments = np.hstack(times_roi[:iSegment+1])
        states_of_segments = [v for k, v in state_of_roi.items()]
        # plot sfuff in axes
        current_modelled = observation(times, np.array(states_of_segments), thetas_true)
        axes['a)'].plot(times, current_modelled, '--', label=r'$\lambda$ = '+str(int(lambd)) + ',  runtime: {0:.3g}'.format(total_time) + ' s.', linewidth=1, alpha=0.7)
        axes['a)'].set_xlim(times_of_segments[0],times_of_segments[-1])
        axes['b)'].plot(times, state_of_roi[state_names[0]], '--', label=r'$\lambda$ = '+str(int(lambd)), linewidth=1, alpha=0.7)
        axes['c)'].plot(times, state_of_roi[state_names[1]], '--', label=r'$\lambda$ = '+str(int(lambd)), linewidth=1, alpha=0.7)
        # plot errors in axes1
        axes1['a)'].plot(times, current_modelled - current_true, '--', label=r'$\lambda$ = '+str(int(lambd)), linewidth=1, alpha=0.7)
        axes1['b)'].plot(times, np.array(rhs_of_roi[state_names[0]]) - np.array(deriv_of_roi[state_names[0]]), '--',label=r'$\lambda$ = '+str(int(lambd)), linewidth=1, alpha=0.7)
        axes1['c)'].plot(times, np.array(rhs_of_roi[state_names[1]]) - np.array(deriv_of_roi[state_names[1]]), '--', label=r'$\lambda$ = '+str(int(lambd)), linewidth=1, alpha=0.7)
        axes1['d)'].plot(times, np.array(state_of_roi[state_names[0]]) - states_true[0,:], '--', label=r'$\lambda$ = '+str(int(lambd)), linewidth=1, alpha=0.7)
        axes1['e)'].plot(times, np.array(state_of_roi[state_names[1]]) - states_true[1,:], '--', label=r'$\lambda$ = '+str(int(lambd)), linewidth=1, alpha=0.7)
    ## end loop over lambdas
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
    fig.savefig('Figures/states_pints_all_lambdas.png', dpi=400)
    iAx = 0
    for _, ax in axes1.items():
        ax.set_ylabel(y_labels1[iAx], fontsize=12)
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
        ax.legend(fontsize=12, loc='best')
        iAx += 1
    # plt.tight_layout(pad=0.3)
    fig1.savefig('Figures/errors_pints_all_lambdas.png', dpi=400)

# imports
import numpy as np

from setup import *
import multiprocessing as mp
from itertools import repeat
import traceback
matplotlib.use('AGG')
plt.ioff()

# definitions
def hh_model(t, x, theta):
    a, r = x[:2]
    *p, g = theta[:9]
    v = V(t)
    k1 = p[0] * np.exp(p[1] * v)
    k2 = p[2] * np.exp(-p[3] * v)
    k3 = p[4] * np.exp(p[5] * v)
    k4 = p[6] * np.exp(-p[7] * v)
    a_inf = k1 / (k1 + k2)
    tau_a = 1 / (k1 + k2)
    r_inf = k4 / (k3 + k4)
    tau_r = 1 / (k3 + k4)
    da = (a_inf - a) / tau_a
    dr = (r_inf - r) / tau_r
    return [da,dr]

def observation(t, x, theta):
    # I
    a, r = x[:2]
    *ps, g = theta[:9]
    return g * a * r * (V(t) - EK)
# get Voltage for time in ms
def V(t):
    return volts_intepolated((t)/ 1000)

### Only consider a -- all params in log scale
def ode_a_only(t, x, theta):
    # call the model with a smaller number of unknown parameters and one state known
    a = x
    v = V(t)
    k1 =  np.exp(theta[0] + np.exp(theta[1]) * v)
    k2 =  np.exp(theta[2] -np.exp(theta[3]) * v)
    a_inf = k1 / (k1 + k2)
    tau_a = 1 / (k1 + k2)
    da = (a_inf - a) / tau_a
    return da

### Only  consider r -- log space on a parameters
def ode_r_only(t, x, theta):
    # call the model with a smaller number of unknown parameters and one state known
    r = x
    v = V(t)
    k3 =  np.exp(theta[0] + np.exp(theta[1]) * v)
    k4 =  np.exp(theta[2] - np.exp(theta[3]) * v)
    r_inf = k4 / (k3 + k4)
    tau_r = 1 / (k3 + k4)
    dr = (r_inf - r) / tau_r
    return dr

def optimise_first_segment(roi,input_roi,output_roi,support_roi,state_known_roi,init_betas, sigma0_betas):
    nOutputs = 3
    # define a class that outputs only b-spline surface features
    class bsplineOutputSegment(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            # given times and return the simulated values
            tck = (support_roi, parameters, degree)
            dot_ = sp.interpolate.splev(times, tck, der=1)
            fun_ = sp.interpolate.splev(times, tck, der=0)
            # the RHS must be put into an array
            rhs = ion_channel_model_one_state(times, fun_, Thetas_ODE)
            return np.array([fun_, dot_, rhs]).T

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs

        def n_outputs(self):
            # Return the dimension of the output vector
            return nOutputs
    # define a class that outputs only b-spline surface features
    model_bsplines = bsplineOutputSegment()
    values_to_match_output_dims = np.transpose(np.array([output_roi, input_roi, state_known_roi]))
    problem_inner = pints.MultiOutputProblem(model=model_bsplines, times=roi, values=values_to_match_output_dims)
    error_inner = InnerCriterion(problem=problem_inner)
    boundaries_betas = pints.RectangularBoundaries(np.zeros_like(init_betas), upper_bound_beta * np.ones_like(init_betas))
    optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, sigma0=sigma0_betas,
                                                   boundaries=boundaries_betas, method=pints.CMAES)
    optimiser_inner.set_max_iterations(60000)
    optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-10)
    optimiser_inner.set_parallel(False)
    optimiser_inner.set_log_to_screen(False)
    betas_roi, cost_roi = optimiser_inner.run()
    nEvaluations = optimiser_inner._evaluations
    return betas_roi, cost_roi, nEvaluations

def optimise_segment(roi,input_roi,output_roi,support_roi,state_known_roi,init_betas, sigma0_betas, first_spline_coeff):
    nOutputs = 3
    # define a class that outputs only b-spline surface features
    class bsplineOutputSegment(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            parameters_with_first = np.insert(parameters, 0, first_spline_coeff)
            # given times and return the simulated values
            tck = (support_roi, parameters_with_first, degree)
            dot_ = sp.interpolate.splev(times, tck, der=1)
            fun_ = sp.interpolate.splev(times, tck, der=0)
            # the RHS must be put into an array
            rhs = ion_channel_model_one_state(times, fun_, Thetas_ODE)
            return np.array([fun_, dot_, rhs]).T

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs-1

        def n_outputs(self):
            # Return the dimension of the output vector
            return nOutputs
    # define a class that outputs only b-spline surface features
    model_bsplines = bsplineOutputSegment()
    values_to_match_output_dims = np.transpose(np.array([output_roi, input_roi, state_known_roi]))
    problem_inner = pints.MultiOutputProblem(model=model_bsplines, times=roi, values=values_to_match_output_dims)
    error_inner = InnerCriterion(problem=problem_inner)
    boundaries_betas = pints.RectangularBoundaries(np.zeros_like(init_betas), upper_bound_beta * np.ones_like(init_betas))
    optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, sigma0=sigma0_betas,
                                                   boundaries=boundaries_betas, method=pints.CMAES)
    optimiser_inner.set_max_iterations(60000)
    # optimiser_inner.method().set_population_size(min(5, len(init_betas)/2))
    optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-10)
    optimiser_inner.set_parallel(False)
    optimiser_inner.set_log_to_screen(False)
    betas_roi, cost_roi = optimiser_inner.run()
    nEvaluations = optimiser_inner._evaluations
    coeffs_with_first = np.insert(betas_roi,0,first_spline_coeff)
    return coeffs_with_first, cost_roi, nEvaluations

# define inner optimisation as a function to parallelise the CMA-ES
def inner_optimisation(theta, times_roi, voltage_roi, current_roi, knots_roi, states_known_roi, init_betas_roi):
    # assign the variable that is readable in the class of B-spline evaluation
    global Thetas_ODE # declrae the global variable to be used in classess across all functions
    Thetas_ODE = theta.copy()
    # fit the b-spline surface given the sampled value of the ODE parameter vector
    betas_sample = []
    inner_costs_sample = []
    end_of_roi = []
    state_fitted_roi = {key: [] for key in hidden_state_names}
    for iSegment in range(1):
        segment = times_roi[iSegment]
        input_segment = voltage_roi[iSegment]
        output_segment = current_roi[iSegment]
        knots = knots_roi[iSegment]
        state_known_segment = states_known_roi[iSegment]
        # initialise inner optimisation
        init_betas = init_betas_roi[iSegment]
        sigma0_betas = 0.2 * np.ones(nBsplineCoeffs)
        try:
            betas_segment, inner_cost_segment, evals_segment = optimise_first_segment(segment,
                                                                                      input_segment,
                                                                                      output_segment,
                                                                                      knots,
                                                                                      state_known_segment, init_betas, sigma0_betas)
        except Exception:
            traceback.print_exc()
            print('Error encountered during optimisation.')
            optimisationFailed = True
            return (np.NaN,np.NaN,np.NaN) # return dummy values
        else:
            # check collocation solution against truth
            model_output = model_bsplines_test.simulate(betas_segment, knots, segment)
            state_at_estimate, deriv_at_estimate, rhs_at_estimate = np.split(model_output, 3, axis=1)
            # add all costs and performance metrics to store for the run
            betas_sample.append(betas_segment)
            inner_costs_sample.append(inner_cost_segment)
            # save the final value of the segment
            end_of_roi.append(state_at_estimate[-1, :])
            for iState, stateName in enumerate(hidden_state_names):
                state_fitted_roi[stateName] += list(state_at_estimate[:, iState])
    ####################################################################################################################
    #  optimise the following segments by matching the first B-spline height to the previous segment
    for iSegment in range(1, len(times_roi)):
        segment = times_roi[iSegment]
        input_segment = voltage_roi[iSegment]
        output_segment = current_roi[iSegment]
        knots = knots_roi[iSegment]
        collocation_segment = collocation_roi[iSegment]
        state_known_segment = states_known_roi[iSegment]
        # find the scaling coeff of the first height by matiching its height at t0 of the segment to the final value of the previous segment
        first_spline_coeff = end_of_roi[-1] / collocation_segment[0, 0]
        # initialise inner optimisation
        # we must re-initalise the optimisation with that excludes the first coefficient
        init_betas = init_betas_roi[iSegment]
        # drop every nth coefficient from this list that corresponds to the first b-spline for each state
        init_betas = np.delete(init_betas, indeces_to_drop)
        sigma0_betas = 0.2 * np.ones(nBsplineCoeffs - len(hidden_state_names))  # inital spread of values
        try:
            betas_segment, inner_cost_segment, evals_segment = optimise_segment(segment, input_segment,
                                                                                output_segment,
                                                                                knots,
                                                                                state_known_segment,init_betas, sigma0_betas, first_spline_coeff)
        except Exception:
            traceback.print_exc()
            print('Error encountered during opptimisation.')
            optimisationFailed = True
            return (np.NaN,np.NaN,np.NaN) # return dummy values
        # check collocation solution against truth
        else:
            model_output = model_bsplines_test.simulate(betas_segment, knots, segment)
            state_at_estimate, deriv_at_estimate, rhs_at_estimate = np.split(model_output, 3, axis=1)
            # add all costs and performance metrics to store for the run
            betas_sample.append(betas_segment)
            inner_costs_sample.append(inner_cost_segment)
            # store end of segment and the whole state for the
            end_of_roi.append(state_at_estimate[-1, :])
            for iState, stateName in enumerate(hidden_state_names):
                state_fitted_roi[stateName] += list(state_at_estimate[1:, iState])

    result = (betas_sample, inner_costs_sample, state_fitted_roi)
    return result


class BoundariesOneState(pints.Boundaries):
    """
    Boundary constraints on the parameters for a single state variable

    """

    def __init__(self):

        super(BoundariesOneState, self).__init__()

        # Limits on p1-p4 for a signle gative variable model
        self.lower_alpha = 1e-7  # Kylie: 1e-7
        self.upper_alpha = 1e3  # Kylie: 1e3
        self.lower_beta = 1e-7  # Kylie: 1e-7
        self.upper_beta = 0.4  # Kylie: 0.4

        # Lower and upper bounds for all parameters
        self.lower = [
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
        ]
        self.upper = [
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
        ]

        self.lower = np.array(self.lower)
        self.upper = np.array(self.upper)

        # Limits on maximum reaction rates
        self.rmin = 1.67e-5
        self.rmax = 1000

        # Voltages used to calculate maximum rates
        self.vmin = -120
        self.vmax = 60

    def n_parameters(self):
        return 4

    def check(self, transformed_parameters):

        debug = False

        # # check if parameters are sampled in log space
        # if InLogScale:
        #     # Transform parameters back to decimal space
        #     parameters = np.exp(transformed_parameters)
        # else:
        #     # leave as is
        #     parameters = transformed_parameters

        # Transform parameters back to decimal space
        parameters = np.exp(transformed_parameters)

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug:
                print('Lower')
            return False
        if np.any(parameters > self.upper):
            if debug:
                print('Upper')
            return False

        # Check maximum rate constants
        p1, p2, p3, p4 = parameters[:]

        # Check positive signed rates
        r = p1 * np.exp(p2 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r1')
            return False

        # Check negative signed rates
        r = p3 * np.exp(-p4 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r3')
            return False

        return True

# main
if __name__ == '__main__':
    #  load the voltage data:
    volts = np.genfromtxt("./protocol-staircaseramp.csv", skip_header=1, dtype=float, delimiter=',')
    #  check when the voltage jumps
    # read the times and valued of voltage clamp
    volt_times, volts = np.genfromtxt("./protocol-staircaseramp.csv", skip_header=1, dtype=float, delimiter=',').T
    # interpolate with smaller time step (milliseconds)
    volts_intepolated = sp.interpolate.interp1d(volt_times, volts, kind='previous')

    ## define the time interval on which the fitting will be done
    tlim = [3500, 14300]
    times = np.linspace(*tlim, tlim[-1]-tlim[0],endpoint=False)
    volts_new = V(times)
    ## Generate the synthetic data
    # parameter values for the model
    EK = -80
    thetas_true = [2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524]
    *ps, g = thetas_true[:9]
    # initialise and solve ODE
    x0 = [0, 1]
    state_names = ['a','r']
    # solve initial value problem
    solution = sp.integrate.solve_ivp(hh_model, [0,tlim[-1]], x0, args=[thetas_true], dense_output=True,method='LSODA',rtol=1e-8,atol=1e-8)
    x_ar = solution.sol(times)
    current_true = observation(times, x_ar, thetas_true)

    ## single state model
    # use a as unknown state
    theta_true = [np.log(2.26e-4), np.log(0.0699), np.log(3.45e-5), np.log(0.05462)]
    inLogScale = True
    param_names = ['p_1','p_2','p_3','p_4']
    a0 = [0]
    ion_channel_model_one_state = ode_a_only
    solution_one_state = sp.integrate.solve_ivp(ion_channel_model_one_state, [0,tlim[-1]], a0, args=[theta_true], dense_output=True, method='LSODA',
                                      rtol=1e-8, atol=1e-8)
    state_known_index = state_names.index('r')  # assume that we know r
    state_known = x_ar[state_known_index, :]
    state_name = hidden_state_names = 'a'

    # ## use r as unknown state
    # theta_true = [np.log(0.0873), np.log(8.91e-3), np.log(5.15e-3), np.log(0.03158)]
    # inLogScale = True
    # param_names = ['p_5','p_6','p_7','p_8']
    # r0 = [1]
    # ion_channel_model_one_state = ode_r_only
    # solution_one_state = sp.integrate.solve_ivp(ion_channel_model_one_state, [0,tlim[-1]], r0, args=[theta_true], dense_output=True,
    #                                     method='LSODA',
    #                                     rtol=1e-8, atol=1e-10)
    # state_known_index = state_names.index('a')  # assume that we know a
    # state_known = x_ar[state_known_index,:]
    # state_name = hidden_state_names = 'r'
    ################################################################################################################
    ## store true hidden state
    state_hidden_true = x_ar[state_names.index(state_name), :]
    rhs_true = ion_channel_model_one_state(times, state_hidden_true, theta_true)
    ## rectangular boundaries of thetas from Clerx et.al. paper - they are the same for two gating variables
    theta_lower_boundary = [np.log(10 ** (-5)), np.log(10 ** (-5)), np.log(10 ** (-5)), np.log(10 ** (-5))]
    theta_upper_boundary = [np.log(10 ** (3)), np.log(0.4), np.log(10 ** (3)), np.log(0.4)]
    ################################################################################################################
    ## B-spline representation setup
    # set times of jumps and a B-spline knot sequence
    nPoints_closest = 4  # the number of points from each jump where knots are placed at the finest grid
    nPoints_between_closest = 2  # step between knots at the finest grid
    nPoints_around_jump = 80  # the time period from jump on which we place medium grid
    step_between_knots = 16  # this is the step between knots around the jump in the medium grid
    nPoints_between_jumps = 2  # this is the number of knots at the coarse grid corresponding to slowly changing values
    ## find switchpoints
    d2v_dt2 = np.diff(volts_new, n=2)
    dv_dt = np.diff(volts_new)
    der1_nonzero = np.abs(dv_dt) > 1e-6
    der2_nonzero = np.abs(d2v_dt2) > 1e-6
    switchpoints = [a and b for a, b in zip(der1_nonzero, der2_nonzero)]
    ####################################################################################################################
    # get the times of all jumps
    a = [0] + [i for i, x in enumerate(switchpoints) if x] + [len(times)-1]  # get indeces of all the switchpoints, add t0 and tend
    # remove consecutive numbers from the list
    b = []
    for i in range(len(a)):
        if len(b) == 0:  # if the list is empty, we add first item from 'a' (In our example, it'll be 2)
            b.append(a[i])
        else:
            if a[i] > a[i - 1] + 1:  # for every value of a, we compare the last digit from list b
                b.append(a[i])
    jump_indeces = b.copy()
    print('V jump indeces:', jump_indeces)
    ## create multiple segments limited by time instances of jumps
    times_roi = []
    states_roi = []
    states_known_roi = []
    current_roi = []
    voltage_roi = []
    knots_roi = []
    collocation_roi = []
    colderiv_roi = []
    init_betas_roi = []
    for iJump, jump in enumerate(jump_indeces[:-1]):  # loop oversegments (nJumps - )
        # define a region of interest - we will need this to preserve the
        # trajectories of states given the full clamp and initial position, while
        ROI_start = jump
        ROI_end = jump_indeces[iJump + 1] + 1  # add one to ensure that t_end equals to t_start of the following segment
        ROI = times[ROI_start:ROI_end]
        x_ar = solution.sol(ROI)
        # get time points to compute the fit to ODE cost
        times_roi.append(ROI)
        # save states
        states_roi.append(solution_one_state.sol(ROI))
        states_known_roi.append(x_ar[state_known_index, :])
        # save current
        current_roi.append(observation(ROI, x_ar, thetas_true))
        # save voltage
        voltage_roi.append(V(ROI))
        ## add colloation points
        abs_distance_lists = [[(num - index) for num in range(ROI_start, ROI_end)] for index in
                              [ROI_start, ROI_end]]  # compute absolute distance between each time and time of jump
        min_pos_distances = [min(filter(lambda x: x >= 0, lst)) for lst in zip(*abs_distance_lists)]
        max_neg_distances = [max(filter(lambda x: x <= 0, lst)) for lst in zip(*abs_distance_lists)]
        # create a knot sequence that has higher density of knots after each jump
        knots_after_jump = [((x <= nPoints_closest) and (x % nPoints_between_closest == 0)) or (
                (nPoints_closest < x <= nPoints_around_jump) and (x % step_between_knots == 0)) for
                            x in min_pos_distances]  ##  ((x <= 2) and (x % 1 == 0)) or
        # knots_before_jump = [((x >= -nPoints_closest) and (x % (nPoints_closest + 1) == 0)) for x in
        #                      max_neg_distances]  # list on knots befor each jump - use this form if you don't want fine grid before the jump
        knots_before_jump = [(x >= -1) for x in max_neg_distances]  # list on knots before each jump - add a fine grid
        knots_jump = [a or b for a, b in
                      zip(knots_after_jump, knots_before_jump)]  # logical sum of mininal and maximal distances
        # convert to numeric array again
        knot_indeces = [i + ROI_start for i, x in enumerate(knots_jump) if x]
        indeces_inner = knot_indeces.copy()
        # add additional coarse grid of knots between two jumps:
        for iKnot, timeKnot in enumerate(knot_indeces[:-1]):
            # add coarse grid knots between jumps
            if knot_indeces[iKnot + 1] - timeKnot > step_between_knots:
                # create evenly spaced points and drop start and end - those are already in the grid
                knots_between_jumps = np.rint(
                    np.linspace(timeKnot, knot_indeces[iKnot + 1], num=nPoints_between_jumps + 2)[1:-1]).astype(int)
                # add indeces to the list
                indeces_inner = indeces_inner + list(knots_between_jumps)
            # add copies of the closest points to the jump
        ## end loop over knots
        indeces_inner.sort()  # sort list in ascending order - this is done inplace
        degree = 3
        # define the Boor points to
        indeces_outer = [indeces_inner[0]] * 3 + [indeces_inner[-1]] * 3
        boor_indeces = np.insert(indeces_outer, degree,
                                 indeces_inner)  # create knots for which we want to build splines
        knots = times[boor_indeces]
        # save knots for the segment - including additional points at the edges
        knots_roi.append(knots)
        # build the collocation matrix using the defined knot structure
        coeffs = np.zeros(len(knots) - degree - 1)  # number of splines will depend on the knot order
        spl_ones = BSpline(knots, np.ones_like(coeffs), degree)
        splinest = [None] * len(coeffs)
        splineder = [None] * len(coeffs)  # the grid of indtividual splines is required to generate a collocation matrix
        for i in range(len(coeffs)):
            coeffs[i] = 1.
            splinest[i] = BSpline(knots, coeffs.copy(), degree,
                                  extrapolate=False)  # create a spline that only has one non-zero coeff
            coeffs[i] = 0.
        collocation_roi.append(collocm(splinest, ROI))
        # create inital values of beta to be used at the true value of parameters
        init_betas_roi.append(0.5 * np.ones_like(coeffs))
    ##^ this loop stores the time intervals from which to draw collocation points and the data for piece-wise fitting # this to be used in params method of class ForwardModel
    ####################################################################################################################
    ## make indexing of B-spline coeffs generalisable for a set number of hidden states
    nBsplineCoeffs = len(coeffs) * len(hidden_state_names)  # this is the number of splinese per segment!
    print('Number of B-spline coeffs per segment: ' + str(nBsplineCoeffs))
    ## create a list of indeces to insert first B-spline coeffs for each segment
    indeces_to_add = [0]
    for iState in range(1, len(hidden_state_names)):
        indeces_to_add.append((len(coeffs) - 1) * iState)
    ## create a list of indeces to drop from the B-spline coeff sets for each segment
    indeces_to_drop = [0]
    for iState in range(1, len(hidden_state_names)):
        indeces_to_drop.append(int(len(coeffs) * iState))
    upper_bound_beta = 0.9999
    ####################################################################################################################
    # define classes for the optimisation
    nOutputs = 3
    # define a class that outputs only b-spline surface features
    class bsplineOutput(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, knots, times):
            # given times and return the simulated values
            tck = (knots, parameters, degree)
            dot_ = sp.interpolate.splev(times, tck, der=1)
            fun_ = sp.interpolate.splev(times, tck, der=0)
            # the RHS must be put into an array
            rhs = ion_channel_model_one_state(times, fun_, Thetas_ODE)
            return np.array([fun_,dot_,rhs]).T

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs

        def n_outputs(self):
            # Return the dimension of the output vector
            return nOutputs

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
            volts_for_model = self._values[:,1] # we need to make sure that voltage is read at the times within ROI so we pass it in as part of values
            d_y = g * x[:,0] * self._values[:,2] * (volts_for_model - EK) - self._values[:,0]
            data_fit_cost = np.transpose(d_y) @ d_y
            # compute the gradient matching cost
            d_deriv = (x_dot[:] - rhs[:]) ** 2
            integral_quad = sp.integrate.simpson(y=d_deriv, even='avg',axis=0)
            gradient_match_cost = np.sum(integral_quad, axis=0)
            # not the most elegant implementation because it just grabs global lambda
            return data_fit_cost + lambd * gradient_match_cost

    # # define a class that outputs only b-spline surfaces and its derivative for a segment
    nThetas = len(theta_true)
    class SegmentOutput(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            # given segments return the values for a segment
            coeffs = betas_segment
            tck = (knots, coeffs, degree)
            fun_ = sp.interpolate.splev(times, tck, der=0)
            dot_ = sp.interpolate.splev(times, tck, der=1)
            return np.array([fun_,dot_]).T

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nThetas

        def n_outputs(self):
            # Return the dimension of the output vector
            return 2

    # define an error w.r.t. the ODE parameters that assumes that it knows B-spline parameters - simply data fit
    class OuterCriterion(pints.ProblemErrorMeasure):
        # do I need to redefine custom init or can just drop this part?
        def __init__(self, problem, weights=None):
            super(OuterCriterion, self).__init__(problem)
            if weights is None:
                weights = [1] * self._n_outputs
            elif self._n_outputs != len(weights):
                raise ValueError(
                    'Number of weights must match number of problem outputs.')
            # Check weights
            self._weights = np.asarray([float(w) for w in weights])
        # this function is the function of theta - ODE parameters
        def __call__(self, thetas):
            # evaluate the integral at the value of ODE parameters
            # model_output = self._problem.evaluate(thetas)   # the output of the model with be an array of size nTimes x nOutputs
            # x, x_dot = np.split(model_output, 2, axis=1)
            x = state_all_segments
            # compute the data fit
            d_y = g * x[:] * state_known * (self._values[:,1] - EK) - self._values[:,0] # this part depends on theta_g
            data_fit_cost = np.transpose(d_y) @ d_y
            return data_fit_cost
    ####################################################################################################################
    ## Create objects for the optimisation
    # lambd = 1 # 0.3 # 0 # 1
    lambdas = [1, 10, 100, 1000]
    # set initial values and boundaries depending on the scale of search space
    if inLogScale:
        # theta in log scale
        init_thetas = -5 * np.ones(nThetas)
        sigma0_thetas = 2 * np.ones(nThetas)
        boundaries_thetas = pints.RectangularBoundaries(theta_lower_boundary, theta_upper_boundary)
        boundaries_thetas_Michael = BoundariesOneState()
    else:
        # theta in decimal scale
        init_thetas = 0.001 * np.ones(nThetas)
        sigma0_thetas = 0.0005 * np.ones(nThetas)
        boundaries_thetas = pints.RectangularBoundaries(np.exp(theta_lower_boundary), np.exp(theta_upper_boundary))
    # outer optimisation settings
    ### BEAR IN MIND THAT OUTER OPTIMISATION is conducted on the entire time-series
    model_bsplines_test = bsplineOutput()
    model_segments = SegmentOutput()
    ## create the problem of comparing the modelled current with measured current
    voltage = V(times)  # must read voltage at the correct times to match the output
    current_true = observation(times, solution.sol(times), thetas_true)
    values_to_match_output_ode = np.transpose(np.array([current_true, voltage]))
    # ^ we actually only need first two columns in this array but pints wants to have the same number of values and outputs
    problem_outer = pints.MultiOutputProblem(model=model_segments, times=times,
                                             values=values_to_match_output_ode)
    ## associate the cost with it
    # error_outer = OuterCriterion(problem=problem_outer)
    error_outer = OuterCriterion(problem=problem_outer)
    init_betas = 0.5 * np.ones(nBsplineCoeffs) # initial values of B-spline coefficients
    sigma0_betas = 0.2 * np.ones(nBsplineCoeffs)
    tic = tm.time()
    model_bsplines = bsplineOutput()
    ####################################################################################################################
    ## loop over lambdas
    for lambd in lambdas:
        print('lambda = ', lambd)
        # run the ebtire optimisation for a given lambda below, plot and store all results
        ####################################################################################################################
        # fit states at the true ODE param values to get the baseline value
        Thetas_ODE = theta_true.copy()
        # fit the b-spline surface given the true value of the ODE parameter vector
        result_at_truth = inner_optimisation(Thetas_ODE,times_roi,voltage_roi,current_roi,knots_roi,states_known_roi,init_betas_roi)
        betas_sample, inner_costs_sample, state_fitted_roi = result_at_truth
        if len(state_fitted_roi.items())>1:
            list_of_states = [state_values for _, state_values in state_fitted_roi.items()]
            state_all_segments = np.array(list_of_states)
        else:
            state_all_segments = np.array(state_fitted_roi[hidden_state_names])
        ## this is to be read at outer cost computation
        #### end of loop over segments
        # evaluate the cost functions at the sampled value of ODE parameter vector
        InnerCost_given_true_theta = sum(inner_costs_sample)
        OuterCost_given_true_theta = error_outer(Thetas_ODE)
        GradCost_given_true_theta = (InnerCost_given_true_theta - OuterCost_given_true_theta) / lambd
        print('Costs at truth:')
        print('True theta: ', theta_true)
        print('Lambda: {0:8.3f}'.format(lambd))
        # print all of the above three costs in one print command
        print('Inner cost: {0:8.8f} \t Data cost: {1:8.8f} \t Gradient matching cost: {2:8.8f}'.format(InnerCost_given_true_theta,
                                                                                           OuterCost_given_true_theta,
                                                                                           GradCost_given_true_theta))
        ####################################################################################################################
        # take 1: loosely based on ask-tell example from  pints
        convergence_threshold =1e-8
        iter_for_convergence = 20
        # Create an outer optimisation object
        # optimiser_outer = pints.CMAES(x0=init_thetas,sigma0=sigma0_thetas, boundaries=boundaries_thetas) # with simple rectangular boundaries
        optimiser_outer = pints.CMAES(x0=init_thetas, sigma0=sigma0_thetas, boundaries=boundaries_thetas_Michael) # with boundaries accounting for the reaction rates
        optimiser_outer.set_population_size(min(len(theta_true)*5,30))
        ## Create placeholders for the optimisation
        theta_visited = []
        theta_guessed = []
        f_guessed = []
        theta_best = []
        f_outer_best = []
        f_inner_best = []
        f_gradient_best = []
        InnerCosts_all = []
        OuterCosts_all = []
        GradCost_all = []
        # run outer optimisation for some iterations
        # create a folder to store the results for a given lambda and state
        folderName = 'Results_' + state_name + '_lambda_' + str(int(lambd))
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        # create a csv file to store the results
        csv_file_name = folderName+'/iterations_one_state_'+state_name+'.csv'
        column_names = ['Iteration','Walker','Theta_1','Theta_2','Theta_3','Theta_4','Inner Cost', 'Outer Cost', 'Gradient Cost']
        # parallelisation settings
        ncpu = mp.cpu_count()
        ncores = 10
        big_tic = tm.time()
        # open the file to write to
        with open(csv_file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)
            # run the outer optimisation
            for i in range(200):
                # get the next points (multiple locations)
                thetas = optimiser_outer.ask()
                # create the placeholder for cost functions
                OuterCosts = []
                InnerCosts = []
                GradCosts = []
                betas_visited = []
                # for each theta in the sample
                tic = tm.time()
                # run inner optimisation for each theta sample
                with mp.get_context('fork').Pool(processes=min(ncpu, ncores)) as pool:
                    results = pool.starmap(inner_optimisation,
                                           zip(thetas, repeat(times_roi), repeat(voltage_roi), repeat(current_roi),
                                               repeat(knots_roi), repeat(states_known_roi), repeat(init_betas_roi)))
                # package results is a list of tuples
                # extract the results
                for iSample, result in enumerate(results):
                    betas_sample, inner_costs_sample, state_fitted_at_sample = result
                    # get the states at this sample
                    if len(state_fitted_roi.items()) > 1:
                        list_of_states = [state_values for _, state_values in state_fitted_at_sample.items()]
                        state_all_segments = np.array(list_of_states)
                    else:
                        state_all_segments = np.array(state_fitted_at_sample[hidden_state_names])
                    # evaluate the cost functions at the sampled value of ODE parameter vector
                    InnerCost = sum(inner_costs_sample)
                    OuterCost = error_outer(thetas[iSample, :])
                    GradCost = (InnerCost - OuterCost) / lambd
                    # store the costs
                    InnerCosts.append(InnerCost)
                    OuterCosts.append(OuterCost)
                    GradCosts.append(GradCost)
                    betas_visited.append(betas_sample)
                # tell the optimiser about the costs
                optimiser_outer.tell(OuterCosts)
                # store the best point
                index_best = OuterCosts.index(min(OuterCosts))
                theta_best.append(thetas[index_best, :])
                f_outer_best.append(OuterCosts[index_best])
                f_inner_best.append(InnerCosts[index_best])
                f_gradient_best.append(GradCosts[index_best])
                betas_best = betas_visited[index_best]
                # ad hoc solution to the problem of the optimiser getting stuck at the boundary
                init_betas_roi = []
                for betas_to_intitialise in betas_best:
                    if any(betas_to_intitialise == upper_bound_beta):
                        # find index of the offending beta
                        index = np.where(betas_to_intitialise == upper_bound_beta)[0]
                        betas_to_intitialise[index] = upper_bound_beta * 0.9
                    init_betas_roi.append(betas_to_intitialise)
                # store the costs for all samples in the iteration
                InnerCosts_all.append(InnerCosts)
                OuterCosts_all.append(OuterCosts)
                GradCost_all.append(GradCosts)
                # store the visited points
                theta_visited.append(thetas)

                # print the results
                print('Iteration: ', i)
                print('Best parameters: ', theta_best[-1])
                print('Best objective: ', f_outer_best[-1])
                print('Mean objective: ', np.mean(OuterCosts))
                print('Inner objective at best sample: ', f_inner_best[-1])
                print('Gradient objective at best sample: ', f_gradient_best[-1])
                print('Time elapsed: ', tm.time() - tic)

                # write the results to a csv file
                for iWalker in range(len(thetas)):
                    row = [i, iWalker] + list(thetas[iWalker]) + [InnerCosts[iWalker], OuterCosts[iWalker],
                                                                  GradCosts[iWalker]]
                    writer.writerow(row)
                file.flush()

                # check for convergence
                if (i > iter_for_convergence):
                    # check how the cost increment changed over the last 10 iterations
                    d_cost = np.diff(f_outer_best[-iter_for_convergence:])
                    # if all incrementa are below a threshold break the loop
                    if all(d <= convergence_threshold for d in d_cost):
                        print("No changes in" + str(iter_for_convergence) + "iterations. Terminating")
                        break
                ## end convergence check condition
            ## end loop over iterations
        big_toc = tm.time()
        # convert the lists to numpy arrays
        theta_best = np.array(theta_best)
        f_outer_best = np.array(f_outer_best)
        f_inner_best = np.array(f_inner_best)
        f_gradient_best = np.array(f_gradient_best)
        print('Total time taken: ', big_toc - big_tic)
        print('===========================================================================================================')
        ####################################################################################################################
        ## save the best betas as a table to csv file
        df_betas = pd.DataFrame()
        for i, beta in enumerate(init_betas_roi):
            df_betas['segment_'+str(i)] = beta
        df_betas.to_csv(folderName+'/best_betas_'+state_name+'.csv', index=False)
        ####################################################################################################################
        # produce optimised model output
        Thetas_ODE = theta_best[-1]
        state_fitted_roi = {key: [] for key in hidden_state_names}
        deriv_fitted_roi = {key: [] for key in hidden_state_names}
        rhs_fitted_roi = {key: [] for key in hidden_state_names}
        for iSegment in range(1):
            segment = times_roi[iSegment]
            knots = knots_roi[iSegment]
            betas_segment = init_betas_roi[iSegment]
            model_output = model_bsplines_test.simulate(betas_segment, knots, segment)
            state_at_estimate, deriv_at_estimate, rhs_at_estimate = np.split(model_output, 3, axis=1)
            for iState, stateName in enumerate(hidden_state_names):
                state_fitted_roi[stateName] += list(state_at_estimate[:, iState])
                deriv_fitted_roi[stateName] += list(deriv_at_estimate[:, iState])
                rhs_fitted_roi[stateName] += list(rhs_at_estimate[:, iState])
        for iSegment in range(1, len(times_roi)):
            segment = times_roi[iSegment]
            knots = knots_roi[iSegment]
            betas_segment = init_betas_roi[iSegment]
            model_output = model_bsplines_test.simulate(betas_segment, knots, segment)
            state_at_estimate, deriv_at_estimate, rhs_at_estimate = np.split(model_output, 3, axis=1)
            for iState, stateName in enumerate(hidden_state_names):
                state_fitted_roi[stateName] += list(state_at_estimate[1:, iState])
                deriv_fitted_roi[stateName] += list(deriv_at_estimate[1:, iState])
                rhs_fitted_roi[stateName] += list(rhs_at_estimate[1:, iState])
        # stitch segments together
        if len(state_fitted_roi.items()) > 1:
            list_of_states = [state_values for _, state_values in state_fitted_roi.items()]
            state_all_segments = np.array(list_of_states)
            list_of_derivs = [deriv_values for _, deriv_values in deriv_fitted_roi.items()]
            deriv_all_segments = np.array(list_of_states)
            list_of_rhs = [rhs_values for _, rhs_values in rhs_fitted_roi.items()]
            rhs_all_segments = np.array(list_of_states)
        else:
            state_all_segments = np.array(state_fitted_roi[hidden_state_names])
            deriv_all_segments = np.array(deriv_fitted_roi[hidden_state_names])
            rhs_all_segments = np.array(rhs_fitted_roi[hidden_state_names])

        current_model = g * state_all_segments[:] * state_known * (voltage - EK)
        # save the model output into a pickle file - in case the plots break again!
        with open(folderName + '/model_output_' + state_name + '.pkl', 'wb') as f:
            pkl.dump([times, current_model, state_all_segments, deriv_all_segments, rhs_all_segments, InnerCost_given_true_theta, OuterCost_given_true_theta, GradCost_given_true_theta], f)
        ####################################################################################################################
        # plot evolution of inner costs
        plt.figure(figsize=(10, 6))
        plt.semilogy()
        plt.xlabel('Iteration')
        plt.ylabel('Inner optimisation cost')
        for iIter in range(len(f_outer_best) - 1):
            plt.scatter(iIter * np.ones(len(InnerCosts_all[iIter])), InnerCosts_all[iIter], c='k', marker='.', alpha=.5,
                        linewidths=0)
        iIter += 1
        plt.scatter(iIter * np.ones(len(InnerCosts_all[iIter])), InnerCosts_all[iIter], c='k', marker='.', alpha=.5,
                    linewidths=0, label='Sample cost min: J(C / Theta, Y) = ' + "{:.5e}".format(min(InnerCosts_all[iIter])))
        plt.plot(f_inner_best, '-b', linewidth=1.5,
                 label='Best cost:J(C / Theta_{best}, Y) = ' + "{:.5e}".format(
                     f_inner_best[-1]))
        plt.plot(range(len(f_inner_best)), np.ones(len(f_inner_best)) * InnerCost_given_true_theta, '--m', linewidth=2.5,
                 alpha=.5,
                 label='Collocation solution: J(C / Theta_{true}, Y) = ' + "{:.5e}".format(InnerCost_given_true_theta))
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(folderName + '/inner_cost_ask_tell_one_state_' + state_name + '.png', dpi=400)

        # plot evolution of outer costs
        plt.figure(figsize=(10, 6))
        plt.semilogy()
        plt.xlabel('Iteration')
        plt.ylabel('Outer optimisation cost')
        for iIter in range(len(f_outer_best) - 1):
            plt.scatter(iIter * np.ones(len(OuterCosts_all[iIter])), OuterCosts_all[iIter], c='k', marker='.', alpha=.5,
                        linewidths=0)
        iIter += 1
        plt.scatter(iIter * np.ones(len(OuterCosts_all[iIter])), OuterCosts_all[iIter], c='k', marker='.', alpha=.5,
                    linewidths=0, label='Sample cost: H(Theta / C, Y)')
        # plt.plot(range(iIter), np.ones(iIter) * OuterCost_true, '-m', linewidth=2.5, alpha=.5,label=r'B-splines fit to true state: $H(\Theta \mid  \hat{C}_{direct}, \bar{\mathbf{y}}) = $' + "{:.7f}".format(
        #              OuterCost_true))
        plt.plot(range(len(f_outer_best)), np.ones(len(f_outer_best)) * OuterCost_given_true_theta, '--m', linewidth=2.5,
                 alpha=.5, label='Collocation solution: H(Theta_{true} /  C, Y) = ' + "{:.5e}".format(
                OuterCost_given_true_theta))
        plt.plot(f_outer_best, '-b', linewidth=1.5,
                 label='Best cost:H(Theta_{best} / C, Y) = ' + "{:.5e}".format(f_outer_best[-1]))
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(folderName + '/outer_cost_ask_tell_one_state_' + state_name + '.png', dpi=400)

        # plot evolution of outer costs
        plt.figure(figsize=(10, 6))
        plt.semilogy()
        plt.xlabel('Iteration')
        plt.ylabel('Gradient matching cost')
        for iIter in range(len(f_gradient_best) - 1):
            plt.scatter(iIter * np.ones(len(GradCost_all[iIter])), GradCost_all[iIter], c='k', marker='.', alpha=.5,
                        linewidths=0)
        iIter += 1
        plt.scatter(iIter * np.ones(len(GradCost_all[iIter])), GradCost_all[iIter], c='k', marker='.', alpha=.5,
                    linewidths=0, label='Sample cost: G_{ODE}(C / Theta, Y)')
        plt.plot(range(len(f_gradient_best)), np.ones(len(f_gradient_best)) * GradCost_given_true_theta, '--m',
                 linewidth=2.5, alpha=.5, label='Collocation solution: G_{ODE}( C /  Theta_{true}, Y) = ' + "{:.5e}".format(
                GradCost_given_true_theta))
        plt.plot(f_gradient_best, '-b', linewidth=1.5,
                 label='Best cost:G_{ODE}(C / Theta, Y) = ' + "{:.5e}".format(f_gradient_best[-1]))
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(folderName + '/gradient_cost_ask_tell_one_state_' + state_name + '.png', dpi=400)

        # plot parameter values after search was done on decimal scale
        fig, axes = plt.subplots(len(theta_true), 1, figsize=(5 * len(theta_true), 8), sharex=True)
        for iAx, ax in enumerate(axes.flatten()):
            for iIter in range(len(theta_best)):
                x_visited_iter = theta_visited[iIter][:, iAx]
                ax.scatter(iIter * np.ones(len(x_visited_iter)), x_visited_iter, c='k', marker='.', alpha=.2, linewidth=0)
            ax.plot(range(iIter + 1), np.ones(iIter + 1) * theta_true[iAx], '--m', linewidth=2.5, alpha=.5,
                    label=r"true: log(" + param_names[iAx] + ") = " + "{:.6f}".format(theta_true[iAx]))
            # ax.plot(theta_guessed[:,iAx],'--r',linewidth=1.5,label=r"guessed: $\theta_{"+str(iAx+1)+"} = $" +"{:.4f}".format(theta_guessed[-1,iAx]))
            ax.plot(theta_best[:, iAx], '-b', linewidth=1.5,
                    label=r"best: log(" + param_names[iAx] + ") = " + "{:.6f}".format(theta_best[-1, iAx]))
            ax.set_ylabel('log(' + param_names[iAx] + ')')
            ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(folderName + '/ODE_params_one_state_log_scale_' + state_name + '.png', dpi=400)

        # plot parameter values converting from log scale to decimal
        fig, axes = plt.subplots(len(theta_true), 1, figsize=(5 * len(theta_true), 8), sharex=True)
        for iAx, ax in enumerate(axes.flatten()):
            for iIter in range(len(theta_best)):
                x_visited_iter = theta_visited[iIter][:, iAx]
                ax.scatter(iIter * np.ones(len(x_visited_iter)), np.exp(x_visited_iter), c='k', marker='.', alpha=.2,
                           linewidth=0)
            ax.plot(range(iIter + 1), np.ones(iIter + 1) * np.exp(theta_true[iAx]), '--m', linewidth=2.5, alpha=.5,
                    label="true: " + param_names[iAx] + " = " + "{:.6f}".format(np.exp(theta_true[iAx])))
            # ax.plot(np.exp(theta_guessed[:,iAx]),'--r',linewidth=1.5,label="guessed: $a_{"+str(iAx+1)+"} = $" +"{:.4f}".format(np.exp(theta_guessed[-1,iAx])))
            ax.plot(np.exp(theta_best[:, iAx]), '-b', linewidth=1.5,
                    label="best: " + param_names[iAx] + " = " + "{:.6f}".format(np.exp(theta_best[-1, iAx])))
            ax.set_ylabel(param_names[iAx])
            ax.set_yscale('log')
            ax.legend(loc='best')
        ax.set_xlabel('Iteration')
        plt.tight_layout()
        plt.savefig(folderName + '/ODE_params_one_state_' + state_name + '.png', dpi=400)
        ####################################################################################################################
        # plot evolution of inner costs
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        # y_labels = ['I', '$\dot{' + state_name + '}$', '$' + state_name + '$']
        y_labels = ['I', 'd' + state_name, state_name]
        axes[0].plot(times, current_true, '-k', label='Current true')
        axes[0].plot(times, current_model, '--r', label='Optimised model output')
        axes[1].plot(times, rhs_true, '-k', label='RHS true')
        axes[1].plot(times, deriv_all_segments, '--r', label='B-spline derivative')
        axes[1].plot(times, rhs_all_segments, '--c', label='RHS at collocation solution')
        axes[2].plot(times, state_hidden_true, '-k', label=state_name + 'true')
        axes[2].plot(times, state_all_segments, '--r', label='Collocation solution')
        for iAx, ax in enumerate(axes.flatten()):
            ax.legend(fontsize=12, loc='best')
            ax.set_ylabel(y_labels[iAx], fontsize=12)
        ax.set_xlabel('time,ms', fontsize=12)
        plt.tight_layout(pad=0.3)
        plt.savefig(folderName + '/cost_terms_ask_tell_one_state_' + state_name + '.png', dpi=400)

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        # y_labels = ['$I_{true} - I_{model}$', '$\dot{' + state_name + r'} - RHS(\beta)$', r'$' + state_name + r'$ - $\Phi\beta$']
        y_labels = ['I_true - I_model', 'd' + state_name + '(C) - RHS(C)', state_name + '(C) - Phi C']
        axes[0].plot(times, current_true - current_model, '-k', label='Data error')
        axes[1].plot(times, deriv_all_segments - rhs_all_segments, '-k', label='Derivative error')
        axes[2].plot(times, state_hidden_true - state_all_segments, '-k', label='Approximation error')
        for iAx, ax in enumerate(axes.flatten()):
            ax.legend(fontsize=12, loc='best')
            ax.set_ylabel(y_labels[iAx], fontsize=12)
        ax.set_xlabel('time,ms', fontsize=12)
        plt.tight_layout(pad=0.3)
        plt.savefig(folderName + '/erros_ask_tell_one_state_' + state_name + '.png', dpi=400)
        ################################################################################################################
    ## end of loop over lambda values
# imports
import numpy as np

import setup
from setup import *
import multiprocessing as mp
from itertools import repeat
import traceback
matplotlib.use('AGG')
plt.ioff()

# definitions

# get Voltage for time in ms
def V(t):
    return volts_intepolated(t/ 1000)

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

def kemp_model(t, x, theta):
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
    h_inf = ah/(ah + bh)
    tau_h = 1/(ah + bh)
    dh = (h_inf - h)/tau_h
    return [dop,dc1,dh]

def kemp_model_ss(t, x, theta):
    op, c1, h = x[:3]
    *p, g = theta[:13]
    v = -80
    a1 = p[0] * np.exp(p[1] * v)
    b1 = p[2] * np.exp(-p[3] * v)

    ah = p[6] * np.exp(-p[7] * v)
    bh = p[4] * np.exp(p[5] * v)

    a2 = p[8] * np.exp(p[9] * v)
    b2 = p[10] * np.exp(-p[11] * v)
    dop = a2*c1 - b2*op
    dc1 = b2*op + a1*(1 - op - c1) - (a2 + b1)*c1
    h_inf = ah/(ah + bh)
    tau_h = 1/(ah + bh)
    dh = (h_inf - h)/tau_h
    return [dop,dc1,dh]

def two_state_model(t, x, theta):
    a, r = x[:2]
    p = theta[:8]
    v = V(t)
    k1 = np.exp(p[0] + np.exp(p[1]) * v)
    k2 = np.exp(p[2]-np.exp(p[3]) * v)
    k3 = np.exp(p[4] + np.exp(p[5]) * v)
    k4 = np.exp(p[6] -np.exp(p[7]) * v)
    a_inf = k1 / (k1 + k2)
    tau_a = 1 / (k1 + k2)
    r_inf = k4 / (k3 + k4)
    tau_r = 1 / (k3 + k4)
    da = (a_inf - a) / tau_a
    dr = (r_inf - r) / tau_r
    return [da,dr]

def kemp_observation(t, x, theta):
    op, c1, h = x[:3]
    *p, g = theta[:13]
    return g * op * h * (V(t) - EK)

def observation(t, x, theta):
    # I
    a, r = x[:2]
    # *ps, g = theta[:9]
    return g * a * r * (V(t) - EK)
# get Voltage for time in ms
def V(t):
    return volts_intepolated((t)/ 1000)

def optimise_first_segment(roi,input_roi,output_roi,support_roi,state_known_roi,init_betas, sigma0_betas):
    nOutputs = 6
    # define a class that outputs only b-spline surface features
    class bsplineOutputSegment(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            coeffs_a, coeffs_r = np.split(parameters, 2)
            tck_a = (support_roi, coeffs_a, degree)
            tck_r = (support_roi, coeffs_r, degree)
            dot_a = sp.interpolate.splev(times, tck_a, der=1)
            dot_r = sp.interpolate.splev(times, tck_r, der=1)
            fun_a = sp.interpolate.splev(times, tck_a, der=0)
            fun_r = sp.interpolate.splev(times, tck_r, der=0)
            # the RHS must be put into an array
            # print(Thetas_ODE) #just to make sure the global variable is being updated
            dadr = two_state_model(times, [fun_a, fun_r], Thetas_ODE)
            rhs_theta = np.array(dadr)
            spline_surface = np.array([fun_a, fun_r])
            spline_deriv = np.array([dot_a, dot_r])
            # pack all required variables into the same array - will be the wrong orientation from pints preferred nTimes x nOutputs
            packed_output = np.concatenate((spline_surface, spline_deriv, rhs_theta), axis=0)
            return np.transpose(packed_output)

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs

        def n_outputs(self):
            # Return the dimension of the output vector
            return nOutputs
    # define a class that outputs only b-spline surface features
    model_bsplines = bsplineOutputSegment()
    values_to_match_output_dims = np.transpose(np.array([output_roi, input_roi, state_known_roi,output_roi, input_roi, state_known_roi]))
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
    return betas_roi, cost_roi, nEvaluations

def optimise_segment(roi,input_roi,output_roi,support_roi,state_known_roi,init_betas, sigma0_betas,first_spline_coeff):
    nOutputs = 6
    # define a class that outputs only b-spline surface features
    class bsplineOutputSegment(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            # given times and return the simulated values
            coeffs_a, coeffs_r = np.split(parameters, 2)
            coeffs_with_first_a = np.insert(coeffs_a, 0, first_spline_coeff[0])
            coeffs_with_first_r = np.insert(coeffs_r, 0, first_spline_coeff[1])
            tck_a = (support_roi, coeffs_with_first_a, degree)
            tck_r = (support_roi, coeffs_with_first_r, degree)
            dot_a = sp.interpolate.splev(times, tck_a, der=1)
            dot_r = sp.interpolate.splev(times, tck_r, der=1)
            fun_a = sp.interpolate.splev(times, tck_a, der=0)
            fun_r = sp.interpolate.splev(times, tck_r, der=0)
            # the RHS must be put into an array
            dadr = two_state_model(times, [fun_a, fun_r], Thetas_ODE)
            rhs_theta = np.array(dadr)
            spline_surface = np.array([fun_a, fun_r])
            spline_deriv = np.array([dot_a, dot_r])
            # pack all required variables into the same array - will be the wrong orientation from pints preferred nTimes x nOutputs
            packed_output = np.concatenate((spline_surface, spline_deriv, rhs_theta), axis=0)
            return np.transpose(packed_output)

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs-len(hidden_state_names)

        def n_outputs(self):
            # Return the dimension of the output vector
            return nOutputs
    # define a class that outputs only b-spline surface features
    model_bsplines = bsplineOutputSegment()
    values_to_match_output_dims = np.transpose(np.array([output_roi, input_roi, state_known_roi, output_roi, input_roi, state_known_roi]))
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
    coeffs_with_first = np.insert(betas_roi,indeces_to_add,first_spline_coeff)
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
        # sigma0_betas = 0.2 * np.ones(nBsplineCoeffs)
        sigma0_betas = None
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
        # sigma0_betas = 0.2 * np.ones(nBsplineCoeffs - len(hidden_state_names))  # inital spread of values
        sigma0_betas = None
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


class BoundariesTwoStates(pints.Boundaries):
    """
    Boundary constraints on the parameters for a two state variables

    """

    def __init__(self):

        super(BoundariesTwoStates, self).__init__()

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
        return 8

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
        p1, p2, p3, p4, p5, p6, p7, p8 = parameters[:8]
        # Check positive signed rates
        r = p1 * np.exp(p2 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r1')
            return False
        r = p5 * np.exp(p6 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r2')
            return False

        # Check negative signed rates
        r = p3 * np.exp(-p4 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r3')
            return False
        r = p7 * np.exp(-p8 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r4')
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
    # define the weight on the gradienet matching cost
    lambd = 1000000  # 0.3 # 0 # 1 ## - found out that with multiple states a cost with lambda 1 does not cope for segments where a is almost flat
    ## define the time interval on which the fitting will be done
    tlim = [300, 14899]
    times = np.linspace(*tlim, tlim[-1]-tlim[0],endpoint=False)
    volts_new = V(times)
    ## Generate the synthetic data
    # parameter values for the model
    EK = -80
    thetas_true = [2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524]
    theta_true = np.log(thetas_true[:-1])
    param_names = [f'p_{i}' for i in range(1,len(theta_true)+1)]
    state_names = ['a', 'r']
    inLogScale = True
    ## HH model
    # g = 0.1524
    # x0 = [0, 1]
    # # solve initial value problem
    # solution = sp.integrate.solve_ivp(hh_model, [0,tlim[-1]], x0, args=[thetas_true], dense_output=True,method='LSODA',rtol=1e-8,atol=1e-8)
    # state_hidden_true = solution.sol(times)
    # current_HH = observation(times, state_hidden_true, thetas_true)
    # current_true = current_HH

    ## Kemp model
    p_kemp = [8.5318e-03, 8.3176e-02, 1.2628e-02, 1.03628e-07, 2.702763e-01, 1.580004e-02, 7.6669948e-02, 2.2457500e-02,
              1.490338e-01, 2.431569e-02, 5.58072e-04, 4.06619e-02, 8.471005e-02]
    g = 8.471005e-02
    # find steady state at -80mV to use as initial condition
    x0_init = [0.5, 0.5, 0]
    # run for a long time for the slow rate states to settle
    t_end = 10e5
    solution_ss = sp.integrate.solve_ivp(kemp_model_ss, [0, t_end], x0_init, args=[p_kemp], dense_output=True,
                                         method='LSODA', rtol=1e-8, atol=1e-8)
    x0_kemp = solution_ss.sol(t_end)
    print('Steady state at V=-80mv: ', x0_kemp)
    solution_kemp = sp.integrate.solve_ivp(kemp_model, [0, tlim[-1]], x0_kemp, args=[p_kemp], dense_output=True,
                                           method='LSODA', rtol=1e-8, atol=1e-8)
    x_kemp = solution_kemp.sol(times)
    current_kemp = kemp_observation(times, x_kemp, p_kemp)
    current_true = current_kemp
    state_hidden_true = x_kemp[1:,:]

    state_names = hidden_state_names= ['a','r']
    ## rectangular boundaries of thetas from Clerx et.al. paper - they are the same for two gating variables
    theta_lower_boundary = [np.log(10 ** (-5)), np.log(10 ** (-5)), np.log(10 ** (-5)), np.log(10 ** (-5)), np.log(10 ** (-5)), np.log(10 ** (-5)), np.log(10 ** (-5)), np.log(10 ** (-5))]
    theta_upper_boundary = [np.log(10 ** (3)), np.log(0.4), np.log(10 ** (3)), np.log(0.4), np.log(10 ** (3)), np.log(0.4), np.log(10 ** (3)), np.log(0.4)]
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
    der1_nonzero = np.abs(dv_dt) > 1e-1
    der2_nonzero = np.abs(d2v_dt2) > 1e-1
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
        x_ar = solution_kemp.sol(ROI)
        # get time points to compute the fit to ODE cost
        times_roi.append(ROI)
        # save states
        # states_roi.append(x_ar)
        states_known_roi.append([1]*len(ROI)) # adding ones in case we have situation where one of the known states is involved in output fn
        # save current
        current_roi.append(kemp_observation(ROI, x_ar, thetas_true))
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
        init_betas_roi.append(0.5 * np.ones(len(coeffs)*len(hidden_state_names)))
    ##^ this loop stores the time intervals from which to draw collocation points and the data for piece-wise fitting # this to be used in params method of class ForwardModel
    ####################################################################################################################
    ## make indexing of B-spline coeffs generalisable for a set number of hidden states
    nBsplineCoeffs = len(coeffs) * len(hidden_state_names) # this is the number of splinese per segment!
    ## create a list of indeces to insert first B-spline coeffs for each segment
    indeces_to_add = [0]
    for iState in range(1,len(hidden_state_names)):
        indeces_to_add.append((len(coeffs)-1)*iState)
    ## create a list of indeces to drop from the B-spline coeff sets for each segment
    indeces_to_drop = [0]
    for iState in range(1, len(hidden_state_names)):
        indeces_to_drop.append(int(len(coeffs) * iState))
    upper_bound_beta = 0.99
    ####################################################################################################################
    ## create pints classes for the optimisation
    roi = []
    print('Number of B-spline coeffs per segment: ' + str(nBsplineCoeffs))
    # define a class that outputs only b-spline surface features
    class bsplineOutput(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, support, times):
            # given times and return the simulated values
            coeffs_a, coeffs_r = np.split(parameters, 2)
            tck_a = (support, coeffs_a, degree)
            tck_r = (support, coeffs_r, degree)
            dot_a = sp.interpolate.splev(times, tck_a, der=1)
            dot_r = sp.interpolate.splev(times, tck_r, der=1)
            fun_a = sp.interpolate.splev(times, tck_a, der=0)
            fun_r = sp.interpolate.splev(times, tck_r, der=0)
            # the RHS must be put into an array
            dadr = two_state_model(times, [fun_a, fun_r], Thetas_ODE)
            rhs_theta = np.array(dadr)
            spline_surface = np.array([fun_a, fun_r])
            spline_deriv = np.array([dot_a, dot_r])
            # pack all required variables into the same array - will be the wrong orientation from pints preferred nTimes x nOutputs
            packed_output = np.concatenate((spline_surface, spline_deriv, rhs_theta), axis=0)
            return np.transpose(packed_output)

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs

        def n_outputs(self):
            # Return the dimension of the output vector
            return 6
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
            model_output = self._problem.evaluate(betas)  # the output of the model with be an array of size nTimes x nOutputs
            x, x_dot, rhs = np.split(model_output, 3, axis=1)  # we split the array into states, state derivs, and RHSs
            # compute the data fit
            volts_for_model = self._values[:,1]  # we need to make sure that voltage is read at the times within ROI so we pass it in as part of values
            d_y = g * np.prod(x, axis=1) * (volts_for_model - EK) - self._values[:, 0]
            data_fit_cost = np.transpose(d_y) @ d_y
            # compute the gradient matching cost
            d_deriv = (x_dot - rhs) ** 2
            integral_quad = sp.integrate.simpson(y=d_deriv, even='avg', axis=0)
            gradient_match_cost = np.sum(integral_quad, axis=0)
            # not the most elegant implementation because it just grabs global lambda
            return data_fit_cost + lambd * gradient_match_cost

    ## this one is not really used within outer criterion, how to define one without the problem/model?
    # # define a class that outputs only b-spline surfaces for all segments
    nThetas = len(theta_true)
    betas_segment = []
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
        def __call__(self, theta):
            # evaluate the integral at the value of ODE parameters
            # model_output = self._problem.evaluate(thetas)   # the output of the model with be an array of size nTimes x nOutputs
            # x, x_dot = np.split(model_output, 2, axis=1)
            x = state_all_segments
            # current_model = observation(times, state_all_segments, thetas)
            # d_y = current_model - self._values[:,0]
            # compute the data fit
            d_y = g * np.prod(x, axis=0) * (self._values[:,1] - EK) - self._values[:,0] # this part depends on theta_g
            data_fit_cost = np.transpose(d_y) @ d_y
            return data_fit_cost
    ####################################################################################################################
    ## Create objects for the optimisation
    # set initial values and boundaries
    if inLogScale:
        # theta in log scale
        # init_thetas = -5 * np.ones(nThetas)
        init_thetas = theta_true # start around the true solution to see how long it takes to converge
        sigma0_thetas = 0.1 * np.ones(nThetas)
        boundaries_thetas = pints.RectangularBoundaries(theta_lower_boundary, theta_upper_boundary)
        boundaries_thetas_Michael = BoundariesTwoStates()
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
    current_true = kemp_observation(times, solution_kemp.sol(times), thetas_true)
    values_to_match_output_ode = np.transpose(np.array([current_true, voltage]))
    # ^ we actually only need first two columns in this array but pints wants to have the same number of values and outputs
    problem_outer = pints.MultiOutputProblem(model=model_segments, times=times,
                                             values=values_to_match_output_ode)
    ## associate the cost with it
    # error_outer = OuterCriterion(problem=problem_outer)
    error_outer = OuterCriterion(problem=problem_outer)
    init_betas = 0.5 * np.ones(nBsplineCoeffs) # initial values of B-spline coefficients
    tic = tm.time()
    model_bsplines = bsplineOutput()
    ####################################################################################################################
    # fit states at the true ODE param values to get the baseline values of cost functions
    Thetas_ODE = theta_true.copy()
    result_at_truth = inner_optimisation(Thetas_ODE,times_roi,voltage_roi,current_roi,knots_roi,states_known_roi,init_betas_roi)
    betas_sample, inner_costs_sample, state_fitted_roi = result_at_truth
    list_of_states = [state_values for _, state_values in state_fitted_roi.items()]
    state_all_segments = np.array(list_of_states) ## this is to be read at outer cost computation
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
    convergence_threshold = 1e-7
    iter_for_convergence = 20
    max_iter = 1000
    # Create an outer optimisation object
    big_tic = tm.time()
    # optimiser_outer = pints.CMAES(x0=init_thetas,sigma0=sigma0_thetas, boundaries=boundaries_thetas) # with simple rectangular boundaries
    optimiser_outer = pints.CMAES(x0=init_thetas, sigma0=sigma0_thetas, boundaries=boundaries_thetas_Michael) # with boundaries accounting for the reaction rates
    optimiser_outer.set_population_size(min(len(theta_true)*7,30))
    ## Run optimisation
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
    folderName = 'Results_kemp_lambda_' + str(int(lambd)) + '_start_at_truth'
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    # create a logger file
    csv_file_name = folderName + '/iterations_both_states.csv'
    column_names = ['Iteration', 'Walker', 'Theta_1', 'Theta_2', 'Theta_3', 'Theta_4','Theta_5', 'Theta_6', 'Theta_7', 'Theta_8', 'Inner Cost', 'Outer Cost',
                    'Gradient Cost']
    # parallelisation settings
    ncpu = mp.cpu_count()
    ncores = 12
    # open the file to write to
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        # run the outer optimisation
        for i in range(max_iter):
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
                results = pool.starmap(inner_optimisation, zip(thetas, repeat(times_roi),repeat(voltage_roi), repeat(current_roi), repeat(knots_roi), repeat(states_known_roi), repeat(init_betas_roi)))
            # package results is a list of tuples
            # extract the results
            for iSample, result in enumerate(results):
                betas_sample, inner_costs_sample, state_fitted_at_sample = result
                # get the states at this sample
                list_of_states = [state_values for _, state_values in state_fitted_at_sample.items()]
                state_all_segments = np.array(list_of_states)
                # evaluate the cost functions at the sampled value of ODE parameter vector
                InnerCost = sum(inner_costs_sample)
                OuterCost = error_outer(thetas[iSample,:])
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
            theta_best.append(thetas[index_best,:])
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
            # theta_guessed.append(optimiser_outer.guess())
            # f_guessed.append(optimiser_outer.guesses())
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
                row = [i, iWalker] + list(thetas[iWalker]) + [InnerCosts[iWalker], OuterCosts[iWalker], GradCosts[iWalker]]
                writer.writerow(row)
            file.flush()

            # check for convergence
            if (i > iter_for_convergence):
                    # check how the cost increment changed over the last 10 iterations
                    d_cost = np.diff(f_outer_best[-iter_for_convergence:])
                    # if all incrementa are below a threshold break the loop
                    if all(d<=convergence_threshold for d in d_cost):
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
    print('============================================================================================================')
    ####################################################################################################################
    ## save the best betas as a table to csv file
    df_betas = pd.DataFrame()
    for i, beta in enumerate(init_betas_roi):
        df_betas['segment_'+str(i)] = beta
    df_betas.to_csv('best_betas_both_states.csv', index=False)
    ####################################################################################################################
    # plot optimised model output
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
    ## optimised the following segments
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
        deriv_all_segments = np.array(list_of_derivs)
        list_of_rhs = [rhs_values for _, rhs_values in rhs_fitted_roi.items()]
        rhs_all_segments = np.array(list_of_rhs)
    else:
        state_all_segments = np.array(state_fitted_roi[hidden_state_names])
        deriv_all_segments = np.array(deriv_fitted_roi[hidden_state_names])
        rhs_all_segments = np.array(rhs_fitted_roi[hidden_state_names])
    ## optimised model output
    current_model = g * np.prod(state_all_segments, axis=0) * (voltage - EK)
    # save the model output into a pickle file - in case the plots break again!
    with open(folderName+'/model_output_two_states.pkl', 'wb') as f:
        pkl.dump([times, current_model, state_all_segments, deriv_all_segments, rhs_all_segments], f)
    ####################################################################################################################
    # plot evolution of inner costs
    plt.figure(figsize=(10, 6))
    plt.semilogy()
    plt.xlabel('Iteration')
    plt.ylabel('Inner optimisation cost')
    for iIter in range(len(f_outer_best)-1):
        plt.scatter(iIter*np.ones(len(InnerCosts_all[iIter])),InnerCosts_all[iIter], c='k',marker='.', alpha=.5, linewidths=0)
    iIter += 1
    plt.scatter(iIter * np.ones(len(InnerCosts_all[iIter])), InnerCosts_all[iIter], c='k', marker='.', alpha=.5,
                linewidths=0,label='Sample cost min: J(C / Theta, Y) = '  +"{:.5e}".format(min(InnerCosts_all[iIter])) )
    plt.plot(f_inner_best, '-b', linewidth=1.5,
             label='Best cost:J(C / Theta_{best}, Y) = ' + "{:.5e}".format(
                 f_inner_best[-1]))
    # plt.plot(range(len(f_inner_best)), np.ones(len(f_inner_best)) * InnerCost_given_true_theta, '--m', linewidth=2.5, alpha=.5, label='Collocation solution: J(C / Theta_{true}, Y) = '  +"{:.5e}".format(InnerCost_given_true_theta))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(folderName+'/inner_cost_ask_tell_two_states.png',dpi=400)

    # plot evolution of outer costs
    plt.figure(figsize=(10, 6))
    plt.semilogy()
    plt.xlabel('Iteration')
    plt.ylabel('Outer optimisation cost')
    for iIter in range(len(f_outer_best) - 1):
        plt.scatter(iIter * np.ones(len(OuterCosts_all[iIter])), OuterCosts_all[iIter], c='k', marker='.', alpha=.5,
                    linewidths=0)
    iIter += 1
    plt.scatter(iIter * np.ones(len(OuterCosts_all[iIter])), OuterCosts_all[iIter], c='k', marker='.', alpha=.5,linewidths=0, label='Sample cost: H(Theta / C, Y)')
    # plt.plot(range(iIter), np.ones(iIter) * OuterCost_true, '-m', linewidth=2.5, alpha=.5,label=r'B-splines fit to true state: $H(\Theta \mid  \hat{C}_{direct}, \bar{\mathbf{y}}) = $' + "{:.7f}".format(
    #              OuterCost_true))
    plt.plot(range(len(f_outer_best)), np.ones(len(f_outer_best)) * OuterCost_given_true_theta, '--m', linewidth=2.5, alpha=.5,label='Collocation solution: H(Theta_{true} /  C, Y) = ' + "{:.5e}".format(
                 OuterCost_given_true_theta))
    plt.plot(f_outer_best,'-b',linewidth=1.5,label='Best cost:H(Theta_{best} / C, Y) = ' + "{:.5e}".format(f_outer_best[-1]))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(folderName+'/outer_cost_ask_tell_two_states.png',dpi=400)

    # plot evolution of outer costs
    plt.figure(figsize=(10, 6))
    plt.semilogy()
    plt.xlabel('Iteration')
    plt.ylabel('Gradient matching cost')
    for iIter in range(len(f_gradient_best) - 1):
        plt.scatter(iIter * np.ones(len(GradCost_all[iIter])), GradCost_all[iIter], c='k', marker='.', alpha=.5,
                    linewidths=0)
    iIter += 1
    plt.scatter(iIter * np.ones(len(GradCost_all[iIter])), GradCost_all[iIter], c='k', marker='.', alpha=.5,linewidths=0, label='Sample cost: G_{ODE}(C / Theta, Y)')
    # plt.plot(range(len(f_gradient_best)), np.ones(len(f_gradient_best)) * GradCost_given_true_theta, '--m', linewidth=2.5, alpha=.5,label='Collocation solution: G_{ODE}( C /  Theta_{true}, Y) = ' + "{:.5e}".format(
    #              GradCost_given_true_theta))
    plt.plot(f_gradient_best,'-b',linewidth=1.5,label='Best cost:G_{ODE}(C / Theta, Y) = ' + "{:.5e}".format(f_gradient_best[-1]))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(folderName+'/gradient_cost_ask_tell_two_states.png',dpi=400)

    # plot parameter values after search was done on decimal scale
    fig, axes = plt.subplots(len(theta_true), 1, figsize=(3*len(theta_true), 16), sharex=True)
    for iAx, ax in enumerate(axes.flatten()):
        for iIter in range(len(theta_best)):
            x_visited_iter = theta_visited[iIter][:,iAx]
            ax.scatter(iIter*np.ones(len(x_visited_iter)),x_visited_iter,c='k',marker='.',alpha=.2,linewidth=0)
        # ax.plot(range(iIter+1),np.ones(iIter+1)*theta_true[iAx], '--m', linewidth=2.5,alpha=.5, label=r"true: log("+param_names[iAx]+") = " +"{:.6f}".format(theta_true[iAx]))
        # ax.plot(theta_guessed[:,iAx],'--r',linewidth=1.5,label=r"guessed: $\theta_{"+str(iAx+1)+"} = $" +"{:.4f}".format(theta_guessed[-1,iAx]))
        ax.plot(theta_best[:,iAx],'-b',linewidth=1.5,label=r"best: log("+param_names[iAx]+") = " +"{:.6f}".format(theta_best[-1,iAx]))
        ax.set_ylabel('log('+param_names[iAx]+')')
        ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(folderName+'/ODE_params_log_scale_two_states.png',dpi=400)

    # plot parameter values converting from log scale to decimal
    fig, axes = plt.subplots(len(theta_true), 1, figsize=(3*len(theta_true), 16), sharex=True)
    for iAx, ax in enumerate(axes.flatten()):
        for iIter in range(len(theta_best)):
            x_visited_iter = theta_visited[iIter][:,iAx]
            ax.scatter(iIter*np.ones(len(x_visited_iter)),np.exp(x_visited_iter),c='k',marker='.',alpha=.2,linewidth=0)
        # ax.plot(range(iIter+1),np.ones(iIter+1)*np.exp(theta_true[iAx]), '--m', linewidth=2.5,alpha=.5, label="true: "+param_names[iAx]+" = " +"{:.6f}".format(np.exp(theta_true[iAx])))
        # ax.plot(np.exp(theta_guessed[:,iAx]),'--r',linewidth=1.5,label="guessed: $a_{"+str(iAx+1)+"} = $" +"{:.4f}".format(np.exp(theta_guessed[-1,iAx])))
        ax.plot(np.exp(theta_best[:,iAx]),'-b',linewidth=1.5,label="best: "+param_names[iAx]+" = " +"{:.6f}".format(np.exp(theta_best[-1,iAx])))
        ax.set_ylabel(param_names[iAx])
        ax.set_yscale('log')
        ax.legend(loc='best')
    ax.set_xlabel('Iteration')
    plt.tight_layout()
    plt.savefig(folderName+'/ODE_params_two_states.png',dpi=400)
    ####################################################################################################################
    # plot model outputs given best theta
    # get initial values from the B-spline fit
    x0_optimised_ODE = state_all_segments[:,0]
    # solve ODE with best theta
    solution_optimised_ODE = sp.integrate.solve_ivp(two_state_model, [0,tlim[-1]], x0_optimised_ODE, args=[Thetas_ODE], dense_output=True,method='LSODA',rtol=1e-8,atol=1e-8)
    states_optimised_ODE = solution_optimised_ODE.sol(times)
    RHS_optimised_ODE = two_state_model(times, states_optimised_ODE, Thetas_ODE)
    current_ODE_output = observation(times, states_optimised_ODE, Thetas_ODE)
    # plot model outputs given best theta
    fig, axes = plt.subplot_mosaic([['a)'], ['b)'], ['c)']], layout='constrained',sharex=True)
    y_labels = ['I', 'a', 'r']
    axes['a)'].plot(times, current_true, '-k', label=r'Current true (Kemp model)', linewidth=2, alpha=0.7)
    axes['a)'].plot(times, current_model, '--c', label=r'Current from B-spline approximation')
    axes['a)'].plot(times, current_ODE_output, '--m', label=r'Current from optimised HH ODE output')
    # axes['b)'].plot(times, state_hidden_true[0, :], '-k', label=r'a true', linewidth=2, alpha=0.7)
    axes['b)'].plot(times, state_fitted_roi[state_names[0]], '--c', label=r'B-spline approximation given best theta')
    axes['b)'].plot(times, states_optimised_ODE[0, :], '--m', label=r'HH ODE solution given best theta')
    # axes['c)'].plot(times, state_hidden_true[1, :], '-k', label=r'r true', linewidth=2, alpha=0.7)
    axes['c)'].plot(times, state_fitted_roi[state_names[1]], '--c', label=r'B-spline approximation given best theta')
    axes['c)'].plot(times, states_optimised_ODE[1,:], '--m', label=r'HH ODE solution given best theta')
    iAx = 0
    for _, ax in axes.items():
        ax.set_ylabel(y_labels[iAx], fontsize=12)
        ax.legend(fontsize=12, loc='upper left')
        iAx += 1
    # plt.tight_layout(pad=0.3)
    plt.savefig(folderName+'/model_output_two_states.png', dpi=400)

    # plot errors
    # substract a list from a list

    fig, axes = plt.subplot_mosaic([['a)', 'a)'], ['b)', 'c)'], ['d)', 'e)']], layout='constrained')
    # y_labels = ['$I_{true} - I_{model}$',r'$\dot{a} - RHS(\beta_a)$',r'$\dot{r} - RHS(\beta_r)$',r'$a$ - $\Phi\beta_a$', r'$r$ - $\Phi\beta_r$']
    y_labels = ['I_{true} - I_{model}', 'da(C) - RHS(C)', 'dr(C) - RHS(C)',
                'a - Phi C_a', 'r - Phi C_r']
    axes['a)'].plot(times, current_true - current_model, '-k', label='Data error of B-spline approx.')
    axes['a)'].plot(times, current_true - current_ODE_output, '--c', label='Data error of HH ODE solution')
    axes['b)'].plot(times, deriv_all_segments[0, :] - rhs_all_segments[0, :], '-k', label='Derivative - RHS of B-spline approx.')
    axes['b)'].plot(times, deriv_all_segments[0, :] - RHS_optimised_ODE[0], '--c',
                    label='Derivative - RHS of HH ODE')
    axes['d)'].plot(times, state_hidden_true[0, :] - state_all_segments[0, :], '-k', label='B-spline approximation error')
    axes['d)'].plot(times, state_hidden_true[0, :] - states_optimised_ODE[0, :], '--c', label='HH ODE solution error')
    axes['c)'].plot(times, deriv_all_segments[1, :] - rhs_all_segments[1, :], '-k',
                    label='Derivative - RHS of B-spline approx.')
    axes['c)'].plot(times, deriv_all_segments[1, :] - RHS_optimised_ODE[1], '--c',
                    label='Derivative - RHS of HH ODE')
    axes['e)'].plot(times, state_hidden_true[1, :] - state_all_segments[1, :], '-k',
                    label='B-spline approximation error')
    axes['e)'].plot(times, state_hidden_true[1, :] - states_optimised_ODE[1, :], '--c', label='HH ODE solution error')
    iAx = 0
    for _, ax in axes.items():
        ax.set_ylabel(y_labels[iAx], fontsize=12)
        ax.legend(fontsize=12, loc='upper left')
        iAx += 1
    plt.tight_layout(pad=0.3)
    plt.savefig(folderName+'/erros_ask_tell_two_states.png', dpi=400)
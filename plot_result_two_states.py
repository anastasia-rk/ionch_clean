# imports
import numpy as np
from setup import *
import multiprocessing as mp
from itertools import repeat
import traceback
matplotlib.use('AGG')
# plt.ioff()

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

def observation(t, x, theta):
    # I
    a, r = x[:2]
    ps = theta[:8]
    return g * a * r * (V(t) - EK)
# get Voltage for time in ms
def V(t):
    return volts_interpolated((t)/ 1000)


if __name__ == '__main__':
    #  load the voltage data:
    volts = np.genfromtxt("./protocol-staircaseramp.csv", skip_header=1, dtype=float, delimiter=',')
    #  check when the voltage jumps
    # read the times and valued of voltage clamp
    volt_times, volts = np.genfromtxt("./protocol-staircaseramp.csv", skip_header=1, dtype=float, delimiter=',').T
    # interpolate with smaller time step (milliseconds)
    volts_interpolated = sp.interpolate.interp1d(volt_times, volts, kind='previous')
    ## define the time interval on which the fitting will be done
    tlim = [300, 14899]
    times = np.linspace(*tlim, tlim[-1] - tlim[0], endpoint=False)
    volts_new = V(times)
    ################################################################################################################
    ## find switchpoints
    d2v_dt2 = np.diff(volts_new, n=2)
    dv_dt = np.diff(volts_new)
    der1_nonzero = np.abs(dv_dt) > 1e-1
    der2_nonzero = np.abs(d2v_dt2) > 1e-1
    switchpoints = [a and b for a, b in zip(der1_nonzero, der2_nonzero)]
    # get the times of all jumps - for plotting background in axes:
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
    jumps_odd = jump_indeces[0::2]
    jumps_even = jump_indeces[1::2]
    ################################################################################################################
    ## Generate the synthetic data
    # parameter values for the model
    EK = -80
    thetas_true = [2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524]
    theta_true = np.log(thetas_true[:-1])
    param_names = [f'p_{i}' for i in range(1, len(theta_true) + 1)]
    g = 0.1524
    inLogScale = True
    # initialise and solve ODE
    x0 = [0, 1]
    state_names = ['a', 'r']
    # solve initial value problem
    solution = sp.integrate.solve_ivp(hh_model, [0, tlim[-1]], x0, args=[thetas_true], dense_output=True,
                                      method='LSODA', rtol=1e-8, atol=1e-8)
    state_hidden_true = solution.sol(times)
    current_true = observation(times, state_hidden_true, thetas_true)
    voltage = V(times)
    ################################################################################################################
    lambd = 1000000  # 0.3 # 0 # 1
    folderName = 'Results_two_state_lambda_' + str(int(lambd))
    folderForOutput = 'Test_plot_output'
    with open(folderName + '/model_output_two_states.pkl', 'rb') as f:
        # load the model output
        model_output = pkl.load(f)
    times, current_model, state_all_segments, deriv_all_segments, rhs_all_segments = model_output
    # current_model = g * np.prod(state_all_segments, axis=0) * (voltage - EK)
    ####################################################################################################################
    # load the optimisation metrix from the csv file in the folderName directory
    df = pd.read_csv(folderName + '/iterations_both_states.csv')
    # load best betas from the csv file in the folderName directory
    df_betas = pd.read_csv(folderName + '/iterations_both_states.csv')
    ################################################################################################################
    ### remake all the lists for plotting from the dataframe
    # create a list of InnerCosts_all from the dataframe by grouping entries in Inner Cost column by iteration
    InnerCosts_all = [group["Inner Cost"].tolist() for i, group in df.groupby("Iteration")]
    # create a list of OuterCosts_all from the dataframe by grouping entries in Outer Cost column by iteration
    OuterCosts_all = [group["Outer Cost"].tolist() for i, group in df.groupby("Iteration")]
    # create a list of GradCosts_all from the dataframe by grouping entries in Gradient cost column by iteration
    GradCosts_all = [group["Gradient Cost"].tolist() for i, group in df.groupby("Iteration")]
    # create a list of thetas from the dataframe by grouping entries in columns Theta_1 to Theta_8 by iteration
    theta_visited = [group[["Theta_" + str(i) for i in range(1, 1+len(theta_true))]].values.tolist() for i, group in
                     df.groupby("Iteration")]
    # get a number of iterations
    nIter = len(df["Iteration"].unique())
    theta_best = []
    f_outer_best = []
    f_inner_best = []
    f_gradient_best = []
    for iIter in range(nIter):
        OuterCosts = OuterCosts_all[iIter]
        InnerCosts = InnerCosts_all[iIter]
        GradCosts = GradCosts_all[iIter]
        index_best = OuterCosts.index(np.nanmin(OuterCosts))
        theta_best.append(theta_visited[iIter][index_best][:])
        f_outer_best.append(OuterCosts[index_best])
        f_inner_best.append(InnerCosts[index_best])
        f_gradient_best.append(GradCosts[index_best])
    theta_best = np.array(theta_best)
    f_outer_best = np.array(f_outer_best)
    f_inner_best = np.array(f_inner_best)
    f_gradient_best = np.array(f_gradient_best)
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
                linewidths=0,
                label='Sample cost min: J(C / Theta, Y) = ' + "{:.5e}".format(min(InnerCosts_all[iIter])))
    plt.plot(f_inner_best, '-b', linewidth=1.5,
             label='Best cost:J(C / Theta_{best}, Y) = ' + "{:.5e}".format(
                 f_inner_best[-1]))
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
    # plt.plot(range(len(f_inner_best)), np.ones(len(f_inner_best)) * InnerCost_given_true_theta, '--m',
    #          linewidth=2.5, alpha=.5,
    #          label='Collocation solution: J(C / Theta_{true}, Y) = ' + "{:.5e}".format(InnerCost_given_true_theta))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(folderForOutput + '/inner_cost_ask_tell_two_states.png', dpi=400)

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
    # plt.plot(range(len(f_outer_best)), np.ones(len(f_outer_best)) * OuterCost_given_true_theta, '--m',
    #          linewidth=2.5, alpha=.5, label='Collocation solution: H(Theta_{true} /  C, Y) = ' + "{:.5e}".format(
    #         OuterCost_given_true_theta))
    plt.plot(f_outer_best, '-b', linewidth=1.5,
             label='Best cost:H(Theta_{best} / C, Y) = ' + "{:.5e}".format(f_outer_best[-1]))
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
    plt.legend(loc='best')
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
    plt.tight_layout()
    plt.savefig(folderForOutput + '/outer_cost_ask_tell_two_states.png', dpi=400)

    # plot evolution of outer costs
    plt.figure(figsize=(10, 6))
    plt.semilogy()
    plt.xlabel('Iteration')
    plt.ylabel('Gradient matching cost')
    for iIter in range(len(f_gradient_best) - 1):
        plt.scatter(iIter * np.ones(len(GradCosts_all[iIter])), GradCosts_all[iIter], c='k', marker='.', alpha=.5,
                    linewidths=0)
    iIter += 1
    plt.scatter(iIter * np.ones(len(GradCosts_all[iIter])), GradCosts_all[iIter], c='k', marker='.', alpha=.5,
                linewidths=0, label='Sample cost: G_{ODE}(C / Theta, Y)')
    # plt.plot(range(len(f_gradient_best)), np.ones(len(f_gradient_best)) * GradCost_given_true_theta, '--m',
    #          linewidth=2.5, alpha=.5,
    #          label='Collocation solution: G_{ODE}( C /  Theta_{true}, Y) = ' + "{:.5e}".format(
    #              GradCost_given_true_theta))
    plt.plot(f_gradient_best, '-b', linewidth=1.5,
             label='Best cost:G_{ODE}(C / Theta, Y) = ' + "{:.5e}".format(f_gradient_best[-1]))
    plt.legend(loc='best')
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
    plt.tight_layout()
    plt.savefig(folderForOutput + '/gradient_cost_ask_tell_two_states.png', dpi=400)

    # plot parameter values after search was done on decimal scale
    fig, axes = plt.subplots(len(theta_true), 1, figsize=(3 * len(theta_true), 16), sharex=True)
    for iAx, ax in enumerate(axes.flatten()):
        for iIter in range(len(theta_best)):
            x_visited_iter = [theta_visited[iIter][i][iAx] for i in range(len(theta_visited[iIter]))]
            ax.scatter(iIter * np.ones(len(x_visited_iter)), x_visited_iter, c='k', marker='.', alpha=.2,
                       linewidth=0)
        # ax.plot(range(iIter + 1), np.ones(iIter + 1) * theta_true[iAx], '--m', linewidth=2.5, alpha=.5,
        #         label=r"true: log(" + param_names[iAx] + ") = " + "{:.6f}".format(theta_true[iAx]))
        # ax.plot(theta_guessed[:,iAx],'--r',linewidth=1.5,label=r"guessed: $\theta_{"+str(iAx+1)+"} = $" +"{:.4f}".format(theta_guessed[-1,iAx]))
        ax.plot(theta_best[:, iAx], '-b', linewidth=1.5,
                label=r"best: log(" + param_names[iAx] + ") = " + "{:.6f}".format(theta_best[-1, iAx]))
        ax.set_ylabel('log(' + param_names[iAx] + ')')
        ax.legend(loc='best')
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
    plt.tight_layout()
    plt.savefig(folderForOutput + '/ODE_params_log_scale_two_states.png', dpi=400)

    # plot parameter values converting from log scale to decimal
    fig, axes = plt.subplots(len(theta_true), 1, figsize=(3 * len(theta_true), 16), sharex=True)
    for iAx, ax in enumerate(axes.flatten()):
        for iIter in range(len(theta_best)):
            x_visited_iter = [theta_visited[iIter][i][iAx] for i in range(len(theta_visited[iIter]))]
            ax.scatter(iIter * np.ones(len(x_visited_iter)), np.exp(x_visited_iter), c='k', marker='.', alpha=.2,
                       linewidth=0)
        # ax.plot(range(iIter + 1), np.ones(iIter + 1) * np.exp(theta_true[iAx]), '--m', linewidth=2.5, alpha=.5,
        #         label="true: " + param_names[iAx] + " = " + "{:.6f}".format(np.exp(theta_true[iAx])))
        # ax.plot(np.exp(theta_guessed[:,iAx]),'--r',linewidth=1.5,label="guessed: $a_{"+str(iAx+1)+"} = $" +"{:.4f}".format(np.exp(theta_guessed[-1,iAx])))
        ax.plot(np.exp(theta_best[:, iAx]), '-b', linewidth=1.5,
                label="best: " + param_names[iAx] + " = " + "{:.6f}".format(np.exp(theta_best[-1, iAx])))
        ax.set_ylabel(param_names[iAx])
        ax.set_yscale('log')
        ax.legend(loc='best')
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
    ax.set_xlabel('Iteration')
    plt.tight_layout()
    plt.savefig(folderForOutput + '/ODE_params_two_states.png', dpi=400)
    ####################################################################################################################
    ## plot model outputs given best theta
    # ge the best theta value
    Thetas_ODE = theta_best[-1]
    # get initial values from the B-spline fit
    x0_optimised_ODE = state_all_segments[:, 0]
    # solve ODE with best theta
    solution_optimised_ODE = sp.integrate.solve_ivp(two_state_model, [0, tlim[-1]], x0_optimised_ODE,
                                                    args=[Thetas_ODE], dense_output=True, method='LSODA', rtol=1e-8,
                                                    atol=1e-8)
    states_optimised_ODE = solution_optimised_ODE.sol(times)
    RHS_optimised_ODE = two_state_model(times, states_optimised_ODE, Thetas_ODE)
    current_ODE_output = observation(times, states_optimised_ODE, Thetas_ODE)
    # plot model outputs given best theta
    fig, axes = plt.subplot_mosaic([['a)'], ['b)'], ['c)']], layout='constrained', sharex=True)
    y_labels = ['I', 'a', 'r']
    # add segment shading to all axes
    for _, ax in axes.items():
        for iSegment, SegmentStart in enumerate(jumps_odd):
            ax.axvspan(times[SegmentStart], times[jumps_even[iSegment]], facecolor='0.2', alpha=0.2)
    axes['a)'].plot(times, current_true, '-k', label=r'Current true (HH model)', linewidth=2, alpha=0.7)
    axes['a)'].plot(times, current_model, '--c', label=r'Current from B-spline approximation')
    axes['a)'].plot(times, current_ODE_output, '--m', label=r'Current from optimised HH ODE output')
    axes['b)'].plot(times, state_hidden_true[0, :], '-k', label=r'a true', linewidth=2, alpha=0.7)
    axes['b)'].plot(times, state_all_segments[0,:], '--c',
                    label=r'B-spline approximation given best theta')
    axes['b)'].plot(times, states_optimised_ODE[0, :], '--m', label=r'HH ODE solution given best theta')
    axes['c)'].plot(times, state_hidden_true[1, :], '-k', label=r'r true', linewidth=2, alpha=0.7)
    axes['c)'].plot(times, state_all_segments[1,:], '--c',
                    label=r'B-spline approximation given best theta')
    axes['c)'].plot(times, states_optimised_ODE[1, :], '--m', label=r'HH ODE solution given best theta')
    iAx = 0
    for _, ax in axes.items():
        ax.set_ylabel(y_labels[iAx], fontsize=12)
        ax.legend(fontsize=12, loc='upper left')
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
        ax.set_xlim(times[jump_indeces[0]],times[jump_indeces[-1]])
        iAx += 1
    # plt.tight_layout(pad=0.3)
    plt.savefig(folderForOutput + '/model_output_two_states.png', dpi=400)

    # plot errors
    # substract a list from a list

    fig, axes = plt.subplot_mosaic([['a)', 'a)'], ['b)', 'c)'], ['d)', 'e)']], layout='constrained')
    # y_labels = ['$I_{true} - I_{model}$',r'$\dot{a} - RHS(\beta_a)$',r'$\dot{r} - RHS(\beta_r)$',r'$a$ - $\Phi\beta_a$', r'$r$ - $\Phi\beta_r$']
    y_labels = ['I_{true} - I_{model}', 'da(C) - RHS(C)', 'dr(C) - RHS(C)',
                'a - Phi C_a', 'r - Phi C_r']
    # add segment shading to all axes
    for _, ax in axes.items():
        for iSegment, SegmentStart in enumerate(jumps_odd):
            ax.axvspan(times[SegmentStart], times[jumps_even[iSegment]], facecolor='0.2', alpha=0.2)
    #  plot stuff
    axes['a)'].plot(times, current_true - current_model, '-k', label='Data error of B-spline approx.')
    axes['a)'].plot(times, current_true - current_ODE_output, '--c', label='Data error of HH ODE solution')
    axes['b)'].plot(times, deriv_all_segments[0, :] - rhs_all_segments[0, :], '-k',
                    label='Derivative - RHS of B-spline approx.')
    axes['b)'].plot(times, deriv_all_segments[0, :] - RHS_optimised_ODE[0], '--c',
                    label='Derivative - RHS of HH ODE.')
    axes['d)'].plot(times, state_hidden_true[0, :] - state_all_segments[0, :], '-k',
                    label='B-spline approximation error')
    axes['d)'].plot(times, state_hidden_true[0, :] - states_optimised_ODE[0, :], '--c',
                    label='HH ODE solution error')
    axes['c)'].plot(times, deriv_all_segments[1, :] - rhs_all_segments[1, :], '-k',
                    label='Derivative - RHS of B-spline approx.')
    axes['c)'].plot(times, deriv_all_segments[1, :] - RHS_optimised_ODE[1], '--c',
                    label='Derivative - RHS of HH ODE.')
    axes['e)'].plot(times, state_hidden_true[1, :] - state_all_segments[1, :], '-k',
                    label='B-spline approximation error')
    axes['e)'].plot(times, state_hidden_true[1, :] - states_optimised_ODE[1, :], '--c',
                    label='HH ODE solution error')
    iAx = 0
    for _, ax in axes.items():
        ax.set_ylabel(y_labels[iAx], fontsize=12)
        ax.legend(fontsize=12, loc='upper left')
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
        ax.set_xlim(times[jump_indeces[0]],times[jump_indeces[-1]])
        iAx += 1
    plt.tight_layout(pad=0.3)
    plt.savefig(folderForOutput + '/erros_ask_tell_two_states.png', dpi=400)

    # run the validation protocols on the models
    # check the AP voltage protocol for validation
    ap_protocol_data = np.genfromtxt('ap_protocol/ap.csv', delimiter=',', skip_header=1)
    # get the time and voltage
    time_ap, voltage_ap = ap_protocol_data.T
    # these voltage values will now be interpolated
    time_ap_sec = time_ap/1000
    volts_interpolated = sp.interpolate.interp1d(time_ap_sec, voltage_ap, kind='previous')
    # part 1: rerun the model for the new voltage protocol
    Thetas_ODE = theta_best[-1]
    # get initial values from the B-spline fit
    x0_optimised_ODE = state_all_segments[:, 0]
    # solve ODE with best theta
    solution_optimised_ODE = sp.integrate.solve_ivp(two_state_model, [0, time_ap[-1]], x0_optimised_ODE,
                                                    args=[Thetas_ODE], dense_output=True, method='LSODA', rtol=1e-8,
                                                    atol=1e-8)
    states_optimised_ODE = solution_optimised_ODE.sol(time_ap)
    RHS_optimised_ODE = two_state_model(time_ap, states_optimised_ODE, Thetas_ODE)
    current_ODE_output = observation(time_ap, states_optimised_ODE, Thetas_ODE)

    #part 2: rerun the model with true values on the new protocol
    Thetas_ODE = theta_true
    solution_true_ODE = sp.integrate.solve_ivp(two_state_model, [0, time_ap[-1]], x0,
                                                    args=[Thetas_ODE], dense_output=True, method='LSODA', rtol=1e-8,
                                                    atol=1e-8)
    states_true_ODE = solution_optimised_ODE.sol(time_ap)
    RHS_true_ODE = two_state_model(time_ap, states_true_ODE, Thetas_ODE)
    current_true = observation(time_ap, states_true_ODE, Thetas_ODE)

    # plot outputs to compare
    fig, axes = plt.subplot_mosaic([['a)'], ['b)'], ['c)']], layout='constrained', sharex=True)
    y_labels = ['I', 'a', 'r']
    # # add segment shading to all axes
    # for _, ax in axes.items():
    #     for iSegment, SegmentStart in enumerate(jumps_odd):
    #         ax.axvspan(time_ap[SegmentStart], time_ap[jumps_even[iSegment]], facecolor='0.2', alpha=0.2)
    axes['a)'].plot(time_ap, current_true, '-k', label=r'Current true (HH model)', linewidth=2, alpha=0.7)
    axes['a)'].plot(time_ap, current_ODE_output, '--m', label=r'Current from optimised HH ODE output')
    axes['b)'].plot(time_ap, states_true_ODE[0, :], '-k', label=r'a true', linewidth=2, alpha=0.7)
    axes['b)'].plot(time_ap, states_optimised_ODE[0, :], '--m', label=r'HH ODE solution given best theta')
    axes['c)'].plot(time_ap, states_true_ODE[1, :], '-k', label=r'r true', linewidth=2, alpha=0.7)
    axes['c)'].plot(time_ap, states_optimised_ODE[1, :], '--m', label=r'HH ODE solution given best theta')
    iAx = 0
    for _, ax in axes.items():
        ax.set_ylabel(y_labels[iAx], fontsize=12)
        ax.legend(fontsize=12, loc='upper left')
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
        # ax.set_xlim(times[jump_indeces[0]],times[jump_indeces[-1]])
        iAx += 1
    # plt.tight_layout(pad=0.3)
    plt.savefig(folderForOutput + '/validation_output_two_states.png', dpi=400)

    # plot errors
    fig, axes = plt.subplot_mosaic([['a)', 'a)'], ['b)', 'c)'], ['d)', 'e)']], layout='constrained')
    y_labels = ['I_{true} - I_{model}', 'true RHS - model RHS', 'true RHS - model RHS',
                'true a - model a', 'true r - model r']
    axes['a)'].plot(time_ap, current_true - current_ODE_output, '--k', label='Data error of HH ODE solution')
    axes['b)'].plot(time_ap, RHS_true_ODE[0] - RHS_optimised_ODE[0], '--k',
                    label='RHS true - RHS of HH ODE.')
    axes['c)'].plot(time_ap, RHS_true_ODE[1] - RHS_optimised_ODE[1], '--k',
                    label='RHS true - RHS of HH ODE.')
    axes['d)'].plot(time_ap, states_true_ODE[0, :] - states_optimised_ODE[0, :], '--k',
                    label='HH ODE solution error')
    axes['e)'].plot(time_ap, states_true_ODE[1, :] - states_optimised_ODE[1, :], '--k',
                    label='HH ODE solution error')
    iAx = 0
    for _, ax in axes.items():
        ax.set_ylabel(y_labels[iAx], fontsize=12)
        ax.legend(fontsize=12, loc='upper left')
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
        # ax.set_xlim(times[jump_indeces[0]],times[jump_indeces[-1]])
        iAx += 1
    # plt.tight_layout(pad=0.3)
    plt.savefig(folderForOutput + '/validation_errors_two_states.png', dpi=400)

## end of loop over lambda values
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

if __name__ == '__main__':
    #  load the voltage data:
    volts = np.genfromtxt("./protocol-staircaseramp.csv", skip_header=1, dtype=float, delimiter=',')
    #  check when the voltage jumps
    # read the times and valued of voltage clamp
    volt_times, volts = np.genfromtxt("./protocol-staircaseramp.csv", skip_header=1, dtype=float, delimiter=',').T
    # interpolate with smaller time step (milliseconds)
    volts_intepolated = sp.interpolate.interp1d(volt_times, volts, kind='previous')

    ## define the time interval on which the fitting will be done
    tlim = [4500, 11300]
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
    rhs_true = ion_channel_model_one_state(times, state_hidden_true,theta_true)
    lambd = 1 # 0.3 # 0 # 1
    folderName = 'Results_' + state_name + '_lambda_' + str(int(lambd))
    with open(folderName + '/model_output_' + state_name + '.pkl', 'rb') as f:
        # load the model output
        model_output = pkl.load(f)
    # times, current_model, state_all_segments, deriv_all_segments, rhs_all_segments = model_output
    times, current_model, state_all_segments, deriv_all_segments, rhs_all_segments, InnerCost_given_true_theta, OuterCost_given_true_theta, GradCost_given_true_theta = model_output
    # load the optimisation metrix from the csv file in the folderName directory
    df = pd.read_csv(folderName + '/iterations_one_state_' + state_name + '.csv')
    # load best betas from the csv file in the folderName directory
    df_betas = pd.read_csv(folderName + '/best_betas_' + state_name + '.csv')
    ################################################################################################################
    ### remake all the lists for plotting from the dataframe
    # create a list of InnerCosts_all from the dataframe by grouping entries in Inner Cost column by iteration
    InnerCosts_all = [group["Inner Cost"].tolist() for i, group in df.groupby("Iteration")]
    # create a list of OuterCosts_all from the dataframe by grouping entries in Outer Cost column by iteration
    OuterCosts_all = [group["Outer Cost"].tolist() for i, group in df.groupby("Iteration")]
    # create a list of GradCosts_all from the dataframe by grouping entries in Gradient cost column by iteration
    GradCosts_all = [group["Gradient Cost"].tolist() for i, group in df.groupby("Iteration")]
    # create a list of thetas from the dataframe by grouping entries in columns Theta_1 to Theta_4 by iteration
    theta_visited = [group[["Theta_" + str(i) for i in range(1, 5)]].values.tolist() for i, group in  df.groupby("Iteration")]
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
                linewidths=0, label='Sample cost min: J(C / Theta, Y) = ' + "{:.5e}".format(np.nanmin(InnerCosts_all[iIter])))
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
        plt.scatter(iIter * np.ones(len(GradCosts_all[iIter])), GradCosts_all[iIter], c='k', marker='.', alpha=.5,
                    linewidths=0)
    iIter += 1
    plt.scatter(iIter * np.ones(len(GradCosts_all[iIter])), GradCosts_all[iIter], c='k', marker='.', alpha=.5,
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
            #  use list comprehersion to get the iAx-th element of each theta_visited[iIter]
            x_visited_iter = [theta_visited[iIter][i][iAx] for i in range(len(theta_visited[iIter]))]
            # x_visited_iter = theta_visited[iIter][:][iAx]
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
            x_visited_iter = [theta_visited[iIter][i][iAx] for i in range(len(theta_visited[iIter]))]
            # x_visited_iter = theta_visited[iIter][:, iAx]
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
    ####################################################################################################################
from inner_optimisation_scipy import *
import inner_optimisation_scipy
# from generate_data import *
from pints_classes import *
import pandas as pd
import pickle as pkl
# get the colours out
import matplotlib.colors as mcolors

# definitions
# set up variables for the simulation
tlim = [300, 14899]
times = np.linspace(*tlim, tlim[-1] - tlim[0], endpoint=False)
voltage = V(times)  # must read voltage at the correct times to match the output
del tlim
model_name = 'Kemp' # this is the generative model name, can be HH or Kemp
snr_db = 20 # signal to noise ratio in dB
## set up the parameters for the fitted model
fitted_model = hh_model
Thetas_ODE = thetas_hh_baseline
state_names = ['a', 'r'] # how many states we have in the model that we are fitting
# outer optimisation settings
inLogScale = True  # is the search of thetas in log scale
# set the exponents
lambda_exps = [7]  # gradient matching weight - test
# get the colours for each lambda to plot all results in one figure
colours = list(mcolors.TABLEAU_COLORS.values())
colours = colours + colours
colours = colours[:len(lambda_exps)]
####################################################################################################################
### from this point no user changes are required
####################################################################################################################
# load the protocols
load_protocols
voltage = V(times)
# generate the segments with B-spline knots and intialise the betas for  - we won't actully need knots for the optimisation
jump_indeces, times_roi, voltage_roi, knots_roi, collocation_roi, spline_order = generate_knots(times)
jumps_odd = jump_indeces[0::2]
jumps_even = jump_indeces[1::2]
nSegments = len(jump_indeces[:-1])
print('The time axis is split into ' + str(nSegments) + ' segments based on protocol steps.')
nBsplineCoeffs = (len(knots_roi[0]) - spline_order - 1) * len(state_names)
init_betas_roi = nSegments * [0.5 * np.ones(nBsplineCoeffs)]
print('Number of B-spline coeffs per segment: ' + str(nBsplineCoeffs) +'.')

# generate a solution here - we dont need it, just to resolve dependency in data splitting for now
if model_name.lower() not in available_models:
    raise ValueError(f'Unknown model name: {model_name}. Available models are: {available_models}.')
elif model_name.lower() == 'hh':
    thetas_true = thetas_hh_baseline
elif model_name.lower() == 'kemp':
    thetas_true = thetas_kemp
elif model_name.lower() == 'wang':
    thetas_true = thetas_wang

if __name__ == '__main__':
    # prepare all figures in axes
    ## for the states and outputs
    fig, axes = plt.subplot_mosaic([['a)', 'b)'], ['c)', 'd)'], ['e)', 'f)']], layout='constrained', sharex=True, figsize=(15, 6))
    y_labels = ['$I_{collocation}$', '$I_{fitted}$', '$a_{collocation}$', '$a_{fitted}$', '$r_{collocation}$', '$r_{fitted}$']
    for _, ax in axes.items():
        for iSegment, SegmentStart in enumerate(jumps_odd):
            ax.axvspan(times[SegmentStart], times[jumps_even[iSegment]], facecolor='0.2', alpha=0.1)
    # state errors
    fig1, axes1 = plt.subplot_mosaic([['a)', 'b)'], ['c)', 'd)'], ['e)', 'f)']], layout='constrained', sharex=True, figsize=(15, 6))
    y_labels1 = ['$I_{true} - I_{collocation}$', '$I_{true} - I_{fitted}$', 'Gradient error: $da(B) - RHS(B)$', 'Gradient error: $dr(B) - RHS(B)$',
                 'Collocation  error: $a - Phi B_a$', 'Collocation error: $r - Phi B_r$']
    for _, ax in axes1.items():
        for iSegment, SegmentStart in enumerate(jumps_odd):
            ax.axvspan(times[SegmentStart], times[jumps_even[iSegment]], facecolor='0.2', alpha=0.1)

    # plot parameter values in log scale
    fig2, axes2 = plt.subplots(len(Thetas_ODE), 1, figsize=(3 * len(Thetas_ODE), 16), sharex=True)
    # plot parameter values in decimal scale
    fig3, axes3 = plt.subplots(len(Thetas_ODE), 1, figsize=(3 * len(Thetas_ODE), 16), sharex=True)
    ####################################################################################################################
    # iterate over lambdas
    counter = 0 # for plotting current only once and colours of plots
    for weight in lambda_exps:
        lambd = 10 ** weight
        # load the folders
        folderName = 'Results_gen_model_'+model_name+'_lambda_' + str(int(lambd))
        folderForOutput = 'Test_plot_output'
        with open(folderName + '/synthetic_data.pkl', 'rb') as f:
            # load the model output
            model_output = pkl.load(f)
        times, voltage, current_true, states_true, thetas_true, knots_roi, snr_db = model_output
        # send stuff into inner optimisation
        inner_optimisation_scipy.voltage = voltage  # send the voltage to the inner optimisation module
        inner_optimisation_scipy.current_true = current_true  # send the current to the inner optimisation module
        # plot true current - only once is fine!!
        if counter == 0:
            axes['a)'].plot(times, current_true, '--k', label=r'True current', linewidth=1.5, alpha=0.27)
            axes['b)'].plot(times, current_true, '--k', label=r'True current', linewidth=1.5, alpha=0.27)
        ####################################################################################################################
        # load the optimisation metrix from the csv file in the folderName directory
        df = pd.read_csv(folderName + '/iterations_both_states.csv')
        # get the number of thetas for the fitted model
        columnNames = df.columns
        nThetas  = sum('Theta_' in s for s in columnNames)
        ####################################################################################################################
        ### remake all the lists for plotting from the dataframe
        # create a list of InnerCosts_all from the dataframe by grouping entries in Inner Cost column by iteration
        InnerCosts_all = [group["Inner Cost"].tolist() for i, group in df.groupby("Iteration")]
        # create a list of OuterCosts_all from the dataframe by grouping entries in Outer Cost column by iteration
        OuterCosts_all = [group["Outer Cost"].tolist() for i, group in df.groupby("Iteration")]
        # create a list of GradCosts_all from the dataframe by grouping entries in Gradient cost column by iteration
        GradCosts_all = [group["Gradient Cost"].tolist() for i, group in df.groupby("Iteration")]
        # create a list of thetas from the dataframe by grouping entries in columns Theta_1 to Theta_8 by iteration
        theta_visited = [group[["Theta_" + str(i) for i in range(nThetas)]].values.tolist() for i, group in
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
        ####################################################################################################################
        # generate solution - we dont need it beyond sending it into data splitting
        solution, current_model = generate_synthetic_data(model_name, thetas_true, times)
        # set variables definitions for inner optimisation module
        inner_optimisation_scipy.nSegments = nSegments
        inner_optimisation_scipy.state_names = state_names
        inner_optimisation_scipy.spline_order = spline_order
        inner_optimisation_scipy.nBsplineCoeffs = nBsplineCoeffs
        inner_optimisation_scipy.fitted_model = fitted_model
        states_roi, states_known_roi, current_roi = split_generated_data_into_segments(solution, current_true,
                                                                                       jump_indeces, times)
        ## simulate the optimised model using B-splines
        Thetas_ODE = theta_best[-1]
        param_names = [f'p_{i}' for i in range(1, len(Thetas_ODE) + 1)]
        # dont forget to convert to decimal scale if the search was done in log scale
        if inLogScale:
            # convert thetas to decimal scale for inner optimisation
            Thetas_ODE = np.exp(Thetas_ODE)
        else:
            Thetas_ODE = Thetas_ODE
        test_output = inner_optimisation(Thetas_ODE, lambd, times_roi, voltage_roi, current_roi, knots_roi,
                                         collocation_roi)
        betas_sample, inner_cost_sample, data_cost_sample, grad_cost_sample, state_fitted_at_sample = test_output
        state_all_segments = np.array(state_fitted_at_sample)
        current_all_segments = observation_direct_input(state_all_segments, voltage, Thetas_ODE)
        # get the derivative and the RHS
        rhs_of_roi = {key: [] for key in state_names}
        deriv_of_roi = {key: [] for key in state_names}
        for iSegment in range(nSegments):
            model_output_fit = simulate_segment(betas_sample[iSegment], times_roi[iSegment], knots_roi[iSegment],
                                                first_spline_coeffs=None)
            state_at_sample, state_deriv_at_sample, rhs_at_sample = np.split(model_output_fit, 3, axis=1)
            if iSegment == 0:
                index_start = 0  # from which timepoint to store the states
            else:
                index_start = 1  # from which timepoint to store the states
            for iState, stateName in enumerate(state_names):
                deriv_of_roi[stateName] += list(state_deriv_at_sample[index_start:, iState])
                rhs_of_roi[stateName] += list(rhs_at_sample[index_start:, iState])
        ## end of loop over segments
        ## simulate the model using the best thetas and the ODE model used
        x0_optimised_ODE = state_all_segments[:, 0]
        solution_optimised = sp.integrate.solve_ivp(fitted_model, [0, times[-1]], x0_optimised_ODE, args=[Thetas_ODE],
                                                    dense_output=True, method='LSODA', rtol=1e-8, atol=1e-8)
        states_optimised_ODE = solution_optimised.sol(times)
        current_optimised_ODE = observation_direct_input(states_optimised_ODE, voltage, Thetas_ODE)
        ####################################################################################################################
        # add states to the first figure- compare the modelled current with the true current
        axes['a)'].plot(times, current_all_segments, '-', color = colours[counter],
                        label=r'$\lambda = $' + "{:.2e}".format(lambd), linewidth=1, alpha=0.7)
        axes['b)'].plot(times, current_optimised_ODE, '-', color = colours[counter],
                        label=r'$\lambda = $' + "{:.2e}".format(lambd), linewidth=1, alpha=0.7)
        # axes['a)'].set_xlim(times_of_segments[0], times_of_segments[-1])
        # axes['a)'].set_xlim(1890, 1920)
        axes['c)'].plot(times, state_all_segments[0, :], '-', color = colours[counter],
                        label=r'$\lambda$ = ' +  "{:.2e}".format(lambd),
                        linewidth=1, alpha=0.7)
        axes['e)'].plot(times, state_all_segments[1, :], '-', color = colours[counter],
                        label=r'$\lambda$ = ' +  "{:.2e}".format(lambd),
                        linewidth=1, alpha=0.7)
        axes['d)'].plot(times, states_optimised_ODE[0, :], color = colours[counter],
                        label=r'$\lambda$ = ' +  "{:.2e}".format(lambd),
                        linewidth=1, alpha=0.7)
        axes['f)'].plot(times, states_optimised_ODE[1, :], color = colours[counter],
                        label=r'$\lambda$ = ' +  "{:.2e}".format(lambd),
                        linewidth=1, alpha=0.7)
        # plot errors
        axes1['a)'].plot(times, current_all_segments - current_true, '--', color = colours[counter],
                    label=r'$\lambda=$ ' +  "{:.2e}".format(lambd),linewidth=1, alpha=0.7)
        axes1['b)'].plot(times, current_optimised_ODE - current_true, '--',color = colours[counter],
                         label=r'$\lambda=$' +  "{:.2e}".format(lambd), linewidth=1, alpha=0.7)
        axes1['c)'].plot(times, np.array(rhs_of_roi[state_names[0]]) - np.array(deriv_of_roi[state_names[0]]),
                         '--', color = colours[counter],label=r'$\lambda$ = ' +  "{:.2e}".format(lambd),
                         linewidth=1,alpha=0.7)
        axes1['d)'].plot(times, np.array(rhs_of_roi[state_names[1]]) - np.array(deriv_of_roi[state_names[1]]),
                         '--', color = colours[counter],label=r'$\lambda$ = ' +  "{:.2e}".format(lambd),
                         linewidth=1, alpha=0.7)
        axes1['e)'].plot(times, state_all_segments[0, :] - states_optimised_ODE[0, :], '--',color = colours[counter],
                         label=r'$\lambda$ = ' +  "{:.2e}".format(lambd), linewidth=1, alpha=0.7)
        axes1['f)'].plot(times, state_all_segments[1, :] - states_optimised_ODE[1, :], '--',color = colours[counter],
                         label=r'$\lambda$ = ' +  "{:.2e}".format(lambd), linewidth=1, alpha=0.7)

        # plot parameter values after search was done on decimal scale
        fig2, axes2 = plt.subplots(len(Thetas_ODE), 1, figsize=(3 * len(Thetas_ODE), 16), sharex=True)
        for iAx, ax in enumerate(axes2.flatten()):
            for iIter in range(len(theta_best)):
                x_visited_iter = [theta_visited[iIter][i][iAx] for i in range(len(theta_visited[iIter]))]
                ax.scatter(iIter * np.ones(len(x_visited_iter)), x_visited_iter, c=colours[counter], marker='.', alpha=.1,
                           linewidth=0)
            ax.plot(theta_best[:, iAx], '-', color = colours[counter], linewidth=1.5,
                    label=r"best: log(" + param_names[iAx] + ") = " + "{:.6f}".format(theta_best[-1, iAx]))
            ax.set_ylabel('log(' + param_names[iAx] + ')')
            ax.legend(loc='best')

        # plot parameter values converting from log scale to decimal
        fig3, axes3 = plt.subplots(len(Thetas_ODE), 1, figsize=(3 * len(Thetas_ODE), 16), sharex=True)
        for iAx, ax in enumerate(axes3.flatten()):
            for iIter in range(len(theta_best)):
                x_visited_iter = [theta_visited[iIter][i][iAx] for i in range(len(theta_visited[iIter]))]
                ax.scatter(iIter * np.ones(len(x_visited_iter)), np.exp(x_visited_iter), c=colours[counter], marker='.', alpha=.1,
                           linewidth=0)
            ax.plot(np.exp(theta_best[:, iAx]), '-',color = colours[counter], linewidth=1.5,
                    label="best: " + param_names[iAx] + " = " + "{:.6f}".format(np.exp(theta_best[-1, iAx])))
            ax.set_ylabel(param_names[iAx])
            ax.set_yscale('log')
            ax.legend(loc='best')
            ax.set_xlabel('Iteration')

            # ####################################################################################################################
            # # plot evolution of inner costs
            # plt.figure(figsize=(10, 6))
            # plt.semilogy()
            # plt.xlabel('Iteration')
            # plt.ylabel('Inner optimisation cost')
            # for iIter in range(len(f_outer_best) - 1):
            #     plt.scatter(iIter * np.ones(len(InnerCosts_all[iIter])), InnerCosts_all[iIter], c='k', marker='.',
            #                 alpha=.5,
            #                 linewidths=0)
            # iIter += 1
            # plt.scatter(iIter * np.ones(len(InnerCosts_all[iIter])), InnerCosts_all[iIter], c='k', marker='.', alpha=.5,
            #             linewidths=0,
            #             label='Sample cost min: J(C / Theta, Y) = ' + "{:.5e}".format(min(InnerCosts_all[iIter])))
            # plt.plot(f_inner_best, '-b', linewidth=1.5,
            #          label='Best cost:J(C / Theta_{best}, Y) = ' + "{:.5e}".format(
            #              f_inner_best[-1]))
            # plt.legend(loc='best')
            # plt.tight_layout()
            # plt.savefig(folderForOutput + '/inner_cost_evolution.png', dpi=400)
            #
            # # plot evolution of outer costs
            # plt.figure(figsize=(10, 6))
            # plt.semilogy()
            # plt.xlabel('Iteration')
            # plt.ylabel('Outer optimisation cost')
            # for iIter in range(len(f_outer_best) - 1):
            #     plt.scatter(iIter * np.ones(len(OuterCosts_all[iIter])), OuterCosts_all[iIter], c='k', marker='.',
            #                 alpha=.5,
            #                 linewidths=0)
            # iIter += 1
            # plt.scatter(iIter * np.ones(len(OuterCosts_all[iIter])), OuterCosts_all[iIter], c='k', marker='.', alpha=.5,
            #             linewidths=0, label='Sample cost: H(Theta / C, Y)')
            # plt.plot(f_outer_best, '-b', linewidth=1.5,
            #          label='Best cost:H(Theta_{best} / C, Y) = ' + "{:.5e}".format(f_outer_best[-1]))
            # plt.legend(loc='best')
            # plt.tight_layout()
            # plt.savefig(folderForOutput + '/outer_cost_evolution.png', dpi=400)
            #
            # # plot evolution of outer costs
            # plt.figure(figsize=(10, 6))
            # plt.semilogy()
            # plt.xlabel('Iteration')
            # plt.ylabel('Gradient matching cost')
            # for iIter in range(len(f_gradient_best) - 1):
            #     plt.scatter(iIter * np.ones(len(GradCosts_all[iIter])), GradCosts_all[iIter], c='k', marker='.',
            #                 alpha=.5,
            #                 linewidths=0)
            # iIter += 1
            # plt.scatter(iIter * np.ones(len(GradCosts_all[iIter])), GradCosts_all[iIter], c='k', marker='.', alpha=.5,
            #             linewidths=0, label='Sample cost: G_{ODE}(C / Theta, Y)')
            # plt.plot(f_gradient_best, '-b', linewidth=1.5,
            #          label='Best cost:G_{ODE}(C / Theta, Y) = ' + "{:.5e}".format(f_gradient_best[-1]))
            # plt.legend(loc='best')
            # plt.tight_layout()
            # plt.savefig(folderForOutput + '/gradient_cost_evolution.png', dpi=400)
        counter += 1
    ## end of loop over lambda values
    ####################################################################################################################
    ## save the figures
    # save the states plot
    iAx = 0
    for _, ax in axes.items():
        ax.set_ylabel(y_labels[iAx], fontsize=12)
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
        ax.legend(fontsize=12, loc='best')
        iAx += 1
    # plt.tight_layout(pad=0.3)
    fig.savefig(folderForOutput + '/states_model_output.png', dpi=400)
    # save the error plot
    iAx = 0
    for _, ax in axes1.items():
        ax.set_ylabel(y_labels1[iAx], fontsize=12)
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
        ax.legend(fontsize=12, loc='best')
        iAx += 1
    # plt.tight_layout(pad=0.3)
    fig1.savefig(folderForOutput + '/errors_model_output.png', dpi=400)

    # save figure 2 - log scale
    fig2.tight_layout()
    fig2.savefig(folderForOutput + '/ODE_params_log_scale.png', dpi=400)

    # save figure 3 - decimal scale
    fig3.tight_layout()
    fig3.savefig(folderForOutput + '/ODE_params.png', dpi=400)

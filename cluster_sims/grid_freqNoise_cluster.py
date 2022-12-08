from brian2 import *
#prefs.codegen.target = 'numpy'

import itertools
import pickle
import numpy as np

from encoder_class.phase_encoder import PhaseEncoder, run_simulation
from encoder_class.theoretical_functions import bins_mutual_I


def grid_freqNoise_exp(idx):

    ## Params
    M = 10
    N = 10

    model_params = {}
    model_params["tau_m"] = 24*ms
    model_params["R_m"] = 142e6*ohm
    model_params["v_thres"] = 15*mV
    model_params["v_rest"] = 0*mV
    model_params["v_reset"] = 0*mV
    model_params["tau_ref"] = 0*ms
    model_params["v_0"] = 0*mV
    model_params["noise_frac"] = 0.16

    oscillation_params = {}
    oscillation_params["I_osc"] = 40*pA
    oscillation_params["f"] = 5*Hz

    input_params = {}
    input_params["automatic_range"] = True
    input_params["I_min"] = 75*pA
    input_params["I_max"] = 130*pA
    input_params["corr_frac"] = 0.05

    simulation_params = {}
    simulation_params["method"] = "euler"
    simulation_params["num_oscillations"] = 150
    simulation_params["dt"] = 0.05*ms
    simulation_params["record_dt"] = 0.5*ms
    simulation_params["monitor_spikes"] = True
    simulation_params["monitor_voltage"] = False



    ## Experiments
    noise_lims = [0., 0.4]
    f_lims = [1*Hz, 50*Hz]

    noise_fracs = np.linspace(noise_lims[0], noise_lims[1], 100)
    fs = np.linspace(f_lims[0], f_lims[1], 100)

    experiments = list(itertools.product(noise_fracs, fs))
    num_experiments = len(list(experiments))

    noise_frac, f = experiments[idx]  #access

    model_params["noise_frac"] = noise_frac
    oscillation_params["f"] = f
    encoder = PhaseEncoder(num_ensembles=M, ensemble_size=N, model_params=model_params, 
                           oscillation_params=oscillation_params, input_params=input_params,
                           simulation_params=simulation_params, rnd_seed=0)
    
    try:
        phis = run_simulation(encoder, mode='experimental')
        #mus, i_sigmas, phi_sigmas  = get_distr_params(encoder)
    except Exception as e:
        print(e)

    #stoch_sigmas = list(map(lambda eta: model_sigma(eta, encoder), noise_fracs))

    MI = bins_mutual_I(phis)

    ## Save results
    results_dict = {}
    results_dict["MI"] = MI
    #results_dict["phis"] = phis

    filename = 'Data_Noise/' + 'experimental_' + '{0:03d}'.format(int(idx/100)) + '_' + '{0:03d}'.format(int(idx%100)) + ".pickle"

    with open(filename, 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
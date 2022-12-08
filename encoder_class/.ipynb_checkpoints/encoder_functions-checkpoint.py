import numpy as np
from brian2 import *


def get_ng_params(total_neurons, eqs, m_params, s_params, o_params):
    """
    get_ng_params(total_neurons, eqs, m_params, s_params, o_params)
    Returns dict with keys the parameters of NeuronGroup
    
    Parameters
    ----------
    total_neurons : number of neurons of the NeuronGroup
    eqs: Inline equations of the model
    m_params: model_params dictionary
    s_params: simulation_params dictionary
    o_params: oscillation_params dictionary
    
    Returns
    -------
    np : dict with keys the parameters of NeuronGroup
    """
    np = {}
    np["N"] = total_neurons
    np["model"] = eqs
    #np["threshold"] = 'v > {}*mV'.format(m_params["v_thres"]/mV)
    np["threshold"] = 'v > v_thres'
    np["reset"] = 'v = {}*mV'.format(m_params["v_reset"]/mV)
    
    np["refractory"] = m_params["tau_ref"]
    #np["refractory"] = 't <= initial_ref'
    #np["refractory"] = 'ref'
    np["method"] = s_params["method"]
    return np


def get_automatic_range(m_params, i_params, o_params):
    """
    get_automatic_range(m_params, i_params, o_params)
    Returns I_s parameters when "automatic_range" set to true
    
    Parameters
    ----------
    m_params: model_params dictionary
    i_params: input_params dictionary
    o_params: oscillation_params dictionary
    
    Returns
    -------
    I_min : Current of neuron receiving lowest I_s
    I_max : Current of neuron receiving highest I_s
    I_min : Difference in current between I_min/I_max and I_min_lock/I_max_lock
    """
    A = 1/sqrt(1 + (m_params["tau_m"]*o_params["omega"])**2)
    expon = exp(-o_params["T"]/m_params["tau_m"])
    frac_num = (m_params["v_thres"] - m_params["v_reset"]*expon)
    frac_denom = (1 - expon)*m_params["R_m"]*o_params["I_osc"]*A
    
    I_min = (frac_num/frac_denom - 1)*o_params["I_osc"]*A
    I_max = (frac_num/frac_denom + 1)*o_params["I_osc"]*A
    
    corr_frac = i_params["corr_frac"]
    I_corr = corr_frac*(I_max - I_min)
    
    return I_min, I_max, I_corr


#returns matrix of shape Mx(N*num_oscillations), containining
#the conditioned distributions of phi for each input

def min_phi_distribution(spike_monitor, f, total_time, phis_0, M, N):
    """
    min_phi_distribution(spike_monitor, f, total_time, phis_0, M, N)
    Returns conditioned phi distributions
    
    Parameters
    ----------
    spike_monitor: SM recorded during simulation
    f: temporal frequency of the oscillation
    phis_0: list containing the value of phi_0 (phase offset) for each neuron
    M: number of different inputs
    N: number of neurons per input
    
    Returns
    -------
    phi_matrix: matrix of shape Mx(N*num_oscillations). Each array phi_matrix[i] contains the values 
                of phi obtained for input i
    """
    def get_length(phi_list):
        if len(phi_list) == 0:
            return inf
        else:
            return len(phi_list)
    bin_time = 1/f
    num_bins = int(total_time/bin_time)
    phi_lists = []
    spike_trains = list(spike_monitor.spike_trains().values())
    for group in range(M):
        aux = []
        for num_bin in range(1, num_bins):
            for neuron_index in range(group*N, group*N + N):
                neuron_times = spike_trains[neuron_index] + bin_time*phis_0[neuron_index]/(2*pi)
                try:
                    index = np.where(neuron_times >= num_bin*bin_time)[0][0]
                    phi = 2*pi*f*(neuron_times[index]%bin_time)
                    aux.append(phi)
                except:
                    pass
        phi_lists.append(aux)
        
    min_length = min(list(map(lambda phi_list: get_length(phi_list), phi_lists)))
    for neuron_type in range(M):
        phi_lists[neuron_type] = phi_lists[neuron_type][:min_length]
    phi_matrix = np.array(phi_lists)
    return phi_matrix

#returns a matrix Mx1 containing the number of spikes per bin for each input level
def cycle_firing_rate(spike_monitor, num_bins, M, N):
    CFR = np.zeros(M)
    for group in range(M):
        CFR[group] = np.mean(spike_monitor.count[group*N:group*N + N])/num_bins
    return CFR

#returns sigma_W corresponding to an eta value
def model_sigma(eta, encoder):
    v_thres = encoder.model_params["v_thres"]
    v_rest = encoder.model_params["v_rest"]
    tau_m = encoder.model_params["tau_m"]
    R_m = encoder.model_params["R_m"]
    
    pre_sigma = eta*(v_thres - v_rest)/volt
    time_correction = np.sqrt(1/(tau_m/second))
    sigma = pre_sigma*time_correction
    return sigma



def first_passage_time(spike_monitor, freq, total_time, M, N, first_spikes):
    bin_time = 1/freq
    num_bins = int(total_time/bin_time)
    fptds = []
    for group in range(M):
        aux = []
        for neuron_index in range(group*N, group*N + N):
            neuron_times = list(spike_monitor.spike_trains().values())[neuron_index]
            try:
                index = np.where(neuron_times >= first_spikes[i])[0][0]
                fptd = neuron_times[index]
                aux.append(fptd)
            except: 
                pass
        fptds.append(aux)
    return fptds
        
    min_length = min(list(map(lambda ftp: get_length(ftp), ftps)))
    for neuron_type in range(M):
        fptds[neuron_type] = fptds[neuron_type][:min_length]
    fptds = np.array(fptds)
    return fptds

def get_passages(spike_monitors, M, N, min_time, passage_number):
    passage_times = np.zeros((M, N))
    key_list = ["SM_" + str(neuron) for neuron in range(M)]
    for input_, key in enumerate(key_list):
        for neuron, spike_train in enumerate(spike_monitors[key].spike_trains().values()):
            index = np.where(spike_train >= min_time)[0][passage_number]
            passage_times[input_, neuron] = spike_train[index]
    return passage_times

def get_passages_within(spike_monitors, M, N, min_time, passage_number):
    passage_times = np.zeros((M, N))
    key_list = ["SM_" + str(neuron) for neuron in range(M)]
    for input_, key in enumerate(key_list):
        for neuron, spike_train in enumerate(spike_monitors[key].spike_trains().values()):
            index = np.where(spike_train >= min_time)[0][passage_number]
            passage_times[input_, neuron] = spike_train[index]
    return passage_times




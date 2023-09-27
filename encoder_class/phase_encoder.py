'''
Module accepting simulation parameters (number of neurons, physiological parameters, runtime...) and outputting phase distributions and corresponding parameters (either through simulation or analytical solution)
'''

from brian2 import *
from scipy.stats import norm

from encoder_class.encoder_functions import get_ng_params, get_automatic_range, min_phi_distribution, get_passages, first_passage_time

from encoder_class.theoretical_functions import phi_of_I_lin, phi_of_I, v_of_t_prime


class PhaseEncoder(BrianObject):
    
    """
    Phase Encoder Class
    
    Parameters
    ----------
    num_ensembles: int, 
        number of different types of neurons (different tonic input)
    ensemble_size: int,
        number of identical neurons per ensemble 
    model_params: dict,
        contains physiological parameters such as membrane time constant, resistance, noise...     
    input_params: dict,
        contains parameters regarding tonic input   
    oscillation_params: dict,
        contains parameters regarding oscillatory input
    simulation_params: dict,
        contains parameters regarding integration method and timestep, runtime...
    rnd_seed: int,
        sets seed     
    """
    
    
    def __init__(self, num_ensembles, ensemble_size, model_params, input_params, oscillation_params,
                 simulation_params, rnd_seed=None):
        
        self.M = num_ensembles
        self.N = ensemble_size
        #the total number of neurons is the number of ensembles times the ensemble size
        self.total_neurons = self.M*self.N
        
        #params dictionaries are completed with some derived quantities (for example omega from f)
        self.model_params, self.input_params, self.oscillation_params, self.simulation_params = get_params(self.M,
                                        self.N, model_params, input_params, oscillation_params, simulation_params)
        
        #multi-line string defines neuron model in BRIAN2
        #dv/dt = (-(v - v_rest) + R_m*(I_theta + I_s))/tau_m + v_noise/sqrt(tau_m)*xi : volt
        self.eqs = '''
        dv/dt = (-(v - v_rest) + R_m*(I_theta + I_s) + v_noise)/tau_m: volt
        dv_noise/dt = -v_noise/tau_noise + xi*v_noise_sigma/sqrt(tau_noise) : volt
        v_noise_sigma : volt
        I_theta = I_osc*cos(omega*t -pi + phi_0) : amp
        T = 2*pi/omega : second
        v_rest : volt
        R_m: ohm
        I_s: amp
        tau_m: second
        tau_noise : second
        I_osc : amp
        omega : Hz
        phi_0 : 1
        v_thres : volt
        '''
        
        #set random seed
        self.rnd_seed = rnd_seed
        if self.rnd_seed != None:
            # this sets the brain random generator seed
            seed(self.rnd_seed)
            # this sets the numpy seed (just in case)
            numpy.random.seed(self.rnd_seed) 
            
        #set clocks
        self.simu_clock = Clock(self.simulation_params["dt"])
        self.simu_clock_synapse = Clock(self.simulation_params["dt"])
        self.record_clock = Clock(dt=self.simulation_params["record_dt"])
        super().__init__(clock=self.simu_clock)
        
        
        #create groups and subgroups
        self.groups = {}
        ng_params = get_ng_params(self.total_neurons, self.eqs, self.model_params, self.simulation_params, self.oscillation_params)    
        self.groups['encoder'] = NeuronGroup(clock = self.simu_clock, **ng_params)
        for group_bin in range(self.M):
            self.groups[group_bin] = self.groups['encoder'][group_bin*self.N:(group_bin + 1)*self.N]
        #assigns parameters to the NeuronGroup
        self.add_encoder_params()

        #define monitors (if set to)
        self.monitors = {}
        if simulation_params["monitor_spikes"]:
            self.monitors["SM"] = SpikeMonitor(self.groups["encoder"])    
        if simulation_params["monitor_voltage"]:
            self.monitors["VM"] = StateMonitor(self.groups["encoder"], 'v', record=True)

        #superclass contained_objects are extended with instance groups and monitors
        self.make_model()
        

    def add_encoder_params(self):
        m_params = self.model_params
        o_params = self.oscillation_params
        i_params = self.input_params
        
        self.groups["encoder"].v_thres = m_params["v_thres"]
        self.groups["encoder"].v_rest = m_params["v_rest"]
        self.groups["encoder"].R_m = m_params["R_m"]
        self.groups["encoder"].tau_m = m_params["tau_m"]
        self.groups["encoder"].tau_noise = m_params["tau_noise"]
        self.groups["encoder"].v_noise_sigma = m_params["noise_frac"]*(m_params["v_thres"] - m_params["v_rest"])
        self.groups["encoder"].I_osc = o_params["I_osc"]
        self.groups["encoder"].omega = o_params["omega"]
        self.groups['encoder'].I_s = i_params["I_s"]
        
        #all neurons start at v = 0mV
        self.groups['encoder'].v = 0*mV
        #the phase of the spike time without noise
        self.groups['encoder'].phi_0 = list(map(lambda i_s : phi_of_I(self, i_s), self.groups['encoder'].I_s))

       
    def make_model(self):
        self.contained_objects.extend([self.groups[ii] for ii in self.groups])
        self.contained_objects.extend([self.monitors[ii] for ii in self.monitors])
          

def get_params(M, N, m_params, i_params, o_params, s_params):
    
    #compute period and omega from frequency
    o_params["T"] = 1/o_params["f"]
    o_params["omega"] = 2*pi*o_params["f"]
    #get min and max value for phi(I) domain, and a correction proportional to its difference
    i_params["I_min_lock"], i_params["I_max_lock"], i_params["I_corr"] = get_automatic_range(m_params, 
                                                                                             i_params, o_params)
    #if set, the automatic range goes from I_min_lock + I_corr to I_max_lock - I_corr
    if i_params["automatic_range"]: 
        i_params["I_min"] = i_params["I_min_lock"] + i_params["I_corr"]
        i_params["I_max"] = i_params["I_max_lock"] - i_params["I_corr"]
    #give same input to each neuron within ensemble
    i_params["I_s"] = np.repeat(np.linspace(i_params["I_min"], i_params["I_max"], M), N)
    #compute runtime based on the number of oscillations and its period
    s_params["runtime"] = (s_params["num_oscillations"] + 1)*o_params["T"]
    
    return m_params, i_params, o_params, s_params
 

    
def run_simulation(network_model, report_style = 'text', mode='experimental'):
    
    if mode=='experimental':
        net = Network(network_model)
        runtime = network_model.simulation_params["runtime"]
        net.run(runtime, report = report_style)
        f = network_model.oscillation_params["f"]
        spike_monitors = network_model.monitors["SM"]
        phis = min_phi_distribution(spike_monitors, f, runtime, network_model.groups["encoder"].phi_0,
                                    network_model.M, network_model.N)
        
        return phis
    
    
    if mode == 'None':
        net = Network(network_model)
        runtime = network_model.simulation_params["runtime"]
        net.run(runtime, report = report_style)
        return None
        

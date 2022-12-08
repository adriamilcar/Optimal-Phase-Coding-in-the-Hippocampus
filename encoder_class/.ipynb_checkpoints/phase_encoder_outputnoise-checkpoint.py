from brian2 import *
from encoder_class.class_functions import get_ng_params, get_automatic_range
from encoder_class.processing_functions import min_phi_distribution
from encoder_class.theoretical_functions import trunc_Gauss_Sample, phi_of_I_lin



class PhaseEncoder(BrianObject):
    def __init__(self, num_ensembles, ensemble_size, model_params, oscillation_params,
                 input_params, simulation_params, rnd_seed=None):
        
        self.M = num_ensembles
        self.N = ensemble_size
        self.total_neurons = self.M*self.N
        
        self.model_params = model_params
        
        self.oscillation_params = oscillation_params
        self.oscillation_params["oscillation_T"] = 1/self.oscillation_params["f"]
        self.oscillation_params["omega"] = 2*pi*self.oscillation_params["f"]
    
        self.input_params = input_params
        if self.input_params["automatic_range"]: 
            self.input_params["I_min"], self.input_params["I_max"] = get_automatic_range(self.model_params, self.oscillation_params)
        self.input_params["I_s"] = np.linspace(self.input_params["I_min"], self.input_params["I_max"], self.M)
        
        self.simulation_params = simulation_params
        self.simulation_params["runtime"] = self.simulation_params["num_oscillations"]*self.oscillation_params["oscillation_T"]
        
        
        self.rnd_seed = rnd_seed
        
        self.groups = {}
        self.monitors = {}
        
        self.eqs = '''
        dv/dt = (-(v - v_rest) + R_m*(I_theta + I_s))/tau_m + v_noise/sqrt(tau_m)*xi : volt
        I_theta = I_osc*sin(omega*t) : amp
        v_rest : volt
        R_m: ohm
        I_s: amp
        tau_m: second
        v_noise : volt
        I_osc : amp
        omega : Hz
        '''
        

        self.simu_clock = Clock(self.simulation_params["dt"])
        self.simu_clock_synapse = Clock(self.simulation_params["dt"])
        self.record_clock = Clock(dt=self.simulation_params["record_dt"])

        super().__init__(clock=self.simu_clock)
        
        

        # start setting up the network
        ng_params = get_ng_params(self.total_neurons, self.eqs, self.model_params, self.simulation_params)    
        self.groups['encoder'] = NeuronGroup(clock = self.simu_clock, **ng_params)
        for group_bin in range(self.M):
            slc = [i for i in range(group_bin*self.N, (group_bin + 1)*self.N)]
            self.groups[group_bin] = self.groups['encoder'][slc]
        self.add_brain_params()

            
        if self.rnd_seed != None:
            # this sets the brain random generator seed
            seed(self.rnd_seed)
            # this sets the numpy seed (just in case)
            numpy.random.seed(self.rnd_seed)

        
            
        self.monitors["SM"] = SpikeMonitor(self.groups['encoder'])


        self.make_model()
            
    def make_model(self):
        self.contained_objects.extend([self.groups[ii] for ii in self.groups])
        #self.contained_objects.extend([self.synapses[ii] for ii in self.synapses])
        self.contained_objects.extend([self.monitors[ii] for ii in self.monitors])

    def add_brain_params(self):
        m_params = self.model_params
        o_params = self.oscillation_params
        i_params = self.input_params
        
        self.groups["encoder"].v_rest = m_params["v_rest"]
        self.groups["encoder"].R_m = m_params["R_m"]
        self.groups["encoder"].tau_m = m_params["tau_m"]
        self.groups["encoder"].v = m_params["v_0"]
        self.groups["encoder"].v_noise = m_params["noise_frac"]*(m_params["v_thres"] - m_params["v_rest"])
        
        self.groups["encoder"].I_osc = o_params["I_osc"]
        self.groups["encoder"].omega = o_params["omega"]
        

        for index in range(self.M):
            self.groups[index].I_s = i_params["I_s"][index]


def run_simulation(network_model, report_style = 'text', mode='experimental'):
    
    if mode=='experimental':
        net = Network(network_model)
        runtime = network_model.simulation_params["runtime"]
        net.run(runtime, report = report_style)
        spikes = network_model.monitors["SM"]
        f = network_model.oscillation_params["f"]
        omega = network_model.oscillation_params["omega"]
        phis = min_phi_distribution(spikes, f, runtime)
        return phis
        
        
    if mode=='theoretical':
        phis = []
        omega = network_model.oscillation_params["omega"]
        tau_m = network_model.model_params["tau_m"]
        noise_frac = network_model.model_params["noise_frac"]
        R_m = network_model.model_params["R_m"]
        v_th =  network_model.model_params["v_thres"]
        for ensemble in range(network_model.M):
            ie = network_model.input_params["I_s"][ensemble]

            value, inter, sl = phi_of_I_lin(Vthres=network_model.model_params["v_thres"],
                                            p=network_model.oscillation_params["oscillation_T"],
                                            R=network_model.model_params["R_m"],
                                            Iosc=network_model.oscillation_params["I_osc"],
                                            tau=tau_m,
                                            omega=omega,
                                            I_0=ie, I=ie)
            phi_mean = value
            correction = np.sqrt(tau_m/((value + pi)/omega))
            additional = (value + pi)/pi
            sigma = np.abs(noise_frac*sl*correction*v_th/R_m)
            phi_std = np.sqrt(1 + additional)*sigma
            phis.append(trunc_Gauss_Sample(network_model.M, network_model.N, mu=phi_mean, sigma=phi_std, a=-pi,
                                    b=0.5, num_oscillations=network_model.simulation_params["num_oscillations"]))
        phis = np.array(phis)
        return phis

        

import unittest
from math import isclose
from brian2 import *
from phase_encoder import PhaseEncoder
from theoretical_functions import *


M = 1
N = 1


model_params = {}
model_params["tau_m"] = 24*ms
model_params["R_m"] = 142e6*ohm
model_params["v_thres"] = 15*mV
model_params["v_rest"] = 0*mV
model_params["v_reset"] = 0*mV
model_params["tau_ref"] = 0*ms
model_params["v_0"] = 0*mV
model_params["noise_frac"] = 0.2

oscillation_params = {}
oscillation_params["I_osc"] = 40*pA
oscillation_params["f"] = 5*Hz

input_params = {}
input_params["automatic_range"] = True
input_params["I_min"] = 75*pA
input_params["I_max"] = 130*pA

simulation_params = {}
simulation_params["method"] = "euler"
simulation_params["num_oscillations"] = 1
simulation_params["monitor_spikes"] = False
simulation_params["monitor_voltage"] = False
simulation_params["dt"] = 0.05*ms
simulation_params["record_dt"] = 0.5*ms


encoder = PhaseEncoder(num_ensembles=M, ensemble_size=N, model_params=model_params, 
                       oscillation_params=oscillation_params, input_params=input_params,
                       simulation_params=simulation_params, rnd_seed=0)

class TestThFunc(unittest.TestCase):
    
    def test_Gauss(self):
        output = Gauss(0, 0, 1)
        result = 1/np.sqrt(2*pi)
        self.assertEqual(output, result)
        
    def test_AGauss(self):
        output = AGauss(0, 0, 1)
        result = 1/2
        self.assertEqual(output, result)
        
    def test_phi(self):
        output = phi(0)
        result = Gauss(0, 0, 1)
        self.assertEqual(output, result)
        
    def test_big_phi(self):
        output = big_phi(0)
        result = AGauss(0, 0, 1)
        self.assertEqual(output, result)
        
    def test_transform_moments(self):
        output_1, output_2 = transform_moments(180, 5, 172, 188)
        result_1, result_2 = 180, np.sqrt(15.034)
        self.assertAlmostEqual(output_1, result_1, 3)
        self.assertAlmostEqual(output_2, result_2, 3)
        
    def test_phi_of_I(self):
        output = phi_of_I(encoder, I)
        result = 2.08051
        self.assertAlmostEqual(output, result, 3)
        
    def test_phi_of_I_prime(self):
        output = big_phi(0)
        result = AGauss(0, 0, 1)
        self.assertEqual(output, result)
        
    def test_phi_of_I_lin(self):
        output = big_phi(0)
        result = AGauss(0, 0, 1)
        self.assertEqual(output, result)
        
    def test_first_spike_of_phi(self):
        output = big_phi(0)
        result = AGauss(0, 0, 1)
        self.assertEqual(output, result)
        
    def test_last_spike_of_phi(self):
        output = big_phi(0)
        result = AGauss(0, 0, 1)
        self.assertEqual(output, result)
        
    def test_v_of_t_prime(self):
        output = big_phi(0)
        result = AGauss(0, 0, 1)
        self.assertEqual(output, result)
        
    def test_entropy(self):
        output = big_phi(0)
        result = AGauss(0, 0, 1)
        self.assertEqual(output, result)
        
    def test_mixture(self):
        output = big_phi(0)
        result = AGauss(0, 0, 1)
        self.assertEqual(output, result)
        
    def test_zero_order_H(self):
        output = big_phi(0)
        result = AGauss(0, 0, 1)
        self.assertEqual(output, result)
        
    def test_gauss_H(self):
        output = big_phi(0)
        result = AGauss(0, 0, 1)
        self.assertEqual(output, result)
        
    def test_fun_func(self):
        output = big_phi(0)
        result = AGauss(0, 0, 1)
        self.assertEqual(output, result)
        
    def test_second_order_H(self):
        output = big_phi(0)
        result = AGauss(0, 0, 1)
        self.assertEqual(output, result)
        
    def test_bins_mutual_I(self):
        output = big_phi(0)
        result = AGauss(0, 0, 1)
        self.assertEqual(output, result)
        
    def test_approx_mutual_I(self):
        output = big_phi(0)
        result = AGauss(0, 0, 1)
        self.assertEqual(output, result)
        
    def test_SR_entropies(self):
        output = big_phi(0)
        result = AGauss(0, 0, 1)
        self.assertEqual(output, result)
        
    
        
        
if __name__ == '__main__':
    unittest.main()
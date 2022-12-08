from brian2 import *
import numpy as np
import scipy.stats
import scipy.special

def Gauss(x, mu, sigma):
    denom = np.sqrt(2*pi*sigma**2)
    return 1/denom*np.exp(-(x - mu)**2/(2*sigma**2))

def AGauss(x, mu, sigma):
    return (1.0 + scipy.special.erf((x - mu)/(sigma*np.sqrt(2.0))))/2.0

def phi(x):
    return 1/np.sqrt(2*pi)*np.exp(-x**2/2)

def big_phi(x):
    return 1/2*(1 + scipy.special.erf(x/np.sqrt(2)))


'''
def trunc_Gauss(x, mu, sigma, a, b):
    if (x < a) or (x > b):
        return 0
    else:
        num = little_phi((x-mu)/sigma)
        denom = sigma*(big_phi((b-mu)/sigma) - big_phi((a-mu)/sigma))
        return num/denom
'''

def transform_moments(mu, sigma, a, b):
    alpha = (a - mu)/sigma
    beta = (b - mu)/sigma
    zeta = big_phi(beta) - big_phi(alpha)
    
    mean = mu + (phi(alpha) - phi(beta))/zeta*sigma
    
    prod_term = 1 + (alpha*phi(alpha) - beta*phi(beta))/zeta + ((phi(alpha) - phi(beta))/zeta)**2
    std = sigma*np.sqrt(prod_term)
    
    return mean, std


def phi_of_I(encoder, I):
    v_thres = encoder.model_params["v_thres"]
    tau_m = encoder.model_params["tau_m"]
    R_m = encoder.model_params["R_m"]
    omega = encoder.oscillation_params["omega"]
    I_osc = encoder.oscillation_params["I_osc"]
    T = encoder.oscillation_params["T"]

    A = 1/np.sqrt(1 + (tau_m*omega)**2)
    ph = -np.arctan(tau_m*omega) - pi
    m = -1/(I_osc*A)
    b = v_thres/((1 - np.exp(-T/tau_m))*I_osc*R_m*A)
    phi = np.nan_to_num(-np.arccos(m*I + b) - ph)

    return phi


def phi_of_I_prime(encoder, I):
    v_thres = encoder.model_params["v_thres"]
    tau_m = encoder.model_params["tau_m"]
    R_m = encoder.model_params["R_m"]
    omega = encoder.oscillation_params["omega"]
    I_osc = encoder.oscillation_params["I_osc"]
    T = encoder.oscillation_params["T"]
    
    A = 1/np.sqrt(1 + (tau_m*omega)**2)
    m = -1/(I_osc*A)
    b = v_thres/((1 - np.exp(-T/tau_m))*I_osc*R_m*A)
    phi_prime = m/np.sqrt(1 - (m*I + b)**2)
    return phi_prime


def phi_of_I_lin(encoder, I_0, I):
    a_I = phi_of_I(encoder, I_0)
    b_I = phi_of_I_prime(encoder, I_0)
    intercept = a_I - b_I*I_0
    slope = b_I
    lin = a_I + b_I*(I - I_0)
    return lin, intercept, slope


def local_I_from_phi(encoder, I_s, phi):
    phi_mean = phi_of_I(encoder, I_s)
    phi_prime = phi_of_I_prime(encoder, I_s)
    
    return (phi - phi_mean)/phi_prime + I_s 



def first_spike_of_phi(p, phi):
    t_prime = (phi/(2*pi))*p
        
    if np.isnan(t_prime/second):
        t_prime = -1000*ksecond
    return t_prime

def last_spike_of_phi(p, phi):
    t_prime = (phi/(2*pi) - 1)*p
        
    if np.isnan(t_prime/second):
        t_prime = -1000*ksecond
    return t_prime



    
def v_of_t_prime(Vthres, p, Iosc, R, tau, omega, I, t_prime):
    if t_prime > -1*second:
        A = 1/np.sqrt(1 + (tau*omega)**2)
        ph = -np.arctan(tau*omega) - pi
        t = 0*ms
        delta_t = t - t_prime
        V = R*I*(1 - np.exp(-delta_t/tau)) + R*Iosc*A*(np.cos(t*omega + ph) - np.exp(-delta_t/tau)*np.cos(t_prime*omega + ph)) - 1*mV
    else:
        V = 0*mV     
    return V

def entropy(x):
    return -np.sum(np.where(x!=0 , x*np.log(x), 0))
        
def mixture(value, mus, sigmas):
    total = 0
    N = len(mus)
    for mu, sigma in zip(mus, sigmas):
        total += Gauss(value, mu, sigma)
    return total/N


def zero_order_H(mus, sigmas):
    total = 0
    N = len(mus)
    for mu in mus:
        total += np.log(mixture(mu, mus, sigmas))
    return -total/N


def gauss_H(sigma):
    return 1/2*np.log(2*pi*e*sigma**2)


def fun_func(mu, mus, sigmas):  
    ff = 0
    N = len(mus)
    for (mu_j, sigma_j) in zip(mus, sigmas):
        h = 0.01
        mu1 = mu*(1 - h)
        mu2 = mu*(1 + h)
        der = (mixture(mu2, mus, sigmas) - mixture(mu1, mus, sigmas))/(mu2 - mu1)
        term1 = der/mixture(mu, mus, sigmas)*(mu - mu_j)
        term2 = ((mu - mu_j)**2)/sigma_j
        #term1 = 0
        #term2 = 0
        ff += (1/N)*(1/sigma_j)*(term1 + term2 - 1)*Gauss(mu, mu_j, sigma_j)    
    ff = ff/mixture(mu, mus, sigmas)    
    return ff

def second_order_H(mus, sigmas):
    total = 0
    N = len(mus)
    for (mu, sigma) in zip(mus, sigmas):
        total += (1/N)*(1/2)*fun_func(mu, mus, sigmas)*sigma
    return -total


def neuron_entropy(neuron_phis, num_neurons, min_phi, max_phi):
    counts, bins = np.histogram(neuron_phis, bins=num_neurons, range=[min_phi, max_phi])
    pdf = counts/np.sum(counts)
    return entropy(pdf)
    

def bins_mutual_I(phis):
    num_neurons, num_cycles = phis.shape
    R_phis = phis.flatten()
    np.random.shuffle(R_phis)
    R_phis = R_phis.reshape(num_neurons, num_cycles)
    min_phi = min(phis.flatten())
    max_phi = max(phis.flatten())
    
    H_Rs = list(map(lambda neuron_phis : neuron_entropy(neuron_phis, num_neurons, min_phi, max_phi), phis))
    H_RS = np.mean(H_Rs)
    H_R = np.mean(list(map(lambda neuron_phis : neuron_entropy(neuron_phis, num_neurons, min_phi, max_phi), R_phis)))
    return H_R - H_RS


def entropy(x):
    return -np.sum(np.where(x!=0 , x*np.log(x), 0))
    
def const_bins_neuron_entropy(neuron_phis, num_bins, min_phi, max_phi):
    counts, bins = np.histogram(neuron_phis, bins=num_bins, range=[0, 2*pi])
    pdf = counts/np.sum(counts)
    return entropy(pdf)

def const_bins_mutual_I(phis, T, bin_time=1*ms):
    num_neurons, num_cycles = phis.shape
    R_phis = phis.flatten()
    min_phi = min(phis.flatten())
    max_phi = max(phis.flatten())
    np.random.shuffle(R_phis)
    R_phis = R_phis.reshape(num_neurons, num_cycles)
    
    num_bins = int(T/bin_time)
    H_Rs = list(map(lambda neuron_phis : const_bins_neuron_entropy(neuron_phis, num_bins, min_phi, max_phi), phis))
    H_RS = np.mean(H_Rs)
    H_R = np.mean(list(map(lambda neuron_phis : const_bins_neuron_entropy(neuron_phis, num_bins, min_phi, max_phi), R_phis)))
    return H_R - H_RS



def approx_mutual_I(params, method):
 
    if method=='approx_zero':
        mus = params[0]
        sigmas = params[1]
        H_R = zero_order_H(mus, sigmas)
        H_RS = np.mean(list(map(lambda sigma: gauss_H(sigma), sigmas)))
        
    if method=='approx_zero_corrected':
        mus = params[0]
        sigmas = params[1]
        H_R = zero_order_H(mus, sigmas) + 1/2
        H_RS = np.mean(list(map(lambda sigma: gauss_H(sigma), sigmas)))
    
    if method=='approx_second':
        mus = params[0]
        sigmas = params[1]
        H_R = zero_order_H(mus, sigmas) + second_order_H(mus, sigmas)
        H_RS = np.mean(list(map(lambda sigma: gauss_H(sigma), sigmas)))
        
    return H_R - H_RS


def SR_entropies(params):
    mus = params[0]
    sigmas = params[1]
    H_R = zero_order_H(mus, sigmas)
    H_RS = np.mean(list(map(lambda sigma: gauss_H(sigma), sigmas)))
    
    return H_R, H_RS
            
            

            
        


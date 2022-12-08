from brian2 import *
import numpy as np
import scipy.stats
import scipy.special

def Gauss(x, mu, sigma):
    denom = np.sqrt(2*pi*sigma**2)
    return 1/denom*np.exp(-(x - mu)**2/(2*sigma**2))

def AGauss(x, mu, sigma):
   return (1.0 + scipy.special.erf((x - mu)/(sigma*np.sqrt(2.0))))/2.0

def trunc_Gauss(x, mu, sigma, a, b):
  if (x < a) or (x > b):
    return 0
  else:
    denom = AGauss(b, mu, sigma) - AGauss(a, mu, sigma)
    return 1/denom*Gauss(x, mu, sigma)

def trunc_Gauss_Sample(M, N, mu, sigma, a, b, num_oscillations):
  values = np.linspace(a, b, 1000)
  dx = values[1] - values[0]
  probabilities = dx*np.array(list(map(lambda x : trunc_Gauss(x, mu, sigma/np.sqrt(N), a, b), values)))
  max_index = np.argmax(probabilities)
  probabilities[max_index] += (1 - np.sum(probabilities))
  sample = np.random.choice(values, size=num_oscillations, p=probabilities)
  return sample


def phi_of_I(Vthres, p, Iosc, R, tau, omega, I):
  A = 1/np.sqrt(1 + (tau*omega)**2)
  ph = -np.arctan(tau*omega)
  m = -1/(Iosc*A)
  b = Vthres/((1 - np.exp(-p/tau))*Iosc*R*A)
  phi = -np.arccos(m*I + b) - ph
  return phi



def phi_of_I_prime(Vthres, p, Iosc, R, tau, omega, I):
  A = 1/np.sqrt(1 + (tau*omega)**2)
  m = 1/(Iosc*A)
  b = -Vthres/((1 - np.exp(-p/tau))*Iosc*R*A)
  phi_prime = -m/np.sqrt(1 - (m*I + b)**2)
  return phi_prime


def phi_of_I_lin(Vthres, p, Iosc, R, tau, omega, I_0, I):
  a_I = phi_of_I(Vthres, p, Iosc, R, tau, omega, I_0)
  b_I = phi_of_I_prime(Vthres, p, Iosc, R, tau, omega, I_0)
  intercept = a_I +b_I*I_0
  slope = b_I
  lin = a_I + b_I*(I - I_0)
  return lin, intercept, slope

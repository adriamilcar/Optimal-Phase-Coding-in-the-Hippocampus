import numpy as np
from brian2 import *


def min_phi_distribution(spike_monitor, freq, total_time):
  bin_time = 1/freq
  bin_min = 3*bin_time/4
  num_bins = int(total_time/bin_time)
  phi_list = []
  for num_bin in range(num_bins):
    aux = []
    for neuron_times in spike_monitor.spike_trains().values():
      time_from_min = neuron_times - bin_min
      try:
        index = np.where(time_from_min >= num_bin*bin_time)[0][0]
        phi = 2*pi*((time_from_min[index]%bin_time))/bin_time - pi
        aux.append(phi)
      except: 
        aux.append(0)
    phi_list.append(aux)
  phi_array = np.transpose(np.array(phi_list))
  return phi_array



def ensemble_min_phi_distribution(spike_monitor, n_ensembles, size, freq, total_time):
  bin_time = 1/freq
  bin_min = 3*bin_time/4
  num_bins = int(total_time/bin_time)
  phi_list = []
  for num_bin in range(num_bins):
    print("bins processed: {}".format(num_bin))
    aux = []
    for i in range(n_ensembles):
      auxaux = []
      for neuron_times in list(spikes.spike_trains().values())[i*size:(i+1)*size]:
        time_from_min = neuron_times - bin_min
        try:
          index = np.where(time_from_min >= num_bin*bin_time)[0][0]
          phi = 2*pi*((time_from_min[index]%bin_time))/bin_time - pi
          auxaux.append(phi)
        except:
          auxaux.append(0)
      aux.append(min(auxaux))
    phi_list.append(aux)
  phi_array = np.transpose(np.array(phi_list))
  return phi_array


'''
def ensemble_av_phi_distribution(spike_monitor, n_ensembles, size, freq, total_time):
  bin_time = 1/freq
  bin_min = 3*bin_time/4
  num_bins = int(total_time/bin_time)
  phi_list = []
  for num_bin in range(num_bins):
    print("bins processed: {}".format(num_bin))
    aux = []
    for i in range(n_ensembles):
      auxaux = []
      for neuron_times in list(spikes.spike_trains().values())[i*size:(i+1)*size]:
        time_from_min = neuron_times - bin_min
        try:
          index = np.where(time_from_min >= num_bin*bin_time)[0][0]
          phi = 2*pi*((time_from_min[index]%bin_time))/bin_time - pi
          auxaux.append(phi)
        except:
          auxaux.append(0)
      aux.append(np.mean(auxaux))
    phi_list.append(aux)
  phi_array = np.transpose(np.array(phi_list))
  return phi_array
'''

def entropy(x):
  return -np.sum(np.where(x!=0 , x*np.log2(x), 0))


def mutual_I(experiment_phis, num_bins):

    Rs_entropies = []
    R_entropies = []

    for exp_number, phis in enumerate(experiment_phis):
        Rs_entropy = []
        R_entropy = []
        Rs_phis = phis
        Rs_dims = Rs_phis.shape
        R_phis = Rs_phis.flatten()
        np.random.shuffle(R_phis)
        R_phis = R_phis.reshape(Rs_dims)
        min_phi = min(phis.flatten())
        max_phi = max(phis.flatten())
        for neuron in range(num_bins):
            Rs_counts, Rs_bins = np.histogram(Rs_phis[neuron], bins=num_bins, range=[min_phi, max_phi])
            pdf = Rs_counts/np.sum(Rs_counts)
            Rs_entropy.append(entropy(pdf))
            R_counts, R_bins = np.histogram(R_phis[neuron], bins=num_bins, range=[min_phi, max_phi])
            pdf = R_counts/np.sum(R_counts)
            R_entropy.append(entropy(pdf))

        Rs_entropies.append(Rs_entropy)
        R_entropies.append(R_entropy)
        
    mutual_I = np.mean(np.array(R_entropies), axis=1) - np.mean(np.array(Rs_entropies), axis=1)     
    
    return mutual_I

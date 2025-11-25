#!python
# cython: embedsignature=True, binding=True

from .snn_cython cimport SNN_cython_population
from .encoder cimport Encoder
from .augmented cimport Augmented
from .energy cimport Energy

import numpy as np
cimport numpy as np
np.import_array()

cimport cython
ctypedef np.uint8_t uint8

cdef class Runner_SL_cython:
    cdef float dt # time step
    cdef int run_time_margin # simulation time (+ margin to let the spike propagate fully in the network)
    cdef int run_time_original # simulation time original
    cdef int batch_running # number of features running in batch
    cdef int batch_features # number of features in batch
    cdef int batch_population # number of network in batch
    cdef int nb_neurons
    cdef int nb_neurons_actives
    cdef int nb_networks
    cdef int population_len
    cdef bint is_delay
    cdef bint is_refractory
    cdef bint is_threshold_reset
    cdef bint is_record_spike
    cdef bint is_record_voltage
    cdef bint is_record_augmented

    cdef bint is_augmented
    cdef bint is_energy
    cdef public Augmented augmented
    cdef public Energy enrergy

    # network matrix
    cdef np.ndarray voltage
    cdef np.ndarray voltage_reset
    cdef np.ndarray weight
    cdef np.ndarray refractory
    cdef np.ndarray refractory_active
    cdef np.ndarray delay
    cdef int delay_max
    cdef np.ndarray threshold
    cdef np.ndarray tau
    cdef np.ndarray spike_state
    cdef np.ndarray spike_state_sub
    cdef np.ndarray current_kernel
    cdef np.ndarray constant_current_kernel
    cdef np.ndarray current_kernel_delay
    cdef np.ndarray current_kernel_indexes
    cdef np.ndarray current_kernel_delay_indexes

    cdef np.ndarray input_indexes
    cdef np.ndarray hidden_indexes
    cdef np.ndarray output_indexes

    cdef float[:,:,:,:] voltage_view
    cdef float[:,:,:] voltage_sub_view
    cdef float[:,:,:] voltage_sub_view_next
    cdef int[:,:,:,:] spike_state_view
    cdef int[:,:,:] spike_state_view_sub

    cdef float[:,:,:] weight_view
    cdef int[:,:,:] delay_view
    cdef int[:,:] refractory_view
    cdef int[:,:,:] refractory_active_view
    cdef float[:,:] voltage_reset_view
    cdef float[:,:] threshold_view
    cdef float[:,:] tau_view
    cdef float[:,:] current_view
    cdef float[:,:] constant_current_view
    cdef int[:] neurons_actives_indexes_view
    cdef int[:, :] synapses_actives_indexes_view
    

    # private variables
    cdef float[:,:,:,:] input_data # input data
    cdef float input_spike_amplitude # spike amplitude
    cdef int count # count -> could be the futur batch size/index

    cdef list networks_list # list of networks
    cdef list networks_list_no_split # list of networks

    # cpdef void init(self, SNN_cython_population population, np.ndarray[np.int32_t, ndim=1] population_genome_ids, int run_time = *, int run_time_margin = *, float dt = *, float input_spike_amplitude=*, bint is_gpu=*, int batch_running = *, int batch_features=*, int batch_population = *, str neuron_reset=*, list record_layer=*, bint disable_output_threshold = *, str decay_method = *, bint is_delay=*, bint is_refractory=*, set record_decoding_method = *, bint is_last_run = *)
    cpdef void init(self, int run_time=*, int run_time_margin=*, float dt=*, float input_spike_amplitude=*, bint is_augmented=*, bint is_energy=*, str neuron_reset=*, list record_layer=*, bint disable_output_threshold=*, str decay_method=*, bint is_delay=*, bint is_refractory=*, set record_decoding_method=*)
    cpdef void init_run(self, SNN_cython_population population, np.ndarray[np.int32_t, ndim=1] population_genome_ids, int batch_features=*, int batch_running=*, int batch_population=*, bint is_gpu=*, bint is_last_run = *)

    cpdef void run(self, np.ndarray input_data, bint is_encoded=*)
    cdef void run_CPU(self)

    # Network Loop functions
    cdef void init_network_CPU(self, np.ndarray[np.int32_t, ndim=1] pop_idx)
    cdef void input_data_to_network(self, int x_volt)
    cdef void LIF_update(self, int current_time)
    cdef void update_voltage_with_weights_delay_refractory(self, int current_time)

    # Utils functions
    cdef dict record_spikes
    cdef dict record_voltages
    cdef list record_layer
    cdef void init_record(self)
    cdef void record(self, np.ndarray[np.int32_t, ndim=1] pop_idx)

    cdef bint disable_output_threshold
    cdef np.ndarray neuron_to_record_indexes
    cdef bint is_LIF_beta
    cdef list split(self, list lst, int s, bint to_numpy=*)

    # record functions
    cpdef dict get_record_augmented_spikes(self)
    cpdef dict get_record_spikes(self)
    cpdef dict get_record_voltages(self)
    cpdef np.ndarray[np.int32_t, ndim=3] get_record_spikes_raw(self)
    cpdef np.ndarray[np.int32_t, ndim=3] get_record_voltages_raw(self)
    cpdef np.ndarray[np.int32_t, ndim=3] get_record_augmented_raw(self)

    # GPU
    cdef void init_network_GPU(self, np.ndarray[np.int32_t, ndim=1] pop_idx, bint is_re_alloc)
    cdef void run_GPU(self)
    # cpdef void free_GPU(self)
    cdef void* snn_gpu_ptr
    cdef bint is_gpu
    cdef bint is_re_alloc
    cdef bint is_last_run

    # POP
    cdef np.ndarray population_ids
    cdef list population_ids_split
    cdef list population_ids_split_record
    cdef np.ndarray record_spikes_array
    cdef np.ndarray record_voltages_array
    cdef np.ndarray record_augmented_array
    cdef SNN_cython_population population


    cpdef np.ndarray get_encoded_data(self)

    cdef public Encoder encoder
#!python
# cython: embedsignature=True, binding=True

# from snn_cython cimport SNN_cython as SNN
from SNN_cython_cuda.SNN_cython.snn_cython cimport SNN_cython as SNN
from SNN_cython_cuda.SNN_cython.snn_cython cimport SNN_cython_population
from .encoder cimport Encoder
from .augmented cimport Augmented
from .energy cimport Energy
from .r_stdp cimport R_STDP

import numpy as np
cimport numpy as np
np.import_array()

cimport cython
ctypedef np.uint8_t uint8

cdef class Runner_RL_cython:

    cdef public Encoder encoder
    cdef public Augmented augmented
    cdef public Energy energy
    cdef public R_STDP r_stdp
    cdef bint is_augmented
    cdef bint is_energy
    cdef bint is_r_stdp


    cdef float dt # time step
    cdef int run_time_original # simulation time
    cdef int run_time_margin # simulation time + margin
    cdef int run_time_delay_max # simulation time + margin + delay_max
    cdef int nb_episode # number of episode in batch
    cdef int nb_neurons
    cdef int nb_networks
    cdef int nb_neurons_actives
    cdef int population_len
    cdef int time_step
    cdef int first_time_step
    cdef bint online
    cdef bint is_delay
    cdef bint is_refractory
    cdef bint is_threshold_reset
    cdef bint is_voltage_negative

    # network matrix
    cdef np.ndarray voltage
    # cdef np.ndarray voltage_2
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
    cdef int[:,:] refractory_view
    cdef int[:,:,:] refractory_active_view
    cdef float[:,:] voltage_reset_view
    cdef float[:,:] threshold_view
    cdef float[:,:] tau_view
    cdef float[:,:] current_view
    cdef float[:,:] constant_current_view
    cdef int[:,:,:] delay_view
    cdef float[:, :, :, :] current_kernel_delay_2_view
    cdef int[:] neurons_actives_indexes_view
    cdef int[:, :] synapses_actives_indexes_view
    
    # cdef np.ndarray input_indexes
    # cdef np.ndarray hidden_indexes
    # cdef np.ndarray output_indexes


    # private variables
    cdef float[:,:,:,:] input_data # input data
    cdef float input_spike_amplitude # spike amplitude
    cdef list networks_list # list of networks

    # cpdef void init(self, SNN_cython_population population, np.ndarray[np.int32_t, ndim=1] population_genome_ids, int run_time=*, int run_time_margin=*, float dt=*, bint is_gpu=*, int nb_episode=*, bint online=*, str neuron_reset=*, list record_layer=*, bint disable_output_threshold=*, str decay_method=*, bint is_delay=*, bint is_refractory=*, int delay_max=*, set record_decoding_method=*, set spike_count_negative=*, bint is_last_run=*, bint is_voltage_negative=*)
    # cpdef void run(self, str encoder_type, np.ndarray inputs_data, int spike_rate= *, float spike_amplitude = *, int max_nb_spikes = *, int reduce_noise = *, int combinatorial_factor = *, int combinaison_size=*, int combinaison_size_max=*, float combinatorial_combinaison_noise=*, bint combinatorial_roll =*, str combinatorial_filter = *, float[:] combinatorial_modulo = *, bint combinatorial_print_table_debug = *, float direct_min = *, float direct_max = *, float derivative_threshold=*, bint derivative_is_latency=*, bint derivative_is_latency_positional=*, bint derivative_use_prev_input=*, float derivative_max_delta_latency =*)
    # cpdef dict run(self, str type, np.ndarray inputs_data, int spike_rate = *, float spike_amplitude = *, int max_nb_spikes = *, int reduce_noise = *)
    
    cpdef void init(self, int run_time=*, int run_time_margin=*, float dt=*, int nb_episode=*, bint online=*, bint is_augmented=*, bint is_energy=*, bint is_r_stdp=*, str neuron_reset=*, list record_layer=*, bint disable_output_threshold=*, str decay_method=*, bint is_delay=*, bint is_refractory=*, int delay_max=*, set record_decoding_method=*, set spike_count_negative=*, bint is_voltage_negative=*)
    cpdef void init_run(self, SNN_cython_population population, np.ndarray[np.int32_t, ndim=1] population_genome_ids, bint is_gpu=*, bint is_last_run=*)
    
    cpdef void run(self, np.ndarray inputs_data, bint is_encoded=*)
    cdef void run_CPU(self)

    
    # Network Loop functions
    cdef void init_network_GPU(self, np.ndarray[np.int32_t, ndim=1] pop_idx, bint is_re_alloc)
    cdef void init_network_CPU(self, np.ndarray[np.int32_t, ndim=1] pop_idx)
    cdef void init_network_step(self)
    cdef void input_data_to_network(self, int x_volt)
    cdef void LIF_update(self, int current_time)
    cdef void update_voltage_with_weights_delay_refractory(self, int current_time)

    # Utils functions
    cdef bint disable_output_threshold
    cdef bint is_LIF_beta

    # record functions
    cdef bint is_record_spike
    cdef bint is_record_voltage
    cdef bint is_record_augmented

    cdef list record_layer
    cdef np.ndarray neuron_to_record_indexes
    cdef np.ndarray neuron_decoder_to_record_indexes

    cdef dict record_spikes
    cdef dict record_voltages
    cdef dict record_augmented

    cdef void record(self)
    cpdef dict get_record_spikes(self)
    cpdef dict get_record_voltages(self)
    cpdef dict get_record_augmented_spikes(self)

    cpdef np.ndarray get_record_spikes_raw(self)
    cpdef np.ndarray get_record_voltages_raw(self)
    cpdef np.ndarray get_record_augmented_raw(self)


    # GPU parameters
    cdef bint is_gpu
    cdef void *snn_gpu_ptr
    cdef void run_GPU(self)
    cpdef void free_GPU(self)
    cdef bint is_re_alloc


    # Population
    cdef bint is_last_run
    cdef SNN_cython_population population
    cdef np.ndarray population_ids
    cdef list population_ids_list


    # NEGATIVE SPIKES
    cdef set spike_count_negative_layer
    cdef bint is_spike_negative
    cdef bint is_input_spike_negative 
    cdef bint is_output_spike_negative
    cdef bint is_hidden_spike_negative

    cdef int input_indexes_start
    cdef int output_indexes_start
    cdef int hidden_indexes_start


    cdef int input_indexes_end
    cdef int output_indexes_end
    cdef int hidden_indexes_end
    
    
    cpdef np.ndarray get_weight(self)
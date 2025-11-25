import numpy as np
cimport numpy as np
np.import_array()



cdef class Augmented:

    cdef int run_time
    cdef int run_time_margin
    cdef bint is_LIF_beta

    cdef int nb_episodes
    cdef int nb_networks
    cdef int nb_neurons
    cdef np.ndarray population_ids
    cdef np.ndarray input_indexes
    cdef np.ndarray output_indexes
    cdef np.ndarray neuron_to_record_indexes
    cdef bint is_refractory

    cdef np.ndarray voltage_decoder
    cdef np.ndarray threshold_decoder
    cdef np.ndarray tau_decoder
    cdef np.ndarray refractory_decoder

    cdef float[:,:,:] voltage_decoder_view
    cdef float[:,:] threshold_decoder_view
    cdef float[:,:] tau_decoder_view
    cdef float[:,:] refractory_decoder_view
    cdef np.ndarray augmented_decoder
    cdef np.ndarray augmented_decoder_2
    cdef np.ndarray spikes_importance

    cdef bint is_record_augmented
    cdef bint is_augmented_decoder
    cdef bint is_voltage_reset
    cdef bint is_neurons_all_to_decode
    cdef bint is_neurons_update_with_augmented

    cdef np.ndarray neuron_decoder_indexes

    cdef int neuron_decoder_start
    cdef int neuron_decoder_end
    cdef int input_size_decoder
    cdef int spike_max
    cdef int spike_max_time_step
    cdef int spike_distribution_run
    cdef int spike_distribution_importance
    cdef int spike_format_type
    cdef int importance_type
    cdef int index_start_decoder_update    
    cdef bint linear_spike_importance_type

    cdef bint is_normalize
    cdef bint is_interpolate
    cdef float interpolate_max
    cdef float interpolate_min

    cdef dict record_augmented

    cdef bint is_first_init

    # cpdef void init_augmented_spikes_decoder(self, int spike_max=*, int spike_distribution_run=*, int spike_distribution_importance=*, str importance_type=*, str linear_spike_importance_type=*, str spike_type=*, bint is_normalize = *, bint is_interpolate = *, float interpolate_max = *, float interpolate_min = *, bint is_voltage_reset = *)
    # cdef void init(self, int run_time, int run_time_margin, bint is_LIF_beta)
    cpdef void init_param(self, bint is_neurons_all_to_decode, int spike_max=*, int spike_distribution_run=*, int spike_distribution_importance=*, str importance_type=*, str linear_spike_importance_type=*, str spike_type=*, bint is_normalize=*, bint is_interpolate=*, float interpolate_max=*, float interpolate_min=*, bint is_voltage_reset=*, bint is_neurons_update_with_augmented=*)
    cdef void init_run(self, int run_time, int run_time_margin, bint is_LIF_beta, int nb_episodes, int nb_networks, int nb_neurons, np.ndarray[np.int32_t, ndim=1] population_ids, np.ndarray[np.int32_t, ndim=1] input_indexes, np.ndarray[np.int32_t, ndim=1] output_indexes, np.ndarray[np.int32_t, ndim=1] neuron_to_record_indexes, np.ndarray[np.float32_t, ndim=2] threshold, np.ndarray[np.float32_t, ndim=2] tau, np.ndarray[np.int32_t, ndim=2] refractory, bint is_refractory=*)
    cdef void update(self, int current_time)
    cdef np.ndarray build_spike_importance(self, int spikes_distribution, int run_len, bint is_descending=*)
    cdef void apply_importance(self)

    cdef void init_step(self, int first_time_step, bint is_online)
    cdef void reset_record(self)

    cdef dict record_dict(self)
    cdef np.ndarray record_numpy(self)

    # cdef void init_importance(self, int run_time, int run_time_margin, bint is_LIF_beta)

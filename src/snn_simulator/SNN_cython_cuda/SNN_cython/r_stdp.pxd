import numpy as np
cimport numpy as np
np.import_array()

cdef class R_STDP:
    # STDP
    cdef bint is_stdp
    cdef bint is_r_stdp

    cdef float stdp_learning_rate
    cdef float tau_syn
    cdef float weight_level
    cdef float decay_neuron_trace
    cdef float decay_synapse_trace
    cdef float p_noise_stdp
    cdef float w_noise_stdp
    cdef float sign_noise_stdp
    cdef int   t_window
    cdef float [:,:,:] background_spikes
    cdef float [:,:,:] neuron_trace
    cdef float [:,:,:] synapse_trace
    cdef float [:,:] reward

    cpdef void init_param(self, float learning_rate=*, float weight_level=*, int t_window=*, float A_plus=*, float A_minus=*, float tau_plus=*, tau_minus=*, float tau_syn=*, float w_min=*, float w_max=*, float p_noise_stdp=*, float w_noise_stdp=*, float sign_noise_stdp=*, float R_plus_plus=*, float R_plus_minus=*, float R_minus_plus=*, float R_minus_minus=*)
    cdef void init_run(self, int nb_episode, int nb_networks, int nb_neurons, int[:,:] synapses_actives_indexes_view)
    cdef void init_step(self, int first_time_step, bint is_online)
    cdef float add_noise_spikes_and_voltages(self, int episode, int network, int neuron, float voltage, float threshold)
    cdef void update_neuron_trace(self, int episode, int network_idx, int neuron_idx, bint is_spike)
    cdef void update_synpase_trace(self, int episode, int network, int neuron_pre, int neuron_post, bint is_spike_pre, bint is_spike_post)
    cpdef void R_STDP_Apply_Reward(self, float[:,:] rewards, float[:,:,:] weight_view)
    cdef void set_hidden_weight_positive(self, np.ndarray[np.float32_t, ndim=3] weight, int nb_synapses, int output_indexes_start)

    cdef float A_plus
    cdef float A_minus
    cdef float tau_plus
    cdef float tau_minus
    cdef float w_max
    cdef float w_min
    cdef int stdp_time_step
    cdef float _SAFE_EXP(self, float x)
    cdef float[:] output_weight

    cdef float R_plus_plus
    cdef float R_plus_minus
    cdef float R_minus_plus
    cdef float R_minus_minus

cimport cython
import numpy as np
cimport numpy as np
np.import_array()


cdef void run_SNN_GPU_RL(
                        void *snn_gpu_ptr, 
                        float[:,:,:,:] input_data, 
                        )
cdef void run_SNN_GPU_SL(
                        void *snn_gpu_ptr, 
                        float[:,:,:,:] input_data, 
                        )

cdef void* init_SNN_GPU(
    void *snn_host_device_ptr,

    int nb_episodes,
    int nb_networks,
    int nb_neurons,
    int nb_steps,


    # Neurons parameters
    np.ndarray[float, ndim=2, mode='c'] threshold,
    np.ndarray[float, ndim=2, mode='c'] tau,
    np.ndarray[float, ndim=2, mode='c'] constant_current,

    # Synapses parameters
    np.ndarray[float, ndim=3, mode='c'] weights,

    # Optional parameters
    np.ndarray[np.int32_t, ndim=2, mode='c'] refractory,
    np.ndarray[np.int32_t, ndim=3, mode='c'] delay,
    
    # Indexes parameters
    np.ndarray[np.int32_t, ndim=1, mode='c'] input_indexes, 
    np.ndarray[np.int32_t, ndim=1, mode='c'] output_indexes, 
    np.ndarray[np.int32_t, ndim=1, mode='c'] hidden_indexes, 

    np.ndarray[np.int32_t, ndim=2, mode='c'] synapses_actives_indexes,

    # Optional parameters
    bint is_LIF_beta,
    bint is_refractory,
    bint is_delay,
    bint is_record_spikes,
    bint is_online,
    bint is_SL
    )

cdef void* init_augmented_GPU(void *snn_gpu_ptr, np.ndarray[float, ndim=1, mode='c'] spikes_importance, int spike_max_time_step_augmented, bint is_neurons_update_with_augmented, bint is_voltage_reset_augmented, int spikes_format_type_augmented, int importance_type_augmented)
cdef void* init_energy_GPU(void *snn_gpu_ptr, np.ndarray[float, ndim=4, mode='c'] energy_from_cython, np.ndarray[int, ndim=3, mode='c'] energy_idx_from_cython, int energy_length, float[:,:,:] energy_weight, int energy_update_method)
cdef void get_recorded_spikes_augmented_GPU(void *snn_host_device_ptr, np.ndarray[float, ndim=4, mode='c'] spikes_rec_augmented_host)

cdef void get_recorded_spikes_GPU(void *snn_host_device_ptr, np.ndarray[np.int32_t, ndim=4, mode='c'] spikes_rec_host)
cdef void get_recorded_voltages_GPU(void *snn_host_device_ptr, np.ndarray[float, ndim=4, mode='c'] voltages_rec_host)
cdef void get_recorded_spikes_augmented_GPU(void *snn_host_device_ptr, np.ndarray[float, ndim=4, mode='c'] spikes_rec_augmented_host)

cdef void free_SNN_GPU(void *snn_host_device_ptr)
# snn_cuda_wrapper.pyx

import numpy as np
cimport numpy as np
from libc.stdio cimport printf

cdef extern from "snn_cuda_f32.h":
    void* SNN_malloc_Augmented(void *snn_host_device, float* spikes_importance, int spikes_max_per_time_step_augmented, bint is_neurons_update_with_augmented, bint is_voltage_reset_augmented, int spikes_format_type_augmented, int importance_type_augmented)
    void* SNN_malloc_Energy(void *snn_host_device_ptr, float *energy_from_cython, int *energy_idx_from_cython, int energy_length, float *energy_weight, int energy_update_method)
    void run_SNN_RL(void *snn_host_device_ptr, float* input_data)
    void run_SNN_SL(void *snn_host_device_ptr, float* input_data)
    void SNN_free(void *snn_host_device_ptr)
    void get_recorded_spikes(void *snn_host_device_ptr, int* spikes_rec_augmented_host)
    void get_recorded_voltages(void *snn_host_device_ptr, float* spikes_rec_augmented_host)
    void get_recorded_spikes_augmented(void *snn_host_device_ptr, float* spikes_rec_augmented_host)
    void* init_SNN(
        void* snn_host_device_ptr,

        int nb_episodes,
        int nb_networks,
        int nb_neurons,
        int nb_steps,

        # Neurons parameters
        float* threshold,
        float* tau,
        float* constant_current,

        # Synapses parameters
        float* weights,

        # Optional parameters
        int* refractory,
        int* delay,
        
        # Indexes parameters
        int* input_indexes, 
        int* output_indexes, 
        int* hidden_indexes, 

        int* synapses_actives_indexes,

        # Sizes
        # int voltage_size,
        # int neuron_params_size,
        # int synapses_params_size,
        int input_indexes_size,
        int output_indexes_size,
        int hidden_indexes_size,
        int synapses_actives_indexes_size,

        # Optional parameters
        bint is_LIF_beta,
        bint is_refractory,
        bint is_delay,
        bint is_record_spikes,
        bint is_online,
        bint is_SL
        )

cdef void run_SNN_GPU_RL(
                        void *snn_gpu_ptr, 
                        float[:,:,:,:] input_data
                    ):
    run_SNN_RL(
            snn_gpu_ptr,
            &input_data[0, 0, 0, 0]
            )

cdef void run_SNN_GPU_SL(
                        void *snn_gpu_ptr, 
                        float[:,:,:,:] input_data
                    ):
    run_SNN_SL(
            snn_gpu_ptr,
            &input_data[0, 0, 0, 0]
            )

cdef void free_SNN_GPU(void *snn_gpu_ptr):
    SNN_free(snn_gpu_ptr)

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
    ):

    # cdef np.ndarray syn_sort = np.argsort(synapses_actives_indexes[1, :])
    # cdef np.ndarray synapses_actives_indexes_sorted = synapses_actives_indexes[:, syn_sort].copy()
    # print("synapses_actives_indexes\n", synapses_actives_indexes)
    # print("synapses_actives_indexes_sorted\n", synapses_actives_indexes_sorted)
    return init_SNN(
        snn_host_device_ptr,

        nb_episodes,
        nb_networks,
        nb_neurons,
        nb_steps,

        # Neurons parameters
        <float*> threshold.data,
        <float*> tau.data,
        <float*> constant_current.data,

        # Synapses parameters
        <float*> weights.data,

        # Optional parameters
        <int*> refractory.data if is_refractory == True else NULL,
        <int*> delay.data if is_delay == True else NULL,
        
        # Indexes parameters
        <int*> input_indexes.data, 
        <int*> output_indexes.data, 
        <int*> hidden_indexes.data, 

        <int*> synapses_actives_indexes.data,
        
        # Sizes
        # voltage.size,
        # weights.size, #synapses_params_size,
        np.size(input_indexes),
        np.size(output_indexes),
        np.size(hidden_indexes),
        np.size(synapses_actives_indexes),

        # Optional parameters
        is_LIF_beta,
        is_refractory,
        is_delay,
        is_record_spikes,
        is_online,
        is_SL
        )

cdef void* init_augmented_GPU(void *snn_gpu_ptr, np.ndarray[float, ndim=1, mode='c'] spikes_importance, int spike_max_time_step_augmented, bint is_neurons_update_with_augmented, bint is_voltage_reset_augmented, int spikes_format_type_augmented, int importance_type_augmented):
    return SNN_malloc_Augmented(snn_gpu_ptr, <float*>spikes_importance.data, spike_max_time_step_augmented, is_neurons_update_with_augmented, is_voltage_reset_augmented, spikes_format_type_augmented, importance_type_augmented)

cdef void* init_energy_GPU(void *snn_gpu_ptr, np.ndarray[float, ndim=4, mode='c'] energy_from_cython, np.ndarray[int, ndim=3, mode='c'] energy_idx_from_cython, int energy_length, float[:,:,:] energy_weight, int energy_update_method):
    return SNN_malloc_Energy(snn_gpu_ptr, <float*>energy_from_cython.data, <int*>energy_idx_from_cython.data, energy_length, &energy_weight[0,0,0], energy_update_method)

cdef void get_recorded_spikes_GPU(void *snn_host_device_ptr, np.ndarray[np.int32_t, ndim=4, mode='c'] spikes_rec_host):
    get_recorded_spikes(snn_host_device_ptr, <int*>spikes_rec_host.data)

cdef void get_recorded_voltages_GPU(void *snn_host_device_ptr, np.ndarray[float, ndim=4, mode='c'] voltages_rec_host):
    get_recorded_voltages(snn_host_device_ptr, <float*>voltages_rec_host.data)

cdef void get_recorded_spikes_augmented_GPU(void *snn_host_device_ptr, np.ndarray[float, ndim=4, mode='c'] spikes_rec_augmented_host):
    get_recorded_spikes_augmented(snn_host_device_ptr, <float*>spikes_rec_augmented_host.data)

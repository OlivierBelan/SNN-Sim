#ifndef SNN_CUDA_H
#define SNN_CUDA_H

#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
// #include <cuda_bf16.h>

struct SNN {
    
    bool is_LIF_beta;
    
    // input_data
    float *input_data;

    // Neuron parameters
    float *voltages;
    float *thresholds;
    float *tau;
    int *refrac;
    int *refrac_active;
    bool is_refractory;
    // float *constant_current;

    // Synapse parameters
    float *weights;
    int *delays;
    bool is_delay;


    // Some indexes
    int *synapses_actives_idx_in;
    int *synapses_actives_idx_out;
    int *synapses_actives_first_neuron_idx;

    // // Some indexes for synapses threads
    // int *nb_synapses_per_neuron;
    // int nb_synapses_threads;
    // int *decalage;

    int *input_idx;
    int *output_idx;
    int *hidden_idx;

    // recording
    bool   *spikes_rec_inference;
    int *spikes_rec;
    bool is_record_spikes;

    // dt
    float dt;


    // augmented
    float *voltages_augmented;
    float *spikes_rec_augmented;
    float spike_max_time_step_augmented;
    float *spikes_importance_augmented;
    int *spikes_importance_idx_augmented;
    int spikes_format_type_augmented;
    int importance_type_augmented;
    bool is_voltage_reset_augmented;
    bool is_neurons_update_with_augmented;

    // energy
    float *energy;
    // float *energy_from_cython;
    int *energy_idx;
    int energy_length;
    int energy_update_method;
};



struct info_runner {
    int nb_episodes;
    int nb_networks;
    int nb_neurons;
    int nb_synapses;
    int nb_steps;
    int nb_neurons_all_episodes;

    int nb_input;
    int nb_output;
    int nb_hidden;

    bool is_augmented;
    bool is_energy;
    bool is_online;
    bool is_re_alloc;
    bool is_SL;

    // int block_size_neurons;
    // int block_size_synapses;
};

struct SNN_HOST_DEVICE {
    SNN *snn_host;
    SNN *snn_device;

    info_runner *info_host;
    info_runner *info_device;
};



extern "C" {
    void* SNN_malloc_Augmented(void *snn_host_device_ptr, float *spikes_importance_augmented, int spike_max_time_step_augmented, bool is_neurons_update_with_augmented, bool is_voltage_reset_augmented, int spikes_format_type_augmented, int importance_type_augmented);
    void* SNN_malloc_Energy(void *snn_host_device_ptr, float *energy_from_cython, int *energy_idx_from_cython, int energy_length, float *energy_weight, int energy_update_method);
    void SNN_free(void *snn_host_device_ptr);

    void run_SNN_RL(void *snn_host_device_ptr, float *input_data);
    void run_SNN_SL(void *snn_host_device_ptr, float *input_data);

    void get_recorded_spikes(void *snn_host_device_ptr, int *spikes_rec_host);
    void get_recorded_voltages(void *snn_host_device_ptr, float *voltages_host);
    void get_recorded_spikes_augmented(void *snn_host_device_ptr, float *spikes_rec_augmented_host);

    void* init_SNN(      
                        // pointer to possibly re-init
                        void *snn_host_device_ptr,

                        int nb_episodes,
                        int nb_networks,
                        int nb_neurons,
                        int nb_steps,
                                                
                        // Neuron parameters
                        float *threshold, 
                        float *tau, 
                        float *constant_current, 
                        
                        // Synapse parameters
                        float *weight, 

                        // Optional parameters
                        int *refractory_host,
                        int *delay_host,

                        // Indexes
                        int *input_indexes, 
                        int *output_indexes, 
                        int *hidden_indexes, 
                        int *synapses_actives_idx,

                        // Sizes
                        int input_size,
                        int output_size,
                        int hidden_size,
                        int synapses_actives_size,

                        // Optional parameters
                        bool is_LIF_beta,
                        bool is_refractory,
                        bool is_delay,
                        bool is_record_spikes,
                        bool is_online,
                        bool is_SL
                        );
}

#endif // SNN_CUDA_H
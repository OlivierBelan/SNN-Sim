#include <iostream>
#include <stdio.h>
// #include <stdlib.h>
// #include <stdbool.h>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include "snn_cuda_f32.h"

void snn_host_print(int nb_episodes, 
                int nb_networks, 
                int nb_neurons, 
                int nb_synapses, 
                int nb_steps, 
                int voltage_size, 
                int neuron_size, 
                int weight_size, 
                int synapses_actives_size, 
                int nb_input, 
                int nb_output, 
                int nb_hidden,
                float *threshold_host,
                float *tau_host,
                float *constant_current_host,
                float *weight_host,
                int *input_indexes_host,
                int *output_indexes_host,
                int *hidden_indexes_host,
                int *synapses_actives_idx,
                int *syn_host_in,
                int *syn_host_out
                );
void snn_device_print(SNN *snn_host, info_runner *info_host);
void print_info_device(info_runner *info_device);
__global__ void print_info_device_kernel(const info_runner* info_device);
void print_info_host(info_runner *info_host);


void init_host(float *h_voltages, int N, float value) {
    for (int i = 0; i < N; i++) {
        h_voltages[i] = value;
    }
}

void print_gpu_memory_usage(std::string message) {
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);
    if (status != cudaSuccess) {
        fprintf(stderr, "Erreur lors de la récupération de la mémoire GPU: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
    // printf("After %s Mémoire GPU: %lu Mo / %lu Mo\n", message,(total_mem / (1024 * 1024)) - (free_mem / (1024 * 1024)), total_mem / (1024 * 1024));
    std::cout << message << " allocated, GPU memory: " << (total_mem / (1024 * 1024)) - (free_mem / (1024 * 1024)) << "/" << total_mem / (1024 * 1024) << " Mo" << std::endl; 
}

template <typename T>
void init_host_array(T *array, int N, T value) {
    for (int i = 0; i < N; i++) {
        array[i] = value;
    }
}

template <typename T>
void cudaMalloc_with_error_check(std::string name, T **ptr, int size) {
    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Erreur d'allocation de la mémoire pour %s: %s\n", name.c_str(), cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
int get_size_array(T *array) {
    return sizeof(array) / sizeof(array[0]);
}


template <typename T, typename U>
void printArray_(T *array, const std::vector<U>& dimensions, int dim, int offset) {

    if (dim == dimensions.size() - 1) {
        std::cout << "[";
        for (int i = 0; i < dimensions[dim]; ++i) {
            if constexpr (std::is_same<T, __nv_bfloat16>::value) {
                std::cout << __bfloat162float(array[offset + i]);
            } else {
                std::cout << array[offset + i];
            }
            // std::cout << array[offset + i];
            if (i != dimensions[dim] - 1)
                std::cout << ", ";
        }
        std::cout << "]";
    }
    else {
        std::cout << "[";
        int step = 1;
        for(int d = dim + 1; d < dimensions.size(); ++d)
            step *= dimensions[d];
        for (int i = 0; i < dimensions[dim]; ++i) {
            printArray_(array, dimensions, dim + 1, offset + i * step);
            if (i != dimensions[dim] - 1)
                std::cout << ",\n" << std::string(dim * 2 + 1, ' '); // Indentation
        }
        std::cout << "]";
    }
}

// Fonction wrapper
template <typename T, typename U>
void printArray(T *array, const std::vector<U>& dimensions) {
    int dim_total_size = 0;
    for (int i = 0; i < dimensions.size(); i++) {
        if (i == 0) dim_total_size = dimensions[i];
        else dim_total_size *= dimensions[i];
    }

    printArray_(array, dimensions, 0, 0);

    // print dimensions
    std::cout << " (";
    for (int i = 0; i < dimensions.size() - 1; i++) {
        std::cout << dimensions[i] << ", ";
    }
    std::cout << dimensions[dimensions.size()-1]<< "), tot_size = " << dim_total_size << std::endl;
}

template <typename T, typename U>
void print_array_numbers_from_host(std::string name, T *array, const std::vector<U>& dimensions) {
    // std::cout << "HOST "<< name << ": [";
    // for (int i = 0; i < N-1; i++) {
    //     std::cout << array[i] << ", ";
    // }
    // std::cout << array[N-1] << "]" << " size = " << N << std::endl;
    if (dimensions.size() <= 1) std::cout << "HOST " << name << " ";
    else std::cout << "HOST " << name << std::endl;
    printArray(array, dimensions);
}

template <typename T, typename U>
void print_array_numbers_from_device(const std::string& name, T *array, const std::vector<U>& dimensions) {
    int dim_total_size = 0;
    for (int i = 0; i < dimensions.size(); i++) {
        if (i == 0) dim_total_size = dimensions[i];
        else dim_total_size *= dimensions[i];
    }

    T *h_array = (T*)malloc(dim_total_size * sizeof(T));
    cudaMemcpy(h_array, array, dim_total_size * sizeof(T), cudaMemcpyDeviceToHost);

    if (dimensions.size() <= 1) std::cout << "DEVICE " << name << " ";
    else std::cout << "DEVICE " << name << std::endl;
    printArray(h_array, dimensions);
    free(h_array);
}

// TOOLS FUNCTIONS END


info_runner* update_info_runner_device(info_runner *info_host) {
    info_runner *info_device;
    cudaError_t error = cudaMalloc(&info_device, sizeof(info_runner));
    if (error != cudaSuccess) {
        std::cerr << "Erreur lors de l'allocation de la mémoire pour info_device : " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(info_device, info_host, sizeof(info_runner), cudaMemcpyHostToDevice);
    return info_device;
}

void init_info_runner(info_runner *info_device, info_runner *info_host, int nb_episodes, int nb_networks, int nb_neurons, int nb_synapses, int nb_steps, int nb_input, int nb_output, int nb_hidden, bool is_online, bool is_SL) {

    info_host->nb_episodes = nb_episodes;
    info_host->nb_networks = nb_networks;
    info_host->nb_neurons = nb_neurons;
    info_host->nb_synapses = nb_synapses;
    info_host->nb_steps = nb_steps;
    info_host->nb_neurons_all_episodes = nb_episodes * nb_networks * nb_neurons;

    info_host->nb_input = nb_input;
    info_host->nb_output = nb_output;
    info_host->nb_hidden = nb_hidden;

    info_host->is_augmented = false;
    info_host->is_energy = false;
    info_host->is_online = is_online;
    info_host->is_SL = is_SL;

    // int device = 0;
    // cudaDeviceProp deviceProp;
    // cudaError_t err = cudaGetDeviceProperties(&deviceProp, device);
    // if (err != cudaSuccess) {
    //     std::cerr << "Erreur lors de l'obtention des propriétés du GPU : " << cudaGetErrorString(err) << std::endl;
    //     exit(EXIT_FAILURE);
    // }

    // info_host->block_size_neurons = deviceProp.maxThreadsPerBlock;
    // info_host->block_size_synapses = deviceProp.maxThreadsPerBlock;
    cudaMemcpy(info_device, info_host, sizeof(info_runner), cudaMemcpyHostToDevice);
}

template <typename T>
__global__ void set_array_device_with_value_kernel(T *array, T value, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) {
        array[i] = value;
    }
}

template<typename T>
void set_array_device_with_value(T *array, T value, int N, bool is_sync) {
    int block_size = 256;
    int grid_size = (int)((N + block_size - 1) / block_size);
    if (block_size > N) {  // adapt block size to number of elements
        block_size = N; 
        grid_size = 1;
    }

    set_array_device_with_value_kernel<<<grid_size, block_size>>>(array, value, N);
    if (is_sync == true) cudaDeviceSynchronize();
}

template <typename T>
__global__ void set_array_device_with_value_with_delay_kernel(T *array, T value, int nb_steps, int delay, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) {
        if ((index % nb_steps) >= delay) 
            array[i] = value;
    }
}

template<typename T>
void set_array_device_with_value_with_delay(T *array, T value, int N, int nb_steps, int delay, bool is_sync = true) {
    int block_size = 256;
    int grid_size = (int)((N + block_size - 1) / block_size);
    if (block_size > N) {  // adapt block size to number of elements
        block_size = N; 
        grid_size = 1;
    }

    set_array_device_with_value_with_delay_kernel<<<grid_size, block_size>>>(array, value, nb_steps, delay, N);
    if (is_sync) cudaDeviceSynchronize();
}


// SNN FUNCTIONS
SNN* SNN_malloc(info_runner *info, SNN *snn_device, SNN *snn_host, bool is_refrac, bool is_delay, bool is_SL = false) {

    // Initialiser les pointeurs à NULL pour un nettoyage sécurisé en cas d'erreur
    snn_host->voltages = snn_host->thresholds = snn_host->tau = snn_host->weights = NULL;
    snn_host->spikes_rec = NULL;

    // Calcul des tailles nécessaires
    int voltage_size = info->nb_episodes * info->nb_networks * info->nb_neurons * info->nb_steps * sizeof(float);
    int input_size;
    if (is_SL == true) input_size =   info->nb_episodes *         1         * info->nb_input   * info->nb_steps * sizeof(float); // Supervised learning
    else               input_size =   info->nb_episodes * info->nb_networks * info->nb_input   * info->nb_steps * sizeof(float); // Reinforcement learning
    int neuron_size = info->nb_networks * info->nb_neurons * sizeof(float);
    int weight_size = info->nb_networks * (info->nb_neurons * info->nb_neurons) * sizeof(float);
    
    int spike_rec_size =          info->nb_episodes * info->nb_networks * info->nb_neurons * info->nb_steps * sizeof(int);
    int spike_rec_inference_size = info->nb_episodes * info->nb_networks * info->nb_neurons * sizeof(bool);
    
    // Allocate memory for neurons on the device
    cudaMalloc_with_error_check("Input_Data", &(snn_host->input_data), input_size);

    // Voltages
    cudaMalloc_with_error_check("Voltages", &(snn_host->voltages), voltage_size);

    // Neurons
    cudaMalloc_with_error_check("Thresholds", &(snn_host->thresholds), neuron_size);
    cudaMalloc_with_error_check("Tau", &(snn_host->tau), neuron_size);
    if (is_refrac == true) {
        cudaMalloc_with_error_check("Refractory", &(snn_host->refrac), neuron_size);
        cudaMalloc_with_error_check("Refractory_Active", &(snn_host->refrac_active), spike_rec_inference_size);
    }


    // Synapses
    cudaMalloc_with_error_check("Weights", &(snn_host->weights), weight_size);
    if (is_delay == true)
        cudaMalloc_with_error_check("Delays", &(snn_host->delays), weight_size * sizeof(int));


    // Indexes
    cudaMalloc_with_error_check("Synapses_actives_idx_in", &(snn_host->synapses_actives_idx_in), info->nb_synapses * sizeof(int));
    cudaMalloc_with_error_check("Synapses_actives_idx_out", &(snn_host->synapses_actives_idx_out), info->nb_synapses * sizeof(int));
    cudaMalloc_with_error_check("Input_idx", &(snn_host->input_idx),  info->nb_input * sizeof(int));
    cudaMalloc_with_error_check("Output_idx", &(snn_host->output_idx), info->nb_output * sizeof(int));
    cudaMalloc_with_error_check("Hidden_idx", &(snn_host->hidden_idx), info->nb_hidden * sizeof(int));

    // Recording    
    if (snn_host->is_record_spikes == true) cudaMalloc_with_error_check("Spikes_rec", &(snn_host->spikes_rec), spike_rec_size); // No need to record spikes if we use augmented
    cudaMalloc_with_error_check("Spikes_rec_inference", &(snn_host->spikes_rec_inference), spike_rec_inference_size);
    
    
    // Copy SNN struct from host to device
    cudaMemcpy(snn_device, snn_host, sizeof(SNN), cudaMemcpyHostToDevice);

    // print_gpu_memory_usage("SNN_malloc");
    return snn_device;
}

void* SNN_malloc_Augmented(void *snn_host_device_ptr, float *spikes_importance_augmented, int spike_max_time_step_augmented, bool is_neurons_update_with_augmented, bool is_voltage_reset_augmented, int spikes_format_type_augmented, int importance_type_augmented) {
    SNN_HOST_DEVICE *snn_host_device = (SNN_HOST_DEVICE*)snn_host_device_ptr;
    SNN *snn_device = snn_host_device->snn_device;
    SNN *snn_host = snn_host_device->snn_host;
    info_runner *info_host = snn_host_device->info_host;

    int voltage_augmented_size = info_host->nb_episodes * info_host->nb_networks * info_host->nb_neurons;
    int spike_rec_augmented_size = info_host->nb_episodes * info_host->nb_networks * info_host->nb_neurons * info_host->nb_steps;

    if (info_host->is_re_alloc == true) { // Allocate memory for augmented
        cudaMalloc_with_error_check("Voltages_Augmented", &(snn_host->voltages_augmented), voltage_augmented_size * sizeof(float));
        cudaMalloc_with_error_check("Spikes_rec_augmented", &(snn_host->spikes_rec_augmented), spike_rec_augmented_size * sizeof(float));
        cudaMalloc_with_error_check("Spikes_importance_Augmented", &(snn_host->spikes_importance_augmented), (info_host->nb_steps-1) * sizeof(float));
        cudaMalloc_with_error_check("Spikes_importanc_idx", &(snn_host->spikes_importance_idx_augmented), info_host->nb_neurons_all_episodes * sizeof(int));
    }

    info_host->is_augmented = true;
    snn_host->spike_max_time_step_augmented =  static_cast<float>(spike_max_time_step_augmented);
    snn_host->is_neurons_update_with_augmented = is_neurons_update_with_augmented;
    snn_host->is_voltage_reset_augmented = is_voltage_reset_augmented;
    snn_host->spikes_format_type_augmented = spikes_format_type_augmented;
    snn_host->importance_type_augmented = importance_type_augmented;


    set_array_device_with_value(snn_host->voltages_augmented, static_cast<float>(0.0), voltage_augmented_size, false);
    set_array_device_with_value(snn_host->spikes_rec_augmented, static_cast<float>(0.0), spike_rec_augmented_size, false);
    set_array_device_with_value(snn_host->spikes_importance_idx_augmented, -1, info_host->nb_neurons_all_episodes, false);
    cudaMemcpy(snn_host->spikes_importance_augmented, spikes_importance_augmented, (info_host->nb_steps-1) * sizeof(float), cudaMemcpyHostToDevice);

    // Update Host info to Device
    cudaMemcpy(snn_device, snn_host, sizeof(SNN), cudaMemcpyHostToDevice);
    cudaMemcpy(snn_host_device->info_device, info_host, sizeof(info_runner), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    return snn_host_device;
}

void SNN_free_Augmented(void *snn_host_device_ptr) {
    SNN_HOST_DEVICE *snn_host_device = (SNN_HOST_DEVICE*)snn_host_device_ptr;
    SNN *snn_host = snn_host_device->snn_host;

    cudaFree(snn_host->voltages_augmented);
    cudaFree(snn_host->spikes_rec_augmented);
    cudaFree(snn_host->spikes_importance_augmented);
    cudaFree(snn_host->spikes_importance_idx_augmented);
}

void* SNN_malloc_Energy(void *snn_host_device_ptr, float *energy_from_cython, int *energy_idx_from_cython, int energy_length, float *energy_weight, int energy_update_method) {
    SNN_HOST_DEVICE *snn_host_device = (SNN_HOST_DEVICE*)snn_host_device_ptr;
    SNN *snn_host = snn_host_device->snn_host;
    SNN *snn_device = snn_host_device->snn_device;
    info_runner *info_host = snn_host_device->info_host;

    // Allocate memory for energy on the device
    int energy_size = info_host->nb_episodes * info_host->nb_networks * energy_length * info_host->nb_neurons;
    int energy_idx_size = info_host->nb_episodes * info_host->nb_networks * info_host->nb_neurons;
    int weight_size = info_host->nb_networks * (info_host->nb_neurons * info_host->nb_neurons);

    if (info_host->is_re_alloc == true) {
        cudaMalloc_with_error_check("Energy", &(snn_host->energy), energy_size * sizeof(float));
        cudaMalloc_with_error_check("Energy_idx", &(snn_host->energy_idx), energy_idx_size * sizeof(int));
    }

    
    // Copy cython to device
    info_host->is_energy = true;
    // snn_host->energy_from_cython = energy_from_cython; // keep the pointer to use it in the run_SNN function
    cudaMemcpy(snn_host->energy, energy_from_cython, energy_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(snn_host->energy_idx, energy_idx_from_cython, energy_idx_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(snn_host->weights, energy_weight, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    snn_host->energy_length = energy_length;
    snn_host->energy_update_method = energy_update_method;

    // Update SNN struct on the device
    cudaMemcpy(snn_device, snn_host, sizeof(SNN), cudaMemcpyHostToDevice);
    cudaMemcpy(snn_host_device->info_device, info_host, sizeof(info_runner), cudaMemcpyHostToDevice);
    return snn_host_device;
}

void SNN_free_Energy(void *snn_host_device_ptr) {
    SNN_HOST_DEVICE *snn_host_device = (SNN_HOST_DEVICE*)snn_host_device_ptr;
    SNN *snn_host = snn_host_device->snn_host;

    cudaFree(snn_host->energy);
    cudaFree(snn_host->energy_idx);
}

void SNN_free(void *snn_host_device_ptr) {
    SNN_HOST_DEVICE *snn_host_device = (SNN_HOST_DEVICE*)snn_host_device_ptr;
    SNN *snn_host = snn_host_device->snn_host;
    SNN *snn_device = snn_host_device->snn_device;
    info_runner *info_host = snn_host_device->info_host;
    info_runner *info_device = snn_host_device->info_device;

    // input data
    cudaFree(snn_host->input_data);

    // Free Neurons
    cudaFree(snn_host->voltages);
    cudaFree(snn_host->thresholds);
    cudaFree(snn_host->tau);
    // cudaFree(snn_host->constant_current);

    if (snn_host->is_refractory == true) {
        cudaFree(snn_host->refrac);
        cudaFree(snn_host->refrac_active);
    }

    // Free Synapses
    cudaFree(snn_host->weights);

    if (snn_host->is_delay == true)
        cudaFree(snn_host->delays);


    // Free indexes
    cudaFree(snn_host->synapses_actives_idx_in);
    cudaFree(snn_host->synapses_actives_idx_out);
    // cudaFree(snn_host->synapses_actives_first_neuron_idx);
    cudaFree(snn_host->input_idx);
    cudaFree(snn_host->output_idx);
    cudaFree(snn_host->hidden_idx);

    // Free recording
    if (snn_host->spikes_rec != NULL)
        cudaFree(snn_host->spikes_rec);
    cudaFree(snn_host->spikes_rec_inference);

    // Free augmented
    if (info_host->is_augmented == true)
        SNN_free_Augmented(snn_host_device);

    // Free energy
    if (info_host->is_energy == true)
        SNN_free_Energy(snn_host_device);

    // Free struct
    cudaFree(snn_device);
    cudaFree(info_device);
    free(snn_host);
    free(info_host);
    free(snn_host_device);
}

void init_SNN_host_device(SNN *snn_host, SNN *snn_device, 
                float *threshold_host,
                float *tau_host,
                float *weight_host,
                
                int *refrac_host,
                int *delay_host,

                int *synapses_idx_in_host,
                int *synapses_idx_out_host,
                
                int *inputs_idx,
                int *outputs_idx,
                int *hiddens_idx,

                int nb_input,
                int nb_output,
                int nb_hidden,

                int voltage_size,
                int weight_size,
                int synapses_size,
                int nb_neurons_one_network,
                int nb_neurons_one_episode,
                int nb_neurons_all_episodes,

                bool is_LIF_beta,
                bool is_refractory,
                bool is_delay,
                bool is_record_spikes,
                float dt
                ) {

    // 1 - Transfer data from host to device
    cudaMemcpy(snn_host->thresholds, threshold_host, nb_neurons_one_episode * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(snn_host->tau, tau_host, nb_neurons_one_episode * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(snn_host->weights, weight_host, weight_size * sizeof(float), cudaMemcpyHostToDevice);

    if (is_refractory == true){
        cudaMemcpy(snn_host->refrac, refrac_host, nb_neurons_one_episode * sizeof(int), cudaMemcpyHostToDevice);
        set_array_device_with_value(snn_host->refrac_active, static_cast<int>(0), nb_neurons_all_episodes, false);
    } else
        snn_host->refrac = nullptr;
        snn_host->refrac_active = nullptr;

    if (is_delay == true)
        cudaMemcpy(snn_host->delays, delay_host, weight_size * sizeof(int), cudaMemcpyHostToDevice);
    else
        snn_host->delays = nullptr;

    // 2 - Transfer indexes
    cudaMemcpy(snn_host->synapses_actives_idx_in, synapses_idx_in_host, synapses_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(snn_host->synapses_actives_idx_out, synapses_idx_out_host, synapses_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(snn_host->input_idx, inputs_idx, nb_input * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(snn_host->output_idx, outputs_idx, nb_output * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(snn_host->hidden_idx, hiddens_idx, nb_hidden * sizeof(int), cudaMemcpyHostToDevice);

    // 3 - Set spike recording to 0
    // set_array_device_with_value(snn_host->spikes_rec_inference, false, nb_neuron_one_network, false); // true for syncronization of all threads
    set_array_device_with_value(snn_host->spikes_rec_inference, false, nb_neurons_all_episodes, false); // true for syncronization of all threads

    snn_host->is_LIF_beta = is_LIF_beta;
    snn_host->is_refractory = is_refractory;
    snn_host->is_delay = is_delay;
    snn_host->is_record_spikes = is_record_spikes;
    snn_host->dt = dt;

    // 4 - Update SNN struct on the device
    cudaMemcpy(snn_device, snn_host, sizeof(SNN), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

__device__ void input_data_GPU(SNN *snn, info_runner *info, int step, int v_input_idx, int input_idx, int neuron_idx) {
    // 1 - Check if we are not out of bounds
    if (neuron_idx >= info->nb_input) return;
    if (step + 1 >= info->nb_steps) return;

    // 2 - input_data to voltage
    snn->voltages[v_input_idx] = snn->input_data[input_idx];
}


__device__ void LIF_update_GPU(SNN *snn, info_runner *info, int step, int neuron, int v_idx, int v_next_idx, int neuron_idx, int spike_rec_idx) {

    // 0 - Update Active Refractory period
    if (snn->is_refractory == true && snn->refrac_active[spike_rec_idx] > 0) {
        snn->refrac_active[spike_rec_idx] -= 1;
        if (snn->refrac_active[spike_rec_idx] < 0) snn->refrac_active[spike_rec_idx] = 0;
        snn->voltages[v_next_idx] = 0.0;
        snn->spikes_rec_inference[spike_rec_idx] = false;
        return; // No need to update LIF if refractory period is active
    }

    // 1 - If Spike, reset voltage
    if (snn->voltages[v_idx] > snn->thresholds[neuron_idx]) {
        snn->voltages[v_next_idx] = 0.0; // Hard reset voltage
        snn->spikes_rec_inference[spike_rec_idx] = true;

        // 1.1 - Set Refractory period
        if (snn->is_refractory == true)
            snn->refrac_active[spike_rec_idx] = snn->refrac[neuron_idx];

        // 1.2 - Record spikes (normal or augmented)
        if (info->is_augmented && neuron < info->nb_input) {
            snn->spikes_rec_augmented[v_idx] += 1.0;
        }
        else if (snn->is_record_spikes == true)
            snn->spikes_rec[v_idx] += 1;
    } 
    else { // 2 - If no spike, LIF update
        float v_next = snn->voltages[v_next_idx] + snn->voltages[v_idx];
        // printf("v_next: %f, constant_current: %f, tau: %f, dt: %f\n", v_next, snn->constant_current[neuron_idx], snn->tau[neuron_idx], snn->dt);
        // snn->voltages[v_next_idx] = v_next + (-v_next + snn->constant_current[neuron_idx]) / snn->tau[neuron_idx] * snn->dt; // V = V + (-V + I) / tau * dt
        // snn->voltages[v_next_idx] = v_next + -v_next / snn->tau[neuron_idx] * snn->dt; // V = V + -V / tau * dt; No constant current in this version
        snn->voltages[v_next_idx] = v_next + -v_next / snn->tau[neuron_idx]; // V = V + -V / tau * dt; No constant current and dt (dt=1.0) in this version
        snn->spikes_rec_inference[spike_rec_idx] = false;
    }

}


__global__ void input_data_and_LIF_update_GPU(SNN *snn, info_runner *info, int step, bool is_SL) {
    // int nb_episodes = info->nb_episodes;
    int nb_networks = info->nb_networks;
    int nb_neurons  = info->nb_neurons;
    int nb_input    = info->nb_input;
    int nb_steps    = info->nb_steps;

    int index = blockIdx.x * blockDim.x + threadIdx.x; // thread index (here we have one thread per neuron)

    // 1 - Check if we are not out of bounds
    if (step >= nb_steps) return;    
    if (index >= info->nb_neurons_all_episodes) return;

    // 2 - Get episode, network, neuron indexes
    int rem     = index % (nb_networks * nb_neurons);
    int episode = index / (nb_networks * nb_neurons);
    int network = rem / nb_neurons;
    int neuron  = rem % nb_neurons;

    // 3 - Input data if input neuron
    if (neuron < nb_input) {
        int episode_network_idx = ((episode * nb_networks) + network);
        int v_input_idx = (episode_network_idx * nb_neurons + neuron) * nb_steps + step;
        int input_idx;
        if (is_SL == true) input_idx = (episode * nb_input + neuron) * nb_steps + step; // Supervised learning
        else               input_idx = (episode_network_idx * nb_input + neuron) * nb_steps + step; // Reinforcement learning
        input_data_GPU(snn, info, step, v_input_idx, input_idx, neuron);
    }


    // 4 - LIF update (Here I just not update for the last step for having a better tracing of the voltage (good for debugging))
    if (step + 1 >= nb_steps) return;

    // 4.1 - LIF update
    int v_idx = (((episode * nb_networks) + network) * nb_neurons + neuron) * nb_steps + step;
    int v_next_idx = v_idx + 1;
    int neuron_idx = network * nb_neurons + neuron;
    int spike_rec_idx = ((episode * nb_networks) + network) * nb_neurons + neuron;
    LIF_update_GPU(snn, info, step, neuron, v_idx, v_next_idx, neuron_idx, spike_rec_idx);
}

void input_data_and_LIF_update(SNN *snn, info_runner *info_device, info_runner *info_host, int step, bool is_SL = false) {
    // int block_size = info_host->block_size_neurons;
    int block_size = 256;
    int grid_size = (int)((info_host->nb_neurons_all_episodes + block_size - 1) / block_size);
    if (block_size > info_host->nb_neurons_all_episodes) {  // adapt block size to number of elements
        block_size = info_host->nb_neurons_all_episodes; 
        grid_size = 1;
    }
    
    input_data_and_LIF_update_GPU<<<grid_size, block_size>>>(snn, info_device, step, is_SL);
    cudaDeviceSynchronize();

}

__global__ void synapse_update_GPU(SNN *snn, info_runner *info, int step) {

    int nb_networks = info->nb_networks;
    int nb_neurons = info->nb_neurons;
    int nb_steps = info->nb_steps;

    int index = blockIdx.x * blockDim.x + threadIdx.x; // thread index (here we have one thread per neuron)

    // Vérification pour éviter de sortir du tableau
    if (step + 1 >= nb_steps) return;
    if (index >= info->nb_neurons_all_episodes) return;


    // Récupération (episode, network, neuron)
    int rem = index % (nb_networks * nb_neurons);
    int episode = index / (nb_networks * nb_neurons);
    int network = rem / nb_neurons;
    int neuron = rem % nb_neurons;

    // Calcul de l'index pour acceder aux voltages
    // int v_idx_next = (((episode * nb_networks) + network) * nb_neurons + neuron) * nb_steps + step + 1;
    // int spike_rec_idx = ((episode * nb_networks) + network) * nb_neurons + neuron;
    int v_idx_pre_compute = ((episode * nb_networks) + network) * nb_neurons;
    int spike_rec_idx_pre_compute = ((episode * nb_networks) + network) * nb_neurons;
    int weight_idx = network * nb_neurons * nb_neurons;

    int syn_idx = snn->synapses_actives_first_neuron_idx[neuron];
    int syn_idx_end = snn->synapses_actives_first_neuron_idx[neuron + 1];
    if (neuron == nb_neurons - 1) syn_idx_end = info->nb_synapses;

    // printf("weight_idx: %d, syn_idx: %d, syn_idx_end: %d, spike_rec_idx_pre_compute: %d\n", weight_idx, syn_idx, syn_idx_end, spike_rec_idx_pre_compute);
    for (int i = syn_idx, syn_in, syn_out, v_idx_next, spike_rec_idx, w_idx; i < syn_idx_end; i++) {

        syn_in = snn->synapses_actives_idx_in[i];
        syn_out = snn->synapses_actives_idx_out[i];
        v_idx_next = (v_idx_pre_compute + syn_out) * nb_steps + step + 1;
        spike_rec_idx = spike_rec_idx_pre_compute + syn_in;
        // w_idx = weight_idx + syn_in * nb_neurons + syn_out; # L'un ou l'autre depend de la structure de la matrice de poids from numpy/cython
        w_idx = weight_idx + syn_out * nb_neurons + syn_in;

        if (snn->spikes_rec_inference[spike_rec_idx] == true) {
            snn->voltages[v_idx_next] += snn->weights[w_idx];
        //    printf("i: %d, syn_in: %d, syn_out: %d, weight_idx: %d, spike_rec_idx: %d\n", i, syn_in, syn_out, w_idx, spike_rec_idx);
        }

    }

}

__global__ void synapse_update_atomic_GPU(SNN *snn, info_runner *info, int step) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int nb_networks = info->nb_networks;
    int nb_neurons = info->nb_neurons;
    int nb_synapses = info->nb_synapses;
    int nb_steps = info->nb_steps;

    // 1 - Check if we are not out of bounds
    if (step + 1 >= nb_steps) return;
    if (index >= info->nb_episodes * nb_networks * nb_synapses) return;


    // 2 - Get episode, network, neurons indexes
    int total_per_episode = nb_networks * nb_synapses;
    int episode = index / total_per_episode;
    int rem = index % total_per_episode;
    int network = rem / nb_synapses;
    int synapse_idx = rem % nb_synapses;

    int pre_neuron = snn->synapses_actives_idx_in[synapse_idx];
    int post_neuron = snn->synapses_actives_idx_out[synapse_idx];

    // 3 - Check if pre-synaptic neuron spiked - if not no need to update post-synaptic neuron
    int pre_spike_idx = (((episode * nb_networks) + network) * nb_neurons + pre_neuron);
    if (!snn->spikes_rec_inference[pre_spike_idx]) return;  // Check if pre-synaptic neuron spiked

    // int pre_spike_rec_idx = (episode * nb_networks * nb_neurons * nb_steps) + (network * nb_neurons * nb_steps) + (pre_neuron * nb_steps) + step;
    // if (snn->spikes_rec[pre_spike_rec_idx] == 0) return;  // Check if pre-synaptic neuron spiked
    // printf("pre_spike_rec_idx: %d, pre_spike: %f\n", pre_spike_rec_idx, snn->spikes_rec[pre_spike_rec_idx]);

    
    // 4 Check if post and pre synaptic neurons are not in refractory period
    int network_neurons = (episode * nb_networks + network) * nb_neurons;
    if (snn->is_refractory == true && (snn->refrac_active[network_neurons+post_neuron] > 0 || snn->refrac_active[network_neurons+pre_neuron] > 0)) return;

    // 5 - Get weight index of connection between pre and post synaptic neurons
    int weight_idx = (network * nb_neurons * nb_neurons) + (pre_neuron * nb_neurons) + post_neuron;
    // int weight_idx = (network * nb_neurons * nb_neurons) + (post_neuron * nb_neurons) + pre_neuron;

    // 6 - if delay get delay value of the synapse
    int delay = 0;
    if (snn->is_delay == true) {
        if (delay + step + 1 >= nb_steps) return; // Check if we are not out of bounds
        delay = snn->delays[weight_idx];
    }
    // int delay = (info->is_delay == true) ? snn->delays[weight_idx] : 0;
    // if (delay + step + 1 >= nb_steps) return;

    // 7 - Get post-synaptic neuron voltage index at next step (step + 1)
    int post_v_idx_next = (((episode * nb_networks) + network) * nb_neurons + post_neuron) * nb_steps + (step + 1);


    // 8 - Energy update
    int energy_dim;
    int energy_offset;
    if (info->is_energy) {
        energy_dim = snn->energy_idx[episode*(nb_networks*nb_neurons) + network*nb_neurons + pre_neuron];
        energy_offset = episode*(nb_networks * snn->energy_length * nb_neurons) + network*(snn->energy_length * nb_neurons) + energy_dim*(nb_neurons)+ pre_neuron;

        atomicAdd(&(snn->voltages[post_v_idx_next + delay]), snn->weights[weight_idx] * snn->energy[energy_offset]);
    } else
        atomicAdd(&(snn->voltages[post_v_idx_next + delay]), snn->weights[weight_idx]);


    // 7 - Augmented update
    int spike_rec_idx;
    if (info->is_augmented) {
        spike_rec_idx = post_v_idx_next - 1;
        // neuron_idx = (((episode * nb_networks) + network) * nb_neurons + post_neuron);
        // printf("energy_offset: %d, energy_dim: %d, pre_neuron: %d, post_neuron: %d, weight_idx: %d, post_v_idx_next: %d, neuron_idx: %d, spike_rec_idx: %d\n", energy_offset, energy_dim, pre_neuron, post_neuron, weight_idx, post_v_idx_next, neuron_idx, spike_rec_idx);

        if (snn->is_neurons_update_with_augmented == true && snn->spikes_rec_augmented[spike_rec_idx] > 0.0) { // 7.1 Update the voltage with the augmented weight
            if (info->is_energy)
                // atomicAdd(&(snn->voltages_augmented[neuron_idx]), (snn->weights[weight_idx] * snn->spikes_rec_augmented[spike_rec_idx] * snn->energy[energy_offset]));
                atomicAdd(&(snn->spikes_rec_augmented[post_v_idx_next + delay]), (snn->weights[weight_idx] * snn->spikes_rec_augmented[spike_rec_idx] * snn->energy[energy_offset]));
            else
                // atomicAdd(&(snn->voltages_augmented[neuron_idx]), (snn->weights[weight_idx] * snn->spikes_rec_augmented[spike_rec_idx]));
                atomicAdd(&(snn->spikes_rec_augmented[post_v_idx_next + delay]), (snn->weights[weight_idx] * snn->spikes_rec_augmented[spike_rec_idx]));

        } else { // 7.2 Update the voltage with the normal weight
            if (info->is_energy) {
                // printf("energy: %f, weight: %f, added: %f\n", snn->energy[energy_offset], snn->weights[weight_idx], snn->weights[weight_idx] * snn->energy[energy_offset]);
                // atomicAdd(&(snn->voltages_augmented[neuron_idx]), (snn->weights[weight_idx] * snn->energy[energy_offset]));
                atomicAdd(&(snn->spikes_rec_augmented[post_v_idx_next + delay]), (snn->weights[weight_idx] * snn->energy[energy_offset]));
            }
            else
                // atomicAdd(&(snn->voltages_augmented[neuron_idx]), snn->weights[weight_idx]);
                atomicAdd(&(snn->spikes_rec_augmented[post_v_idx_next + delay]), snn->weights[weight_idx]);
        } 
    }
}

void synapse_update(SNN *snn, info_runner *info_device, info_runner *info_host, int step) {
    // int block_size = info_host->block_size_synapses;
    int block_size = 256;
    int total_synapses = info_host->nb_episodes * info_host->nb_networks * info_host->nb_synapses;
    int grid_size = (int)((total_synapses + block_size - 1) / block_size);
    if (block_size > total_synapses) {  // adapt block size to number of elements
        block_size = total_synapses; 
        grid_size = 1;
    }    
    synapse_update_atomic_GPU<<<grid_size, block_size>>>(snn, info_device, step);
    cudaDeviceSynchronize();
}


__global__ void synapse_update_atomic_GPU_2(SNN *snn, info_runner *info, int step, int nb_syn_kernel_to_update) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int nb_networks = info->nb_networks;
    int nb_neurons  = info->nb_neurons;
    int nb_synapses = info->nb_synapses;
    int nb_episodes = info->nb_episodes;
    int nb_steps    = info->nb_steps;
    int total_per_episode = nb_networks * nb_synapses;

    // 1 - Check if we are not out of bounds and Get the Start synapse index
    int total_syn = nb_episodes * nb_networks * nb_synapses;
    int start_syn = index * nb_syn_kernel_to_update;
    if (step + 1 >= nb_steps) return;
    if (start_syn >= total_syn) return;

    // 2 - Get the End synapse index
    int end_syn = start_syn + nb_syn_kernel_to_update;
    if (end_syn > total_syn) {
        end_syn = total_syn; 
    }

    // 3 - Loop over the synapses and Update the neurons voltages if pre-synaptic neuron spiked
    for (int syn_global = start_syn; syn_global < end_syn; syn_global++) {
        // 3.1 Get episode, network, synapse indexes
        int episode = syn_global / total_per_episode;
        int rem = syn_global % total_per_episode;
        int network = rem / nb_synapses;
        int syn = rem % nb_synapses;

        // 3.2 Get pre and post synaptic neurons indexes
        int pre_neuron  = snn->synapses_actives_idx_in[syn];
        int post_neuron = snn->synapses_actives_idx_out[syn];

        // 3.3 Check if pre-synaptic neuron spiked - if not no need to update post-synaptic neuron and continue to the next synapse
        int pre_spike_idx = (((episode * nb_networks) + network) * nb_neurons + pre_neuron);
        if (!snn->spikes_rec_inference[pre_spike_idx]) {
            continue;
        }

        // 3.4 Get weight index of connection between pre and post synaptic neurons
        int weight_idx = (network * nb_neurons * nb_neurons) + (pre_neuron * nb_neurons) + (post_neuron);

        int delay = 0;
        if (snn->is_delay == true) {
            if (delay + step + 1 >= nb_steps) continue; // Check if we are not out of bounds
            delay = snn->delays[weight_idx];
        }

        // 3.5 Get post-synaptic neuron voltage index at next step (step + 1)
        int post_v_idx_next = (((episode * nb_networks) + network) * nb_neurons + post_neuron) * nb_steps + (step + 1);

        // 3.6 Energy update
        int energy_offset;
        if (info->is_energy) {
            int energy_dim = snn->energy_idx[episode*(nb_networks*nb_neurons) + network*nb_neurons + pre_neuron];
            energy_offset = episode*(nb_networks * snn->energy_length * nb_neurons) + network*(snn->energy_length * nb_neurons) + energy_dim*(nb_neurons) + pre_neuron;

            atomicAdd(&(snn->voltages[post_v_idx_next + delay]), snn->weights[weight_idx] * snn->energy[energy_offset]);
        } else {
            atomicAdd(&(snn->voltages[post_v_idx_next + delay]), snn->weights[weight_idx]);
        }

        // 3.7 Augmented update
        if (info->is_augmented) {
            int spike_rec_idx = post_v_idx_next - 1;
            // int neuron_idx = (((episode * nb_networks) + network) * nb_neurons + post_neuron);

            if (snn->is_neurons_update_with_augmented && snn->spikes_rec_augmented[spike_rec_idx] > 0.0) { // 3.7.1 Update the voltage with the augmented weight
                if (info->is_energy) {
                    // atomicAdd(&(snn->voltages_augmented[neuron_idx]), snn->weights[weight_idx] * snn->spikes_rec_augmented[neuron_idx] * snn->energy[energy_offset]);
                    atomicAdd(&(snn->spikes_rec_augmented[post_v_idx_next + delay]), snn->weights[weight_idx] * snn->spikes_rec_augmented[spike_rec_idx] * snn->energy[energy_offset]);
                } else {
                    // atomicAdd(&(snn->voltages_augmented[neuron_idx]), snn->weights[weight_idx] * snn->spikes_rec_augmented[neuron_idx]);
                    atomicAdd(&(snn->spikes_rec_augmented[post_v_idx_next + delay]), snn->weights[weight_idx] * snn->spikes_rec_augmented[spike_rec_idx]);
                }
            } else { // 3.7.2 Update the voltage with the classic weight
                if (info->is_energy) {
                    // atomicAdd(&(snn->voltages_augmented[neuron_idx]), snn->weights[weight_idx] * snn->energy[energy_offset]);
                    atomicAdd(&(snn->spikes_rec_augmented[post_v_idx_next + delay]), snn->weights[weight_idx] * snn->energy[energy_offset]);
                } else {
                    // atomicAdd(&(snn->voltages_augmented[neuron_idx]), snn->weights[weight_idx]);
                    atomicAdd(&(snn->spikes_rec_augmented[post_v_idx_next + delay]), snn->weights[weight_idx]);
                }
            }
        }
    }
}

void synapse_update_2(SNN *snn, info_runner *info_device, info_runner *info_host, int step, int nb_synapses_per_kernel)
{
    // 1 - Define the number of synapses to update per kernel
    int nb_syn_kernel_to_update = 2;

    // 2 - Compute the total number of synapses
    int total_synapses = info_host->nb_episodes * info_host->nb_networks * info_host->nb_synapses;

    // 3 - We want grid_size * block_size * nb_syn_kernel_to_update >= total_synapses
    int total_threads = (total_synapses + nb_syn_kernel_to_update - 1) / nb_syn_kernel_to_update;

    int block_size = 256;
    int grid_size  = (total_threads + block_size - 1) / block_size;
    if (block_size > total_threads && total_threads > 0) {
        block_size = total_threads;
        grid_size  = 1;
    }

    synapse_update_atomic_GPU_2<<<grid_size, block_size>>>(snn, info_device, step, nb_syn_kernel_to_update);
    cudaDeviceSynchronize();
}


__device__ void augmented_update_GPU(SNN *snn, info_runner *info, int step, int v_idx, int neuron_idx, int spike_augmented_rec_idx) {

    snn->voltages_augmented[v_idx] += snn->spikes_rec_augmented[spike_augmented_rec_idx];
    snn->spikes_rec_augmented[spike_augmented_rec_idx] = 0.0; // Reset the delayed information as it has been transfered to the voltage_decoder_view which will update the augmented_decoder_view with in augmented_update method
    // 1 - Update the spikes_rec_augmented with the voltage_decoder (LIF Version)  -> (Σ_W/threshold) * (1 - (1/tau) * (1/(1+Refractory)) ; tau -> [1, +inf], refractory -> [0, +inf]
    if (snn->is_LIF_beta == false && snn->is_refractory == false && snn->voltages_augmented[v_idx] > snn->thresholds[neuron_idx])
        snn->spikes_rec_augmented[spike_augmented_rec_idx] = (snn->voltages_augmented[v_idx] / snn->thresholds[neuron_idx]) * (1.0-(1.0/snn->tau[neuron_idx]));
    
    else if (snn->is_LIF_beta == false && snn->is_refractory == true && snn->voltages_augmented[v_idx] > snn->thresholds[neuron_idx])
        snn->spikes_rec_augmented[spike_augmented_rec_idx] = (snn->voltages_augmented[v_idx] / snn->thresholds[neuron_idx]) * (1.0-(1.0/snn->tau[neuron_idx])) * (1.0/(1.0+snn->refrac[neuron_idx]));
    
    // 1 - Update the spike_state_decoder with the voltage_decoder (Beta Version) -> (Σ_W/threshold) * (tau) * (1/(1+Refractory)) ; tau -> [0, 1], refractory -> [0, +inf]
    else if (snn->is_LIF_beta == true && snn->is_refractory == false && snn->voltages_augmented[v_idx] > snn->thresholds[neuron_idx])
        snn->spikes_rec_augmented[spike_augmented_rec_idx] = (snn->voltages_augmented[v_idx] / snn->thresholds[neuron_idx]) * snn->tau[neuron_idx];
    
    else if (snn->is_LIF_beta == true && snn->is_refractory == true && snn->voltages_augmented[v_idx] > snn->thresholds[neuron_idx])
        snn->spikes_rec_augmented[spike_augmented_rec_idx] = (snn->voltages_augmented[v_idx] / snn->thresholds[neuron_idx]) * snn->tau[neuron_idx] * (1.0/(1.0+snn->refrac[neuron_idx]));

    
    float spike_floor = floorf(snn->spikes_rec_augmented[spike_augmented_rec_idx]);
    // float spike_floor = rintf(snn->spikes_rec_augmented[spike_augmented_rec_idx]);
    // printf("v_idx: %d, neuron_idx: %d, spike_augmented_rec_idx: %d, spike_floor: %f, spike_rec_augmented: %f\n", v_idx, neuron_idx, spike_augmented_rec_idx, spike_floor, snn->spikes_rec_augmented[spike_augmented_rec_idx]);

    // 2 - Update the voltage_decoder with the spike_state_decoder (if there is spikes) otherwise decay the voltage
    // 2.1 - Hard-Reset the voltage if there is a spike
    if (snn->is_voltage_reset_augmented == true && spike_floor >= 1.0)
        snn->voltages_augmented[v_idx] = 0.0;

    // 2.2 - Soft-Reset the voltage if there is a spike (keep the remaining exceding voltage above the threshold)
    else if (snn->is_voltage_reset_augmented == false && spike_floor >= 1.0)
        snn->voltages_augmented[v_idx] = snn->spikes_rec_augmented[spike_augmented_rec_idx] - spike_floor;
        // snn->voltages_augmented[v_idx] = snn->voltages_augmented[v_idx] - spike_floor * snn->thresholds[neuron_idx];

    // 2.3 - LIF - Decay the voltage if there is no spike
    else if (snn->is_LIF_beta == false)
        snn->voltages_augmented[v_idx] *= (1.0 - (1.0/snn->tau[neuron_idx]));
    
    // 2.4 - Beta - Decay the voltage if there is no spike
    else if (snn->is_LIF_beta == true)
        snn->voltages_augmented[v_idx] *= snn->tau[neuron_idx];

    
    // 3 - Add the rounded spike to the spike_state_decoder
    // 3.1 - Positive spike only
    if (snn->spikes_format_type_augmented == 0) // Positive spike only
        snn->spikes_rec_augmented[spike_augmented_rec_idx] = spike_floor;
    
    // 3.2 - Abosolute spike only
    else if (snn->spikes_format_type_augmented == 1) // Absolute spike only
        snn->spikes_rec_augmented[spike_augmented_rec_idx] = fabsf(spike_floor);

    // 3.3 - Both positive and negative spikes
    else if (snn->spikes_format_type_augmented == 2) // Both positive and negative spikes
        snn->spikes_rec_augmented[spike_augmented_rec_idx] = spike_floor;

    // 4 - Clip the spike_state_decoder to the maximum time step
    if (snn->spikes_rec_augmented[spike_augmented_rec_idx] > snn->spike_max_time_step_augmented)
        snn->spikes_rec_augmented[spike_augmented_rec_idx] = snn->spike_max_time_step_augmented;


    // 5 - Augmented Importance Update
    if (snn->importance_type_augmented == 0) { // first index
        if (snn->spikes_rec_augmented[spike_augmented_rec_idx] >= 1.0 && snn->spikes_importance_idx_augmented[v_idx] == -1)
            snn->spikes_importance_idx_augmented[v_idx] = 0;

        if (snn->spikes_importance_idx_augmented[v_idx] >= 0) { // if first index of spikes has been found add the importance at each time step
            snn->spikes_rec_augmented[spike_augmented_rec_idx] += ((snn->spikes_rec_augmented[spike_augmented_rec_idx] / snn->spike_max_time_step_augmented) * snn->spikes_importance_augmented[snn->spikes_importance_idx_augmented[v_idx]]);
            snn->spikes_importance_idx_augmented[v_idx] += 1;
        }
    }
    else if (snn->importance_type_augmented == 1) { // by index
        if (snn->spikes_rec_augmented[spike_augmented_rec_idx] >= 1.0 && snn->spikes_importance_idx_augmented[v_idx] == -1)
            snn->spikes_importance_idx_augmented[v_idx] = 0;

        if (snn->spikes_rec_augmented[spike_augmented_rec_idx] >= 1.0) { // add importance at each spikes
            snn->spikes_rec_augmented[spike_augmented_rec_idx] += ((snn->spikes_rec_augmented[spike_augmented_rec_idx] / snn->spike_max_time_step_augmented) * snn->spikes_importance_augmented[snn->spikes_importance_idx_augmented[v_idx]]);
            snn->spikes_importance_idx_augmented[v_idx] += 1;
        }
    }
}

__device__ void energy_update_GPU(SNN *snn, info_runner *info, int step, int spike_rec_idx, int energy_dim, int energy_dim_idx, int energy_offset) {

    if (snn->spikes_rec_inference[spike_rec_idx] == true) {
        
        // 4 - Update the energy_index
        if (energy_dim < snn->energy_length - 1) 
            snn->energy_idx[energy_dim_idx] += 1;
        else 
            snn->energy_idx[energy_dim_idx] = 0;

        // 5 - Energy Update method when there is a spike
        if (snn->energy_update_method == 1) // Ascending
            snn->energy[energy_offset] += snn->energy[energy_offset] * 0.10;

        else if (snn->energy_update_method == 2) // Descending
            snn->energy[energy_offset] -= snn->energy[energy_offset] * 0.10;

        // else if (snn->energy_update_method == 3) // rate (TODO)
        //     snn->energy[energy_offset] = snn->energy[energy_offset];
        // else if (snn->energy_update_method == 4) // weight_acceleration (TODO)
            // snn->energy[energy_offset] = 0.0;
    }

}

__global__ void augmented_and_energy_update_and_input_data_and_LIF_update_GPU(SNN *snn, info_runner *info, int step, bool is_SL) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // thread index (here we have one thread per neuron)
    if (index >= info->nb_neurons_all_episodes) return;
    
    // int nb_episodes = info->nb_episodes;
    int nb_networks = info->nb_networks;
    int nb_neurons = info->nb_neurons;
    int nb_steps = info->nb_steps;

    // 1 - Check if we are not out of bounds
    if (step + 1 >= nb_steps) return;
    if (index >= info->nb_neurons_all_episodes) return;

    // 2 - Get episode, network, neuron indexes
    int rem = index % (nb_networks * nb_neurons);
    int episode = index / (nb_networks * nb_neurons);
    int network = rem / nb_neurons;
    int neuron = rem % nb_neurons;

    // 3 - Augmented Update
    int v_idx, spike_augmented_rec_idx;
    int neuron_idx = (network * nb_neurons) + neuron;
    if (info->is_augmented)  {
        v_idx = (episode * nb_networks * nb_neurons) + (network * nb_neurons) + neuron;
        spike_augmented_rec_idx = (episode * nb_networks * nb_neurons * nb_steps) + (network * nb_neurons * nb_steps) + (neuron * nb_steps) + step + 1; // +1 because we update the next step
        augmented_update_GPU(snn, info, step, v_idx, neuron_idx, spike_augmented_rec_idx);
    }


    // 4 - Energy Update
    int spike_rec_idx = ((episode * nb_networks) + network) * nb_neurons + neuron;
    if (info->is_energy) {
        int energy_dim_idx = episode*(nb_networks*nb_neurons) + network*nb_neurons + neuron;
        int energy_dim = snn->energy_idx[energy_dim_idx];
        int energy_offset = episode*(nb_networks * snn->energy_length * nb_neurons) + network*(snn->energy_length * nb_neurons) + energy_dim*(nb_neurons)+ neuron;
        energy_update_GPU(snn, info, step, spike_rec_idx, energy_dim, energy_dim_idx, energy_offset);
    }

    // __syncthreads();

    
    step += 1; // step+1 because we update the next step

    // 5 - Input Data if input neuron
    if (neuron < info->nb_input) {
        if (step >= nb_steps) return;
        int episode_network_idx = ((episode * nb_networks) + network);
        int v_input_idx = (episode_network_idx * nb_neurons + neuron) * nb_steps + step;
        int input_idx;
        if (is_SL == true) input_idx = (episode * info->nb_input + neuron) * nb_steps + step;
        else               input_idx = (episode_network_idx * info->nb_input + neuron) * nb_steps + step;
        input_data_GPU(snn, info, step, v_input_idx, input_idx, neuron); // step+1 because we update the next step
    }


    //  6 - LIF Update
    if (step + 1 >= nb_steps) return;
    if (index >= info->nb_neurons_all_episodes) return;
    v_idx = (((episode * nb_networks) + network) * nb_neurons + neuron) * nb_steps + step;
    int v_next_idx = v_idx + 1;
    LIF_update_GPU(snn, info, step, neuron, v_idx, v_next_idx, neuron_idx, spike_rec_idx); // step+1 because we update the next step
}

void augmented_and_energy_update_and_input_data_and_LIF_update(SNN *snn, info_runner *info_device, info_runner *info_host, int step, bool is_SL=false, bool is_sync=true) {
    // int block_size = info_host->block_size_neurons;
    int block_size = 256;
    int grid_size = (int)((info_host->nb_neurons_all_episodes + block_size - 1) / block_size);
    if (block_size > info_host->nb_neurons_all_episodes) {  // adapt block size to number of elements
        block_size = info_host->nb_neurons_all_episodes; 
        grid_size = 1;
    }
    
    augmented_and_energy_update_and_input_data_and_LIF_update_GPU<<<grid_size, block_size>>>(snn, info_device, step, is_SL);
    if (is_sync == true) cudaDeviceSynchronize();
}


void transfer_data_host_to_device(SNN *snn_device, SNN *snn_host, info_runner *info_host, float *input_data_host, bool is_Supervised=false) {
    int nb_episodes = info_host->nb_episodes;
    int nb_networks = info_host->nb_networks;
    int nb_input = info_host->nb_input;
    int nb_steps = info_host->nb_steps;

    // Copy input data to device
    if (is_Supervised == true) // Supervised Learning
        cudaMemcpy(snn_host->input_data, input_data_host, nb_episodes *      1      * nb_input * nb_steps * sizeof(float), cudaMemcpyHostToDevice);    
    else // Reinforcement Learning
        cudaMemcpy(snn_host->input_data, input_data_host, nb_episodes * nb_networks * nb_input * nb_steps * sizeof(float), cudaMemcpyHostToDevice);

    // Update snn_device with snn_host
    cudaMemcpy(snn_device, snn_host, sizeof(SNN), cudaMemcpyHostToDevice);

}

void init_and_reset_runner(SNN *snn_host, info_runner *info_device, info_runner *info_host) {
    int nb_episodes = info_host->nb_episodes;
    int nb_networks = info_host->nb_networks;
    int nb_neurons = info_host->nb_neurons;
    int nb_steps = info_host->nb_steps;
    int voltage_size = nb_episodes * nb_networks * nb_neurons * nb_steps;
    int spike_rec_augmented_size = nb_episodes * nb_networks * nb_neurons * nb_steps;


    // Set spike recording to 0
    if (snn_host->is_record_spikes) 
        set_array_device_with_value(snn_host->spikes_rec, static_cast<int>(0), voltage_size, false);

    // set_array_device_with_value(snn_host->spikes_rec_inference, false, nb_neurons, false);
    set_array_device_with_value(snn_host->spikes_rec_inference, false, nb_episodes * nb_networks * nb_neurons, false);

    if (info_host->is_online == false)
        set_array_device_with_value(snn_host->voltages, static_cast<float>(0.0), voltage_size, false);
    else
        // Keep the previous voltage_decoder until the last delay
        set_array_device_with_value_with_delay(snn_host->voltages, static_cast<float>(0.0), voltage_size, info_host->nb_steps, 2, false); // TODO: 2 need to be changed to the maximum delay possible + ADD LIF for the first step

    if (info_host->is_augmented) {
        if (info_host->is_online == false) {
            set_array_device_with_value(snn_host->voltages_augmented, static_cast<float>(0.0), nb_episodes * nb_networks * nb_neurons, false); // I do this cause in the case of online == True, I have to keep the previous voltage_decoder, which is already done at this stage
        }
        set_array_device_with_value(snn_host->spikes_rec_augmented, static_cast<float>(0.0), spike_rec_augmented_size, false);
        set_array_device_with_value(snn_host->spikes_importance_idx_augmented, -1, info_host->nb_neurons_all_episodes, false);
    }

    if (info_host->is_energy && (info_host->is_online == false)) {
        // TO DO Need a find a way to set the energy in online mode --> (snn_host->energy)
        print_array_numbers_from_host("Energy", snn_host->energy, std::vector<int>{nb_episodes, nb_networks, snn_host->energy_length, nb_neurons});
        std::cout << "Energy length: " << snn_host->energy_length << std::endl;
        exit(0);
        if (snn_host->energy_length > 1)
            set_array_device_with_value(snn_host->energy_idx, 0, nb_episodes * nb_networks * nb_neurons, false);
    }

    cudaDeviceSynchronize();
}

void run_SNN_RL(void *snn_host_device_ptr, float *input_data_host) {
    SNN_HOST_DEVICE *snn_host_device = (SNN_HOST_DEVICE*)snn_host_device_ptr;
    SNN *snn_device = snn_host_device->snn_device;
    SNN *snn_host = snn_host_device->snn_host;
    info_runner *info_host = snn_host_device->info_host;
    info_runner *info_device = snn_host_device->info_device;
    bool is_SL = false;


    // std::vector<int> voltage_dim = {info_host->nb_episodes, info_host->nb_networks, info_host->nb_neurons, info_host->nb_steps};
    // std::vector<int> voltage_augmented_dim = {info_host->nb_episodes, info_host->nb_networks, info_host->nb_neurons};
    // std::vector<int> input_dim = {info_host->nb_episodes, info_host->nb_networks, info_host->nb_input, info_host->nb_steps};
    // std::vector<int> spike_augmented_dim = {info_host->nb_episodes, info_host->nb_networks, info_host->nb_neurons, info_host->nb_steps};
    // std::vector<int> neuron_dim = {info_host->nb_networks, info_host->nb_neurons};
    // std::vector<int> synapses_dim = {info_host->nb_networks, info_host->nb_neurons, info_host->nb_neurons};

    // 0 - Transfer data from host to device & init and reset runner
    transfer_data_host_to_device(snn_device, snn_host, info_host, input_data_host, is_SL);
    init_and_reset_runner(snn_host, info_device, info_host);
    
    // 1 - Input Data and LIF update (fusionned) only at the first step (done later in augmented_and_energy_update_and_input_data_and_LIF_update)
    input_data_and_LIF_update(snn_device, info_device, info_host, 0, is_SL);
    for (int step = 0; step < info_host->nb_steps; step++) {
        // 2 - Neuron update from the synapses (spikes)
        synapse_update(snn_device, info_device, info_host, step);
        // synapse_update_2(snn_device, info_device, info_host, step, 20);

        // 3 - Augmented & Energy update & Input data & LIF update
        augmented_and_energy_update_and_input_data_and_LIF_update(snn_device, info_device, info_host, step, is_SL, true);
    }
    // print_array_numbers_from_device("Voltage", snn_host->voltages, voltage_dim);
    // print_array_numbers_from_device("Weight", snn_host->weights, std::vector<int>{info_host->nb_networks, info_host->nb_neurons, info_host->nb_neurons});
    // print_array_numbers_from_device("Spike augmented", snn_host->spikes_rec_augmented, spike_augmented_dim);
    // print_array_numbers_from_device("Weight", snn_host->weights, synapses_dim);
    // exit(0);
}

void run_SNN_SL(void *snn_host_device_ptr, float *input_data_host) {
    SNN_HOST_DEVICE *snn_host_device = (SNN_HOST_DEVICE*)snn_host_device_ptr;
    SNN *snn_device = snn_host_device->snn_device;
    SNN *snn_host = snn_host_device->snn_host;
    info_runner *info_host = snn_host_device->info_host;
    info_runner *info_device = snn_host_device->info_device;
    bool is_SL = true;


    // std::vector<int> voltage_dim = {info_host->nb_episodes, info_host->nb_networks, info_host->nb_neurons, info_host->nb_steps};
    // std::vector<int> voltage_augmented_dim = {info_host->nb_episodes, info_host->nb_networks, info_host->nb_neurons};
    // std::vector<int> input_dim = {info_host->nb_episodes, 1, info_host->nb_input, info_host->nb_steps};
    // std::vector<int> spike_augmented_dim = {info_host->nb_episodes, info_host->nb_networks, info_host->nb_neurons, info_host->nb_steps};
    // std::vector<int> neuron_dim = {info_host->nb_networks, info_host->nb_neurons};
    // std::vector<int> synapses_dim = {info_host->nb_networks, info_host->nb_neurons, info_host->nb_neurons};


    // 0 - Transfer data from host to device & init and reset runner
    transfer_data_host_to_device(snn_device, snn_host, info_host, input_data_host, is_SL);
    init_and_reset_runner(snn_host, info_device, info_host);

    // 1 - Input Data and LIF update (fusionned) only at the first step (done later in augmented_and_energy_update_and_input_data_and_LIF_update)
    input_data_and_LIF_update(snn_device, info_device, info_host, 0, is_SL);
    for (int step = 0; step < info_host->nb_steps; step++) {
        // 2 - Update the neurons voltages from the synapses (spikes)
        synapse_update(snn_device, info_device, info_host, step);
        // synapse_update_2(snn_device, info_device, info_host, step, 20);

        // 3 - Augmented & Energy update & Input data & LIF update
        augmented_and_energy_update_and_input_data_and_LIF_update(snn_device, info_device, info_host, step, is_SL, true);
    }
    // print_array_numbers_from_device("Voltage", snn_host->voltages, voltage_dim);
    // print_array_numbers_from_device("Weight", snn_host->weights, std::vector<int>{info_host->nb_networks, info_host->nb_neurons, info_host->nb_neurons});
    // print_array_numbers_from_device("Spike augmented", snn_host->spikes_rec_augmented, spike_augmented_dim);
    // print_array_numbers_from_device("Weight", snn_host->weights, synapses_dim);
    // exit(0);
}

void get_recorded_spikes_augmented(void *snn_host_device_ptr, float *spikes_rec_augmented_host) {
    SNN_HOST_DEVICE *snn_host_device = (SNN_HOST_DEVICE*)snn_host_device_ptr;
    SNN *snn_host = snn_host_device->snn_host;
    info_runner *info_host = snn_host_device->info_host;
    int spike_rec_augmented_size = info_host->nb_episodes * info_host->nb_networks * info_host->nb_neurons * info_host->nb_steps;

    // copy_device_to_host_bfloat(spikes_rec_augmented_host, snn_host->spikes_rec_augmented, spike_rec_augmented_size);
    cudaMemcpy(spikes_rec_augmented_host, snn_host->spikes_rec_augmented, spike_rec_augmented_size * sizeof(float), cudaMemcpyDeviceToHost);
}

void get_recorded_spikes(void *snn_host_device_ptr, int *spikes_rec_host) {
    SNN_HOST_DEVICE *snn_host_device = (SNN_HOST_DEVICE*)snn_host_device_ptr;
    SNN *snn_host = snn_host_device->snn_host;
    info_runner *info_host = snn_host_device->info_host;
    int spike_rec_size = info_host->nb_episodes * info_host->nb_networks * info_host->nb_neurons * info_host->nb_steps;

    // copy_device_to_host_bfloat(spikes_rec_augmented_host, snn_host->spikes_rec_augmented, spike_rec_augmented_size);
    cudaMemcpy(spikes_rec_host, snn_host->spikes_rec, spike_rec_size * sizeof(int), cudaMemcpyDeviceToHost);
}

void get_recorded_voltages(void *snn_host_device_ptr, float *voltages_host) {
    SNN_HOST_DEVICE *snn_host_device = (SNN_HOST_DEVICE*)snn_host_device_ptr;
    SNN *snn_host = snn_host_device->snn_host;
    info_runner *info_host = snn_host_device->info_host;
    int voltage_size = info_host->nb_episodes * info_host->nb_networks * info_host->nb_neurons * info_host->nb_steps;

    // copy_device_to_host_bfloat(voltages_host, snn_host->voltages, voltage_size);
    cudaMemcpy(voltages_host, snn_host->voltages, voltage_size * sizeof(float), cudaMemcpyDeviceToHost);
}


void* init_SNN(
                void *snn_host_device_ptr,

                int nb_episodes,
                int nb_networks,
                int nb_neurons,
                int nb_steps,

                // Neuron parameters
                float *threshold_host,
                float *tau_host,
                float *constant_current_host,

                // Synapse parameters
                float *weight_host,

                // Optional parameters
                int *refractory_host,
                int *delay_host,

                // Indexes
                int *input_indexes_host,
                int *output_indexes_host,
                int *hidden_indexes_host,

                int *synapses_actives_idx_host,

                // Size
                int nb_input,
                int nb_output,
                int nb_hidden,
                int synapses_actives_size,

                // Optional parameters
                bool is_LIF_beta,
                bool is_refractory,
                bool is_delay,
                bool is_record_spikes,
                bool is_online,
                bool is_SL
                ) {
    
    // Create info_runner
    int voltage_size = nb_episodes * nb_networks * nb_neurons * nb_steps;
    int weight_size = nb_networks * (nb_neurons * nb_neurons);
    int nb_synapses = synapses_actives_size / 2;
    int nb_neurons_one_network = nb_neurons;
    int nb_neurons_one_episode = nb_networks * nb_neurons;
    int nb_neurons_all_episodes = nb_episodes * nb_networks * nb_neurons;
    float dt = 1.0;

    int* syn_host_in = new int[nb_synapses];
    int* syn_host_out = new int[nb_synapses];

    // int* first_idx_neuron_in = new int[nb_neurons];
    // memset(first_idx_neuron_in, -1, nb_neurons * sizeof(int));
    // std::vector<int> neuron_in;

    // for (int i = 0, current_idx = -1; i < synapses_actives_size; i++) {
    for (int i = 0; i < synapses_actives_size; i++) {
        if (i < nb_synapses) syn_host_in[i] = synapses_actives_idx_host[i];
        else syn_host_out[i - nb_synapses] = synapses_actives_idx_host[i];

        // if (current_idx != syn_host_in[i] && i <= nb_synapses) {
        //     first_idx_neuron_in[syn_host_in[i]] = i;
        //     current_idx = syn_host_in[i];
        //     neuron_in.push_back(current_idx);
        // }
    }

    // print_array_numbers_from_host("Synapses actives idx in", syn_host_in, std::vector<int>{nb_synapses});
    // print_array_numbers_from_host("Synapses actives idx out", syn_host_out, std::vector<int>{nb_synapses});
    // print_array_numbers_from_host("First idx neuron in", first_idx_neuron_in, std::vector<int>{nb_neurons});
    // print_array_numbers_from_host("Nb synapses per neuron", nb_synapses_per_neuron, std::vector<int>{nb_networks, nb_neurons});
    // print_array_numbers_from_host("Neuron in", neuron_in.data(), std::vector<int>{(int)neuron_in.size()});
    // exit(0);

    SNN_HOST_DEVICE *snn_host_device;
    SNN *snn_host;
    SNN *snn_device;
    info_runner *info_host;
    info_runner *info_device;

    // Allocate memory if the pointer is NULL or re-init with new parameters
    if (snn_host_device_ptr == 0 || snn_host_device_ptr == NULL || snn_host_device_ptr == nullptr) { // If the pointer is NULL, we allocate memory
        snn_host_device = (SNN_HOST_DEVICE*)malloc(sizeof(SNN_HOST_DEVICE));
        if (snn_host_device == NULL) { fprintf(stderr, "Failed to allocate memory for snn_host_device from host\n"); exit(0);}

        info_host = (info_runner*)malloc(sizeof(info_runner));
        if (info_host == NULL) { fprintf(stderr, "Failed to allocate memory for info_host from host\n"); exit(0);}
        cudaMalloc_with_error_check("info_runner", &info_device, sizeof(info_runner));
        info_host->is_re_alloc = true;
        init_info_runner(info_device, info_host, nb_episodes, nb_networks, nb_neurons, nb_synapses, nb_steps, nb_input, nb_output, nb_hidden, is_online, is_SL);
        
        snn_host = (SNN*)malloc(sizeof(SNN));
        if (snn_host == NULL) { fprintf(stderr, "Failed to allocate memory for snn_host from host\n"); exit(0);}
        cudaMalloc_with_error_check("SNN", &snn_device, sizeof(SNN));
        snn_device = SNN_malloc(info_host, snn_device, snn_host, is_refractory, is_delay, is_SL);
        // std::cout << "1 - SNN FULL NEW MALLOCS" << std::endl;

    } else {
        snn_host_device = (SNN_HOST_DEVICE*)snn_host_device_ptr;
        snn_host = snn_host_device->snn_host;
        snn_device = snn_host_device->snn_device;
        info_host = snn_host_device->info_host;
        info_device = snn_host_device->info_device;
        info_host->is_re_alloc = false;

        // Re-allocate memory if the size of the parameters has changed
        if (info_host->nb_episodes < nb_episodes || info_host->nb_networks < nb_networks || info_host->nb_neurons < nb_neurons || info_host->nb_steps < nb_steps || info_host->nb_synapses < (synapses_actives_size / 2) || info_host->nb_input < nb_input || info_host->nb_output < nb_output || info_host->nb_hidden < nb_hidden) {
            SNN_free(snn_host_device);
            snn_host_device = (SNN_HOST_DEVICE*)malloc(sizeof(SNN_HOST_DEVICE));
            if (snn_host_device == NULL) { fprintf(stderr, "Failed to allocate memory for snn_host_device from host\n"); return NULL;}

            info_host = (info_runner*)malloc(sizeof(info_runner));
            if (info_host == NULL) { fprintf(stderr, "Failed to allocate memory for info_host from host\n"); return NULL;}
            cudaMalloc_with_error_check("info_runner", &info_device, sizeof(info_runner));
            info_host->is_re_alloc = true;
            init_info_runner(info_device, info_host, nb_episodes, nb_networks, nb_neurons, nb_synapses, nb_steps, nb_input, nb_output, nb_hidden, is_online, is_SL);

            snn_host = (SNN*)malloc(sizeof(SNN));
            if (snn_host == NULL) { fprintf(stderr, "Failed to allocate memory for snn_host from host\n"); return NULL;}
            cudaMalloc_with_error_check("SNN", &snn_device, sizeof(SNN));
            snn_device = SNN_malloc(info_host, snn_device, snn_host, is_refractory, is_delay, is_SL);
            // std::cout << "2 - SNN RE-ALLOCS" << std::endl;
        } else {
            init_info_runner(info_device, info_host, nb_episodes, nb_networks, nb_neurons, nb_synapses, nb_steps, nb_input, nb_output, nb_hidden, is_online, is_SL);
            // std::cout << "3 - SNN NO MALLOCS" << std::endl;
        }
    }

    init_SNN_host_device(
                        snn_host, 
                        snn_device, 
                        
                        threshold_host, 
                        tau_host, 
                        weight_host,

                        refractory_host,
                        delay_host,
                        
                        syn_host_in, 
                        syn_host_out, 
                        
                        input_indexes_host, 
                        output_indexes_host, 
                        hidden_indexes_host, 
                        
                        nb_input, 
                        nb_output, 
                        nb_hidden, 
                        
                        voltage_size, 
                        weight_size, 
                        nb_synapses, 
                        nb_neurons_one_network,
                        nb_neurons_one_episode, 
                        nb_neurons_all_episodes,

                        is_LIF_beta,
                        is_refractory,
                        is_delay,
                        is_record_spikes,
                        dt
                        );

    snn_host_device->snn_host = snn_host;
    snn_host_device->snn_device = snn_device;
    snn_host_device->info_host = info_host;
    snn_host_device->info_device = info_device;

    // Free host (a bit forcing but at least we are sure that the memory is freed)
    delete[] syn_host_in;
    delete[] syn_host_out;

    // Return SNN_HOST_DEVICE as void*
    return (void*)snn_host_device;
}

// SNN FUNCTIONS END


// Print functions
void snn_host_print(int nb_episodes, 
                int nb_networks, 
                int nb_neurons, 
                int nb_synapses, 
                int nb_steps, 
                int voltage_size, 
                int neuron_size, 
                int weight_size, 
                int synapses_actives_size, 
                int nb_input, 
                int nb_output, 
                int nb_hidden,
                float *threshold_host,
                float *tau_host,
                float *constant_current_host,
                float *weight_host,
                int *input_indexes_host,
                int *output_indexes_host,
                int *hidden_indexes_host,
                int *synapses_actives_idx,
                int *syn_host_in,
                int *syn_host_out
                ) {


    std::cout << "Init SNN GPUUUUU" << std::endl;
    std::cout << "\nNumber of episodes: " << nb_episodes << std::endl;
    std::cout << "Number of networks: " << nb_networks << std::endl;
    std::cout << "Number of neurons: " << nb_neurons << std::endl;
    std::cout << "Number of synapses: " << nb_synapses << std::endl;
    std::cout << "Number of steps: " << nb_steps << std::endl;

    std::cout << "Voltage size: " << voltage_size << std::endl;
    std::cout << "Neuron size: " << neuron_size << std::endl;
    std::cout << "Weight size: " << weight_size << std::endl;
    std::cout << "Synapses actives size: (" << synapses_actives_size/2 << ", " << synapses_actives_size/2 << ")" << std::endl;
    std::cout << "Input size: " << nb_input << std::endl;
    std::cout << "Output size: " << nb_output << std::endl;
    std::cout << "Hidden size: " << nb_hidden << std::endl;

    std::vector<int> voltage_dim = {nb_episodes, nb_networks, nb_neurons, nb_steps};
    std::vector<int> neuron_dim = {nb_networks, nb_neurons};
    std::vector<int> synapses_dim = {nb_networks, nb_neurons, nb_neurons};

    std::cout << "\n" << std::endl;
    print_array_numbers_from_host("Threshold", threshold_host, neuron_dim);
    print_array_numbers_from_host("Tau", tau_host, neuron_dim);
    print_array_numbers_from_host("Constant current", constant_current_host, neuron_dim);
    print_array_numbers_from_host("Weight", weight_host, synapses_dim);

    std::cout << std::endl; // add a new line for better readability
    print_array_numbers_from_host("Input indexes", input_indexes_host, std::vector{nb_input});
    print_array_numbers_from_host("Output indexes", output_indexes_host, std::vector{nb_output});
    print_array_numbers_from_host("Hidden indexes", hidden_indexes_host, std::vector{nb_hidden});
    // print_array_numbers_from_host("\nSynapses actives idx", synapses_actives_idx, synapses_actives_size);

    print_array_numbers_from_host("Synapses actives idx in ", syn_host_in, std::vector{nb_synapses});
    print_array_numbers_from_host("Synapses actives idx out", syn_host_out, std::vector{nb_synapses});


}

void snn_device_print(SNN *snn_host, info_runner *info_host) {

    std::vector<int> voltage_dim = {info_host->nb_episodes, info_host->nb_networks, info_host->nb_neurons, info_host->nb_steps};
    std::vector<int> neuron_dim = {info_host->nb_networks, info_host->nb_neurons};
    std::vector<int> synapses_dim = {info_host->nb_networks, info_host->nb_neurons, info_host->nb_neurons};

    print_array_numbers_from_device("Voltages", snn_host->voltages, voltage_dim);
    print_array_numbers_from_device("Thresholds", snn_host->thresholds, neuron_dim);
    print_array_numbers_from_device("Tau", snn_host->tau, neuron_dim);
    print_array_numbers_from_device("Weights", snn_host->weights, synapses_dim);


    print_array_numbers_from_device("Synapses_idx_in ", snn_host->synapses_actives_idx_in, std::vector{info_host->nb_synapses});
    print_array_numbers_from_device("Synapses_idx_out", snn_host->synapses_actives_idx_out, std::vector{info_host->nb_synapses});


    print_array_numbers_from_device("Input_idx ", snn_host->input_idx, std::vector{info_host->nb_input});
    print_array_numbers_from_device("Output_idx", snn_host->output_idx, std::vector{info_host->nb_output});
    print_array_numbers_from_device("Hidden_idx", snn_host->hidden_idx, std::vector{info_host->nb_hidden});

    print_array_numbers_from_device("Spike_rec_inference", snn_host->spikes_rec_inference, std::vector{info_host->nb_episodes, info_host->nb_networks, info_host->nb_neurons});
    // cudaDeviceSynchronize();
    // print_array_numbers_from_device("dt", &snn_host->dt, std::vector{1});
}


void print_info_host(info_runner *info_host) {
    printf("Contenu de info_host sur le CPU:\n");
    printf("nb_episodes: %d\n", info_host->nb_episodes);
    printf("nb_networks: %d\n", info_host->nb_networks);
    printf("nb_neurons: %d\n", info_host->nb_neurons);
    printf("nb_synapses: %d\n", info_host->nb_synapses);
    printf("nb_steps: %d\n", info_host->nb_steps);
    printf("nb_neurons_all_episodes: %d\n", info_host->nb_neurons_all_episodes);
    printf("nb_input: %d\n", info_host->nb_input);
    printf("nb_output: %d\n", info_host->nb_output);
    printf("nb_hidden: %d\n", info_host->nb_hidden);
    printf("is_augmented: %s\n", info_host->is_augmented ? "true" : "false");
    printf("is_energy: %s\n", info_host->is_energy ? "true" : "false");
}

void print_info_device(info_runner *info_device) {
    print_info_device_kernel<<<1,1>>>(info_device);
    cudaDeviceSynchronize();
}

__global__ void print_info_device_kernel(const info_runner* info_device) {
    // Utilisation d'un seul thread pour éviter des impressions redondantes
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Contenu de info_device sur le GPU:\n");
        printf("nb_episodes: %llu\n", (unsigned long long)info_device->nb_episodes);
        printf("nb_networks: %llu\n", (unsigned long long)info_device->nb_networks);
        printf("nb_neurons: %llu\n", (unsigned long long)info_device->nb_neurons);
        printf("nb_synapses: %llu\n", (unsigned long long)info_device->nb_synapses);
        printf("nb_steps: %llu\n", (unsigned long long)info_device->nb_steps);
        printf("nb_neurons_all_episodes: %llu\n", (unsigned long long)info_device->nb_neurons_all_episodes);
        printf("nb_input: %llu\n", (unsigned long long)info_device->nb_input);
        printf("nb_output: %llu\n", (unsigned long long)info_device->nb_output);
        printf("nb_hidden: %llu\n", (unsigned long long)info_device->nb_hidden);
        printf("is_augmented: %s\n", info_device->is_augmented ? "true" : "false");
        printf("is_energy: %s\n", info_device->is_energy ? "true" : "false");    }
}
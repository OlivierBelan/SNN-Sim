#!python
# cython: embedsignature=True, binding=True

cimport cython
# from snn_cython cimport SNN_cython as SNN
from .snn_cython cimport SNN_cython as SNN
from .snn_cython cimport SNN_cython_population # important that the path is the same in the .pxd file

import numpy as np
cimport numpy as np
np.import_array()
from .tools_cython cimport get_time
cimport libc.math as math
if HAS_CUDA:
    from ..SNN_cuda.snn_cuda_wrapper cimport init_SNN_GPU, run_SNN_GPU_SL, free_SNN_GPU, init_augmented_GPU, init_energy_GPU, get_recorded_spikes_augmented_GPU, get_recorded_spikes_GPU, get_recorded_voltages_GPU

from .encoder cimport Encoder
from .augmented cimport Augmented
from .energy cimport Energy

# from time import perf_counter
from libc.stdio cimport printf


from libc.stdlib cimport rand, srand
from libc.stdlib cimport RAND_MAX

cdef class Runner_SL_cython:

    cpdef void init(self, int run_time = 100, int run_time_margin = 0, float dt = 1.0, float input_spike_amplitude=100.0, bint is_augmented=False, bint is_energy=False, str neuron_reset="voltage_reset", list record_layer=["output"], bint disable_output_threshold = True, str decay_method = "lif", bint is_delay=False, bint is_refractory=False, set record_decoding_method = {"spike"}):
        np.set_printoptions(suppress=True)

        # Constant
        self.dt = dt # time step
        self.run_time_margin = run_time + run_time_margin # simulation time + margin
        self.run_time_original = run_time
        self.input_spike_amplitude = input_spike_amplitude # spike amplitude
        self.is_threshold_reset = True if neuron_reset == "threshold_reset" else False
        self.disable_output_threshold = disable_output_threshold
        self.is_LIF_beta = False if decay_method == "lif" else True
        self.record_layer = record_layer
        self.is_delay = is_delay
        self.is_refractory = is_refractory

        self.is_record_spike = "spike" in record_decoding_method
        self.is_record_voltage = "voltage" in record_decoding_method
        self.is_record_augmented = "augmented" in record_decoding_method


        self.encoder = Encoder() # create an encoder instance
        self.encoder.init_SL(self.run_time_original, self.run_time_margin)

        self.is_energy = is_energy
        self.is_augmented = is_augmented
        if self.is_energy == True: self.energy = Energy()
        if self.is_augmented == True: self.augmented = Augmented()

    cpdef void init_run(self, SNN_cython_population population, np.ndarray[np.int32_t, ndim=1] population_genome_ids, int batch_features=1, int batch_running = 1, int batch_population = 1, bint is_gpu = False, bint is_last_run = False):
        self.is_last_run = is_last_run

        self.batch_features = batch_features # batch_features size
        self.batch_running = min(batch_running, self.batch_features) # batch_running size
        self.batch_population = min(batch_population, population.nb_population) # batch_population size
        
        self.population = population
        self.population_ids = population_genome_ids
        self.population_len = population_genome_ids.size
        self.population_ids_split = self.split(population_genome_ids.tolist(), self.batch_population, to_numpy=True)
        self.population_ids_split_record = self.split(np.arange(self.population_len).tolist(), self.batch_population, to_numpy=True)            
        self.is_gpu = is_gpu
        if self.snn_gpu_ptr == NULL: self.is_re_alloc = True
        else:                        self.is_re_alloc = False

        self.encoder.init_SL(self.run_time_original, self.run_time_margin)

    #  PUBLIC METHODS
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void run(self, np.ndarray input_data, bint is_encoded=False):

        self.input_data = self.encoder.encode(input_data, is_encoded)
        
        if self.is_gpu: self.run_GPU()
        else:           self.run_CPU()
       
        # for debug (if I want to compare GPU and CPU)
        # self.run_GPU()
        # self.run_C()


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void run_GPU(self):
        cdef int data_set_len = len(self.input_data)
        cdef int population_split_len = len(self.population_ids_split)
        cdef int i, current_time
        self.count = 0
        self.batch_features = min(self.batch_features, data_set_len)
        self.batch_running = min(self.batch_running, self.batch_features)
        self.init_record()


        while self.count < data_set_len:
            if self.count + self.batch_running > data_set_len:
                self.batch_running = data_set_len - self.count

            for i in range(population_split_len):
                # 1 - Init Network GPU
                self.init_network_GPU(self.population_ids_split[i], self.is_re_alloc) # re/init network GPU

                # 1.1 - Init Energy Network GPU
                if self.is_energy: 
                    self.energy.init_network_run(self.population, self.population_ids_split[i], self.weight_view, self.run_time_original, self.run_time_margin, self.batch_running, self.nb_networks, self.nb_neurons, self.input_indexes.shape[0])
                    self.snn_gpu_ptr = init_energy_GPU(self.snn_gpu_ptr, self.energy, self.energy.energy_index, self.energy.energy_length, self.weight_view, self.energy.energy_update_method)

                # 1.2 - Init Augmented Decoder GPU
                if self.is_augmented: 
                    self.augmented.init_run(self.run_time_original, self.run_time_margin, self.is_LIF_beta, self.batch_running, self.nb_networks, self.nb_neurons, self.population_ids_split[i], self.input_indexes, self.output_indexes, self.neuron_to_record_indexes, self.threshold, self.tau, self.refractory, self.is_refractory)
                    self.snn_gpu_ptr = init_augmented_GPU(self.snn_gpu_ptr, self.augmented.spikes_importance, self.augmented.spike_max_time_step, self.augmented.is_neurons_update_with_augmented, self.augmented.is_voltage_reset, self.augmented.spike_format_type, self.augmented.importance_type)

                # 2 - Run SNN GPU
                run_SNN_GPU_SL(self.snn_gpu_ptr, self.input_data[self.count:self.count+self.batch_running, :, :, :])

                # 3 - Get Recorded Spikes Augmented GPU
                if self.is_record_augmented:
                    get_recorded_spikes_augmented_GPU(self.snn_gpu_ptr, self.augmented.augmented_decoder)
                elif self.is_record_spike:
                    get_recorded_spikes_GPU(self.snn_gpu_ptr, self.spike_state)
                elif self.is_record_voltage:
                    get_recorded_voltages_GPU(self.snn_gpu_ptr, self.voltage)

                self.record(self.population_ids_split_record[i])

            self.count += self.batch_running
   
        if self.is_last_run: free_SNN_GPU(self.snn_gpu_ptr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void run_CPU(self):
        cdef int data_set_len = len(self.input_data)
        cdef int population_split_len = len(self.population_ids_split)
        cdef int i, current_time

        self.batch_running = min(self.batch_running, self.batch_features)
        self.init_record()
        self.count = 0

        while self.count < data_set_len:
            if self.count + self.batch_running > data_set_len:
                self.batch_running = data_set_len - self.count

            for i in range(population_split_len):
                self.init_network_CPU(self.population_ids_split[i])

                if self.is_energy == True:
                    self.energy.init_network_run(self.population, self.population_ids_split[i], self.weight_view, self.run_time_original, self.run_time_margin, self.batch_running, self.nb_networks, self.nb_neurons, self.input_indexes.shape[0])


                if self.is_augmented == True:
                    self.augmented.init_run(self.run_time_original, self.run_time_margin, self.is_LIF_beta, self.batch_running, self.nb_networks, self.nb_neurons, self.population_ids_split[i], self.input_indexes, self.output_indexes, self.neuron_to_record_indexes, self.threshold, self.tau, self.refractory, self.is_refractory)

                for current_time in range(self.run_time_margin):
                    
                    # 1 - Input Data in Networks
                    self.input_data_to_network(current_time) # init network

                    # 2 - LIF Update and Check Networks Spikes and LIF and Apply Refractory (if needed) 
                    self.LIF_update(current_time)
                    
                    # 3 - Update Networks Voltge with Weights + Delay and Refractory update
                    self.update_voltage_with_weights_delay_refractory(current_time)
                    
                    # 4 - Augmented Decoder Update
                    if self.is_augmented == True:
                        self.augmented.update(current_time) # Decoder Update
                    
                    # 5 - Energy Update
                    if self.is_energy == True:
                        self.energy.update(current_time, self.spike_state, self.augmented, self.is_augmented)

                # 5 - Apply important in Augmented Decoder - A REVOIR penser à integrer pour chaque time step
                if self.is_augmented == True:
                    self.augmented.apply_importance()

                # 6 - Record Networks (Spikes)
                self.record(self.population_ids_split_record[i])

            self.count += self.batch_running

    cpdef np.ndarray get_encoded_data(self):
        return np.array(self.input_data)

    # PRIVATES METHODS
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void input_data_to_network(self, int current_time):
        self.voltage[:, :, self.input_indexes, current_time] += self.input_data[self.count:self.count+self.batch_running, :, :, current_time]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void LIF_update(self, int current_time):
        cdef bint index = 1 if current_time + 1 < self.run_time_margin else 0

        # CYTHON view version
        self.voltage_sub_view = self.voltage_view[:, :, :, current_time]
        self.voltage_sub_view_next = self.voltage_view[:, :, :, current_time + 1]
        self.spike_state_view_sub = self.spike_state_view[:, :, :, current_time]
        self.spike_state_sub = self.spike_state[:, :, :, current_time]
        cdef int i, j, k, l
        cdef float voltage, threshold, voltage_next

        for i in range(self.batch_running):
            for j in range(self.nb_networks):
                for l in range(self.nb_neurons_actives):
                    k = self.neurons_actives_indexes_view[l]
                    voltage = self.voltage_sub_view[i, j, k]
                    threshold = self.threshold_view[j, k]

                    # 0 - Update/Reduce refractory_active time with dt/time_step (if refractory used)
                    if self.is_refractory == True and self.refractory_active_view[i, j, k] > 0:
                        self.refractory_active_view[i, j, k] -= 1 # could be replaced by -= self.dt but refractory_active would need to be float
                        if self.refractory_active_view[i, j, k] < 0: self.refractory_active_view[i, j, k] = 0
                        continue # skip the rest of the loop

                    # 1 - Check/Save Spike and add Refractory if needed
                    if voltage > threshold:
                        self.spike_state_view_sub[i, j, k] = 1

                        if self.is_refractory == True:
                            self.refractory_active_view[i, j, k] = self.refractory_view[j, k]

                        if index == 1:
                            if self.is_threshold_reset == True:
                                self.voltage_sub_view_next[i, j, k] = voltage - threshold
                            else:
                                self.voltage_sub_view_next[i, j, k] = 0.0

                    # 2 - Update Voltage for next time step -> LIF (Leaky Integrate and Fire)
                    else:
                        if index == 1:
                            # 3 - LIF
                            voltage_next = self.voltage_sub_view_next[i, j, k] + voltage
                            if self.is_LIF_beta == True: # Voltage + (-Voltage + Constant Current * Resistance) / Tau * dt) here Resistance = 1 and dt = 1
                                # self.voltage_sub_view_next[i, j, k] =  voltage_next * self.tau_view[j, k] * self.dt + self.constant_current_view[j, k]
                                self.voltage_sub_view_next[i, j, k] =  voltage_next * self.tau_view[j, k] * self.dt
                            else: # V * leak * dt + current
                                # self.voltage_sub_view_next[i, j, k] =  voltage_next + (-voltage_next + self.constant_current_view[j, k]) / self.tau_view[j, k] * self.dt
                                self.voltage_sub_view_next[i, j, k] =  voltage_next + (-voltage_next / self.tau_view[j, k] * self.dt)

                            # self.voltage_sub_view_next[i, j, k] += voltage # add the previous voltage to the new voltage (important to keep this form for the delay weights)
                            # self.voltage_sub_view_next[i, j, k] = self.voltage_sub_view_next[i, j, k] + (-self.voltage_sub_view_next[i, j, k] + self.constant_current_view[j, k]) / self.tau_view[j, k] * self.dt


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void update_voltage_with_weights_delay_refractory(self, int current_time):
        if current_time + 1 >= self.run_time_margin: return

        # 1 - Set voltage_weight_view with next time step
        cdef int start = min(current_time+1, self.run_time_margin)
        cdef int end = min(start+1, self.run_time_margin)
        cdef float[:,:,:,:] voltage_view = self.voltage_view[:, :, :, start:end]


        cdef int[:, :, :] spike_state_sub_view = self.spike_state_sub
        cdef int i, j, neuron_in, neuron_out, a, connections
        cdef int delay

        # 2 - Get the active synapses indexes (in order to only iterate on them)
        connections = self.synapses_actives_indexes_view.shape[1]


        cdef float[:,:,:,:] augmented_decoder_view
        cdef float[:,:,:] augmented_decoder_view_prev
        if self.is_augmented == True:
            augmented_decoder_view_prev = self.augmented.augmented_decoder[:, :, :, current_time]
            augmented_decoder_view = self.augmented.augmented_decoder[:, :, :, start:end]
            
        for i in range(self.batch_running):
            for j in range(self.nb_networks):
                for a in range(connections):
                    neuron_in = self.synapses_actives_indexes_view[0, a]
                    neuron_out = self.synapses_actives_indexes_view[1, a]

                    # 2 - Check if refractory and skip if needed
                    if self.is_refractory == True and self.refractory_active_view[i, j, neuron_in] > 0 and self.refractory_active_view[i, j, neuron_out] > 0: continue

                    # 3 - If spike, Update the voltage with the weights
                    if spike_state_sub_view[i, j, neuron_in] == 1:
                        delay = self.delay_view[j, neuron_in, neuron_out] if self.is_delay == True else 0
                        if delay + current_time >= self.run_time_margin: continue

                        if self.is_energy == True:
                            voltage_view[i, j, neuron_out, delay] += self.weight_view[j, neuron_in, neuron_out] * self.energy.energy_view[i, j, self.energy.energy_index_view[i, j, neuron_in], neuron_in]
                        else:
                            voltage_view[i, j, neuron_out, delay] += self.weight_view[j, neuron_in, neuron_out]

                        # 4 - Update the voltage_decoder_view (if augmented_decoder used)
                        if self.is_augmented == True and neuron_out >= self.augmented.neuron_decoder_start and neuron_out <= self.augmented.neuron_decoder_end:
                            if self.augmented.is_neurons_update_with_augmented == True and augmented_decoder_view[i, j, neuron_in, 0] > 0:
                                if self.is_energy == True:
                                    augmented_decoder_view[i, j, neuron_out, delay] += self.weight_view[j, neuron_in, neuron_out] * augmented_decoder_view_prev[i, j, neuron_in] * self.energy.energy_view[i, j, self.energy.energy_index_view[i, j, neuron_in], neuron_in]
                                else:
                                    augmented_decoder_view[i, j, neuron_out, delay] += self.weight_view[j, neuron_in, neuron_out] * augmented_decoder_view_prev[i, j, neuron_in]

                            else:
                                if self.is_energy == True:
                                    augmented_decoder_view[i, j, neuron_out, delay] += self.weight_view[j, neuron_in, neuron_out] * self.energy.energy_view[i, j, self.energy.energy_index_view[i, j, neuron_in], neuron_in]
                                else:
                                    augmented_decoder_view[i, j, neuron_out, delay] += self.weight_view[j, neuron_in, neuron_out] 

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void init_network_CPU(self, np.ndarray[np.int32_t, ndim=1] pop_idx):
        cdef int i, j, k

        self.nb_networks = pop_idx.shape[0]
        self.nb_neurons  = self.population.nb_neurons # total number of neurons
        self.nb_neurons_actives = self.population.neuron_active_global_indexes.size # input + hidden_active_global + output

        self.input_indexes  = self.population.input_indexes
        self.hidden_indexes = self.population.hidden_indexes
        self.output_indexes = self.population.output_indexes

        self.weight    = self.population.weight[pop_idx]
        self.tau       = self.population.tau[pop_idx]
        self.threshold = self.population.threshold[pop_idx]
        self.constant_current_kernel = self.population.constant_current[pop_idx]
        self.voltage_reset = self.population.voltage_init[pop_idx]
        
        if self.is_delay == True:
            self.delay_view = np.ceil(self.population.delay[pop_idx]).astype(np.int32)
            self.delay_max = 0
            self.is_delay = False
            for i in range(self.nb_networks):
                for j in range(self.nb_neurons):
                    for k in range(self.nb_neurons):

                        if self.delay_view[i, j, k] > <int>self.run_time_margin: self.delay_view[i, j, k] = self.run_time_margin - 1 # set to run_time_margin - 1 in case of delay > run_time_margin
                        elif self.delay_view[i, j, k] < 0: self.delay_view[i, j, k] = 0 # set to 0 in case of negative delay

                        if self.delay_view[i, j, k] > <int>self.delay_max: self.delay_max = self.delay_view[i, j, k]
                        if self.delay_view[i, j, k] > 0: self.is_delay = True
            self.delay_max += 1
        
        if self.is_refractory == True:
            self.refractory        = np.ceil(self.population.refractory[pop_idx]).astype(np.int32)
            self.is_refractory     = np.any(self.refractory != 0)
            self.refractory_active = np.zeros((self.batch_running, self.nb_networks, self.nb_neurons), dtype=np.int32)
            self.refractory_active_view = self.refractory_active
            self.refractory_view = self.refractory
        
        if self.is_LIF_beta == True:
            self.tau = np.clip(self.tau, 0.0, 1.0) # in case tau < 0 or tau > 1
            self.tau = 1.0 - self.tau # because tau = 1/tau in the LIF equation
        else:
            self.tau[self.tau <= 0] = 1.0 # in case of tau <= 0

        if self.disable_output_threshold == True: self.threshold[:, self.output_indexes] = 1e14 # set very high voltage to disable output reset

        if self.is_energy == True: self.weight_view = self.weight

        self.voltage     = np.zeros((self.batch_running, self.nb_networks, self.nb_neurons, self.run_time_margin), dtype=np.float32)
        self.spike_state = np.zeros((self.batch_running, self.nb_networks, self.nb_neurons, self.run_time_margin), dtype=np.int32)

        # Transfer to cython memoryview
        self.voltage_view = self.voltage
        self.voltage_reset_view = self.voltage_reset
        self.tau_view = self.tau
        self.constant_current_view = self.constant_current_kernel
        self.threshold_view = self.threshold
        self.spike_state_view = self.spike_state
        self.weight_view = self.weight
        self.neurons_actives_indexes_view = self.population.neuron_active_global_indexes
        self.synapses_actives_indexes_view = self.population.synapse_active_global_indexes


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void init_network_GPU(self, np.ndarray[np.int32_t, ndim=1] pop_idx, bint is_re_alloc):
        cdef int i, j, k
        
        self.nb_networks = pop_idx.shape[0]
        self.nb_neurons = self.population.nb_neurons # input + hidden_active_global + output

        self.input_indexes  = self.population.input_indexes
        self.hidden_indexes = self.population.hidden_indexes
        self.output_indexes = self.population.output_indexes

        self.weight    = self.population.weight[pop_idx]
        self.tau       = self.population.tau[pop_idx]
        self.threshold = self.population.threshold[pop_idx]

        if self.is_delay == True:
            self.delay = np.ceil(self.population.delay[pop_idx]).astype(np.int32)
            self.delay_view = self.delay
            self.delay_max = 0
            self.is_delay = False
            for i in range(self.nb_networks):
                for j in range(self.nb_neurons):
                    for k in range(self.nb_neurons):

                        if self.delay_view[i, j, k] > <int>self.run_time_margin: self.delay_view[i, j, k] = self.run_time_margin - 1 # set to run_time_margin - 1 in case of delay > run_time_margin
                        elif self.delay_view[i, j, k] < 0: self.delay_view[i, j, k] = 0 # set to 0 in case of negative delay

                        if self.delay_view[i, j, k] > <int>self.delay_max: self.delay_max = self.delay_view[i, j, k]
                        if self.delay_view[i, j, k] > 0: self.is_delay = True            
            self.delay_max += 1


        if self.is_refractory == True:
            self.refractory        = np.ceil(self.population.refractory[pop_idx]).astype(np.int32)
            self.is_refractory     = np.any(self.refractory != 0)
        
        if self.is_LIF_beta == True:
            self.tau = np.clip(self.tau, 0.0, 1.0) # in case tau < 0 or tau > 1
            self.tau = 1.0 - self.tau # because tau = 1/tau in the LIF equation
        else:
            self.tau[self.tau <= 0] = 1.0 # in case of tau <= 0

        if self.disable_output_threshold == True: self.threshold[:, self.output_indexes] = 1e14 # set very high threshold to disable output reset

        if self.is_energy == True: self.weight_view = self.weight

        if HAS_CUDA: 
            # re_init SNN without doing a malloc
                self.snn_gpu_ptr = init_SNN_GPU(
                    self.snn_gpu_ptr if is_re_alloc == False else NULL, # Passing NULL to do a complete init with a malloc
                    # Info run
                    self.batch_running,
                    self.nb_networks,
                    self.nb_neurons,
                    self.run_time_margin,

                    # Neuron params
                    self.threshold,
                    self.tau,
                    self.constant_current_kernel,

                    # Synapses params
                    self.weight,

                    # Optional
                    self.refractory,
                    self.delay,

                    # Indexes
                    self.input_indexes,
                    self.output_indexes,
                    self.hidden_indexes,
                    self.population.synapse_active_global_indexes,

                    # Optional
                    self.is_LIF_beta, # is_LIF_BETA
                    self.is_refractory,
                    self.is_delay,
                    self.is_record_spike,
                    is_online=False,
                    is_SL=True
                )
        if self.is_record_spike == True:  self.spike_state = np.zeros((self.batch_running, self.nb_networks, self.nb_neurons, self.run_time_margin), dtype=np.int32)
        if self.is_record_voltage == True: self.voltage    = np.zeros((self.batch_running, self.nb_networks, self.nb_neurons, self.run_time_margin), dtype=np.float32)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void init_record(self):
        self.neuron_to_record_indexes = np.array([], dtype=np.int32)
        if "input" in self.record_layer: self.neuron_to_record_indexes = np.concatenate((self.neuron_to_record_indexes, self.population.input_indexes))
        if "output" in self.record_layer: self.neuron_to_record_indexes = np.concatenate((self.neuron_to_record_indexes, self.population.output_indexes))
        if "hidden" in self.record_layer: self.neuron_to_record_indexes = np.concatenate((self.neuron_to_record_indexes, self.population.hidden_indexes))
        cdef int nb_neurons_to_record = self.neuron_to_record_indexes.shape[0]

        if self.is_record_spike == True: self.record_spikes_array = np.zeros((self.population_len, self.batch_features, nb_neurons_to_record), np.int32)
        if self.is_record_voltage == True: self.record_voltages_array = np.zeros((self.population_len, self.batch_features, nb_neurons_to_record), dtype=np.float32)
        if self.is_record_augmented == True: self.record_augmented_array = np.zeros((self.population_len, self.batch_features, nb_neurons_to_record), dtype=np.float32)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void record(self, np.ndarray[np.int32_t, ndim=1] pop_idx):
        cdef int i, j
        cdef np.ndarray[np.int64_t, ndim=3] spike_sumed
        cdef np.ndarray[np.float32_t, ndim=3] voltage_record
        cdef np.ndarray[np.float32_t, ndim=3] augmented_sumed_decoder

        if self.is_record_spike:     
            spike_sumed = np.sum(self.spike_state[:, :, self.neuron_to_record_indexes], axis=3)
            self.record_spikes_array[pop_idx, self.count:self.count+self.batch_running] = spike_sumed.transpose(1,0,2)

        if self.is_record_voltage:   
            voltage_record = self.voltage[:, :, self.neuron_to_record_indexes, -1]
            self.record_voltages_array[pop_idx, self.count:self.count+self.batch_running] = voltage_record.transpose(1,0,2)

        if self.is_record_augmented:
            self.record_augmented_array[pop_idx, self.count:self.count+self.batch_running] = self.augmented.record_numpy().transpose(1,0,2)



    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef dict get_record_augmented_spikes(self):
        return self.record_augmented

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef dict get_record_spikes(self):
        return self.record_spikes

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef dict get_record_voltages(self):
        return self.record_voltages

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray[np.int32_t, ndim=3] get_record_spikes_raw(self):
        return self.record_spikes_array

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray[np.int32_t, ndim=3] get_record_voltages_raw(self):
        return self.record_voltages_array

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray[np.int32_t, ndim=3] get_record_augmented_raw(self):
        return self.record_augmented_array

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list split(self, list lst, int s, bint to_numpy=False):
        cdef list chunks = []
        cdef int i
        
        for i in range(0, len(lst), s):
            chunk = lst[i:i + (<int>s)]
            if to_numpy == True:
                chunks.append(np.array(chunk, dtype=np.int32))
            else:
                chunks.append(chunk)
    
        return chunks

# !python
# cython: embedsignature=True, binding=True

cimport cython
from SNN_cython_cuda.SNN_cython.snn_cython cimport SNN_cython as SNN
from SNN_cython_cuda.SNN_cython.snn_cython cimport SNN_cython_population
import numpy as np
cimport numpy as np
np.import_array()
from SNN_cython_cuda.SNN_cython.tools_cython cimport get_time, norm_min_max_all, norm_min_max_rows, norm_min_max_columns, norm_L1_all, norm_L1_rows, norm_L1_columns, norm_L1_sum_all, norm_L1_sum_rows, norm_L1_sum_columns, norm_L2_all, norm_L2_rows, norm_L2_columns, norm_L2_sum_all, norm_L2_sum_rows, norm_L2_sum_columns
from .encoder cimport Encoder
from .augmented cimport Augmented
from .energy cimport Energy
from .r_stdp cimport R_STDP

cimport libc.math as math

from libc.stdlib cimport rand, RAND_MAX
from libc.stdio  cimport printf

if HAS_CUDA:
    from SNN_cython_cuda.SNN_cuda.snn_cuda_wrapper cimport init_SNN_GPU, run_SNN_GPU_RL, free_SNN_GPU, init_augmented_GPU, init_energy_GPU, get_recorded_spikes_augmented_GPU, get_recorded_spikes_GPU, get_recorded_voltages_GPU

cdef class Runner_RL_cython:

    cpdef void init(self, int run_time = 100, int run_time_margin = 0, float dt = 1.0, int nb_episode = 1, bint online=False, bint is_augmented=False, bint is_energy=False, bint is_r_stdp=False, str neuron_reset="voltage_reset", list record_layer=["output"], bint disable_output_threshold = False, str decay_method="lif", bint is_delay=False, bint is_refractory=False, int delay_max = 100, set record_decoding_method = {"spike"}, set spike_count_negative = {"none"}, bint is_voltage_negative = True):

        # CONSTANT
        self.dt = dt # time step
        self.run_time_original = run_time # simulation time
        self.run_time_margin = run_time + run_time_margin
        delay_max = <int>max(0, <long>delay_max) # in case its negative (even its type of int)
        self.delay_max = delay_max + 1 # max delay
        self.run_time_delay_max = self.run_time_margin + delay_max  # simulation time + margin + max delay
        self.online = online
        self.is_threshold_reset = True if neuron_reset == "threshold_reset" else False

        # NEGATIVE_SPIKES
        self.is_spike_negative = False if "none" in spike_count_negative else True
        self.spike_count_negative_layer = spike_count_negative
        self.is_input_spike_negative = True if  "input"  in spike_count_negative else False
        self.is_output_spike_negative = True if "output" in spike_count_negative else False
        self.is_hidden_spike_negative = True if "hidden" in spike_count_negative else False


        self.disable_output_threshold = disable_output_threshold
        self.is_LIF_beta = False if decay_method == "lif" else True
        self.is_delay = is_delay
        self.is_refractory = is_refractory
        self.is_voltage_negative = is_voltage_negative

        self.input_spike_amplitude = 100.0 # spike amplitude
        self.nb_episode = nb_episode # nb_episode size

        self.record_layer = record_layer
        self.is_record_spike = "spike" in record_decoding_method
        self.is_record_voltage = "voltage" in record_decoding_method
        self.is_record_augmented = "augmented" in record_decoding_method

        self.encoder = Encoder()

        self.is_energy = is_energy
        self.is_augmented = is_augmented
        self.is_r_stdp = is_r_stdp
        if self.is_energy == True: self.energy = Energy()
        if self.is_augmented == True: self.augmented = Augmented()
        if self.is_r_stdp == True: self.r_stdp = R_STDP()

    cpdef void init_run(self, SNN_cython_population population, np.ndarray[np.int32_t, ndim=1] population_genome_ids, bint is_gpu=False, bint is_last_run=False):
        self.is_gpu = is_gpu
        self.is_last_run = is_last_run
        self.first_time_step = 0
        self.population_len = population_genome_ids.size
        self.population_ids = population_genome_ids
        self.population_ids_list = population_genome_ids.tolist()
        self.population = population
        self.record_spikes = {}
        self.record_voltages = {}
        self.record_augmented = {}
        self.nb_neurons = population.nb_neurons

        if self.is_gpu == True:
            if self.snn_gpu_ptr == NULL: self.is_re_alloc = True
            else:                        self.is_re_alloc = False
            self.init_network_GPU(population_genome_ids, self.is_re_alloc)
        else:
            self.init_network_CPU(population_genome_ids)

        if self.is_augmented == True: self.augmented.init_run(self.run_time_original, self.run_time_margin, self.is_LIF_beta, self.nb_episode, self.nb_networks, self.nb_neurons, population_genome_ids, self.input_indexes, self.output_indexes, self.neuron_to_record_indexes, self.threshold, self.tau, self.refractory, self.is_refractory)
        if self.is_energy == True:    self.energy.init_network_run(self.population, self.population_ids, self.weight_view, self.run_time_original, self.run_time_margin, self.nb_episode, self.nb_networks, self.nb_neurons, self.input_indexes.shape[0])
        if self.is_r_stdp == True:    self.r_stdp.init_run(self.nb_episode, self.nb_networks, self.nb_neurons, self.synapses_actives_indexes_view)

        # GPU
        if self.is_gpu:
            if self.is_energy:    self.snn_gpu_ptr = init_energy_GPU(self.snn_gpu_ptr, self.energy.energy, self.energy.energy_index, self.energy.energy_length, self.weight_view, self.energy.energy_update_method)
            if self.is_augmented: self.snn_gpu_ptr = init_augmented_GPU(self.snn_gpu_ptr, self.augmented.spikes_importance, self.augmented.spike_max_time_step, self.augmented.is_neurons_update_with_augmented, self.augmented.is_voltage_reset, self.augmented.spike_format_type, self.augmented.importance_type)

        self.encoder.init_RL(self.run_time_original, self.run_time_margin)

    #  PUBLIC METHODS
    @cython.boundscheck(False)
    @cython.wraparound(False)
    # cpdef void run(self, str encoder_type, np.ndarray inputs_data, int spike_rate = 2, float spike_amplitude = 100.0, int max_nb_spikes = 3, int reduce_noise = 100, int combinatorial_factor = 1, int combinaison_size=1, int combinaison_size_max=1, float combinatorial_combinaison_noise=0.0, bint combinatorial_roll = True, str combinatorial_filter = "energy", float[:] combinatorial_modulo = None, bint combinatorial_print_table_debug = False, float direct_min = -100_000, float direct_max = 100_000, float derivative_threshold = 0.02, bint derivative_is_latency = True, bint derivative_is_latency_positional=True, bint derivative_use_prev_input=True, float derivative_max_delta_latency = 1.0):
    cpdef void run(self, np.ndarray inputs_data, bint is_encoded=False):

        self.input_data = self.encoder.encode(inputs_data, is_encoded)

        if self.is_gpu: self.run_GPU()
        else:           self.run_CPU()
        
        # Debug (GPU vs CPU)
        # self.run_GPU()
        # self.run_CPU()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void run_GPU(self):
        # 0 - Init networks for this step
        if HAS_CUDA :
            run_SNN_GPU_RL(self.snn_gpu_ptr, self.input_data)
            if self.is_record_augmented:
                get_recorded_spikes_augmented_GPU(self.snn_gpu_ptr, self.augmented.augmented_decoder)
            elif self.is_record_spike:
                get_recorded_spikes_GPU(self.snn_gpu_ptr, self.spike_state)
            elif self.is_record_voltage:
                get_recorded_voltages_GPU(self.snn_gpu_ptr, self.voltage)

            # Debug (GPU vs CPU)
            # print("FROM cython GPU augmented_decoder:", self.augmentedaugmented_decoder, np.shape(self.augmented.augmented_decoder))
            # self.augmented_decoder_2 = np.copy(self.augmented.augmented_decoder)
            # self.augmented_decoder = np.zeros(np.shape(self.augmented.augmented_decoder), dtype=np.float32)
            if self.is_last_run: free_SNN_GPU(self.snn_gpu_ptr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void free_GPU(self):
        if HAS_CUDA : free_SNN_GPU(self.snn_gpu_ptr)
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void run_CPU(self):
        
        # 0 - Init networks for this step
        self.init_network_step()

        for self.time_step in range(self.run_time_margin):
            
            # 1 - Input Data in Networks
            self.input_data_to_network(self.time_step) # init network

            # 2 - LIF Update and Check Networks Spikes and LIF and Apply Refractory (if needed) 
            self.LIF_update(self.time_step) # Check Spike + Update (LIF) Voltage + Refractory
            

            # 3 - Update Networks Voltge with Weights + Delay and Refractory update
            self.update_voltage_with_weights_delay_refractory(self.time_step)

            # 4 - Augmented Decoder Update
            if self.is_augmented == True:
                self.augmented.update(self.time_step)

            # 5 - Energy Update
            if self.is_energy == True:
                self.energy.update(self.time_step, self.spike_state, self.augmented, self.is_augmented) # Energy Update

            # 6 - STD time step update
            if self.is_r_stdp == True:  
                self.r_stdp.stdp_time_step += 1
                # self.weights_concatenated = np.concatenate((self.weights_concatenated, self.weight[0, self.synapses_actives_indexes_view[0], self.synapses_actives_indexes_view[1]][None,...]), axis=0)

        # 7 - Apply importance in Augmented Decoder
        if self.is_augmented == True:
            self.augmented.apply_importance()

        # 8 - Record Networks (Spikes)
        # self.record()

        # Debug (GPU vs CPU)
        # print("voltages\n", self.voltage, "shape", np.shape(self.voltage))
        # print("augmented_augmented_decoder\n", self.augmented.augmented_decoder, "shape", np.shape(self.augmented.augmented_decoder))
        # print("weights\n", self.weight, "shape", np.shape(self.weight))
        # print("self.augmented.augmented_decoder - self.augmented.augmented_decoder_2", self.augmented.augmented_decoder - self.augmented.augmented_decoder_2)
        # exit()

    # PRIVATES METHODS
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void input_data_to_network(self, int current_time):
        self.voltage[:, :, self.input_indexes, current_time] += self.input_data[:, :, :, current_time]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void LIF_update(self, int current_time):
        cdef bint index = 1 if current_time + 1 < self.run_time_margin else 0

        # CYTHON view version
        self.voltage_sub_view = self.voltage_view[:, :, :, current_time]
        self.voltage_sub_view_next = self.voltage_view[:, :, :, current_time + 1] if index == 1 else self.voltage_view[:, :, :, current_time]
        self.spike_state_view_sub = self.spike_state_view[:, :, :, current_time]
        self.spike_state_sub = self.spike_state[:, :, :, current_time]
        cdef int i, j, k, l
        cdef float voltage, threshold, voltage_next
        cdef int neuron_idx

        for i in range(self.nb_episode):
            for j in range(self.nb_networks):
                for l in range(self.nb_neurons_actives):
                    k = self.neurons_actives_indexes_view[l]
                    neuron_idx = self.neurons_actives_indexes_view[l]
                    voltage = self.voltage_sub_view[i, j, k]
                    threshold = self.threshold_view[j, k]

                    # 0 - Update/Reduce refractory_active time with dt/time_step (if refractory used)
                    if self.is_refractory == True and self.refractory_active_view[i, j, k] > 0:
                        self.refractory_active_view[i, j, k] -= 1 # could be replaced by -= self.dt but refractory_active would need to be float
                        if self.refractory_active_view[i, j, k] < 0: self.refractory_active_view[i, j, k] = 0

                    # 0.1 - Reset voltage to 0.0 if negative voltage not allowed
                    if voltage < 0.0 and self.is_voltage_negative == False: voltage = 0.0 # reset negative voltage to 0.0

                    # 0.2 - Add possible noise to voltage and some noisy background spikes (if R-STDP used)
                    if self.is_r_stdp == True:
                        voltage = self.r_stdp.add_noise_spikes_and_voltages(i, j, k, voltage, threshold) 

                    # 1 - Check/Save Spike and add Refractory if needed
                    if (voltage > threshold or # positive spike
                        (self.is_spike_negative == True and voltage < -threshold and # negative spike
                        ((self.is_input_spike_negative  and neuron_idx >= self.input_indexes_start  and neuron_idx <= self.input_indexes_end) or # negative input
                        (self.is_output_spike_negative  and neuron_idx >= self.output_indexes_start and neuron_idx <= self.output_indexes_end) or # negative output
                        (self.is_hidden_spike_negative  and neuron_idx >= self.hidden_indexes_start and neuron_idx <= self.hidden_indexes_end)))): # negative hidden
   
                        if voltage > threshold: # positive spike
                            self.spike_state_view_sub[i, j, k] = 1
                        else: # negative spike
                            self.spike_state_view_sub[i, j, k] = -1

                        if self.is_refractory == True:
                            self.refractory_active_view[i, j, k] = self.refractory_view[j, k]

                        if index == 1:
                            if self.is_threshold_reset == True:
                                self.voltage_sub_view_next[i, j, k] = (math.fabsf(voltage) - threshold) * self.spike_state_view_sub[i, j, k]
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

                    if self.is_r_stdp == True:
                        if self.r_stdp.background_spikes[i,j,k] == 1.0 and self.spike_state_view_sub[i, j, k] == 0: self.spike_state_view_sub[i, j, k] = 1
                        is_spike = True if self.spike_state_view_sub[i, j, k] != 0 else False
                        self.r_stdp.update_neuron_trace(i, j, k, is_spike)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void update_voltage_with_weights_delay_refractory(self, int current_time):
        if current_time + 1 >= self.run_time_margin: return

        # 1 - Set voltage_view with next time step
        cdef int start = min(current_time+1, self.run_time_delay_max)
        cdef int end = min(start+1, self.run_time_delay_max)
        cdef float[:,:,:,:] voltage_view = self.voltage_view[:, :, :, start:end]


        cdef int[:, :, :] spike_state_sub_view = self.spike_state_sub
        cdef int i, j, neuron_in, neuron_out, a, connections
        cdef int delay
        cdef float spike_value

        # 2 - Get the active synapses indexes (in order to only iterate on them)
        connections = self.synapses_actives_indexes_view.shape[1]

        cdef float[:,:,:,:] augmented_decoder_view
        cdef float[:,:,:] augmented_decoder_view_prev
        if self.is_augmented == True:
            augmented_decoder_view_prev = self.augmented.augmented_decoder[:, :, :, current_time]
            augmented_decoder_view = self.augmented.augmented_decoder[:, :, :, start:end]
        
        for i in range(self.nb_episode):
            for j in range(self.nb_networks):
                for a in range(connections):
                    neuron_in = self.synapses_actives_indexes_view[0, a]
                    neuron_out = self.synapses_actives_indexes_view[1, a]

                    if self.is_r_stdp == True:
                        self.r_stdp.update_synpase_trace(i, j, neuron_in, neuron_out,
                                                        math.fabsf(spike_state_sub_view[i, j, neuron_in]) == 1, # is_spike_pre,
                                                        math.fabsf(spike_state_sub_view[i, j, neuron_out]) == 1 # is_spike_post
                                                        )

                    # 2 - Check if refractory and skip if needed
                    if self.is_refractory == True and self.refractory_active_view[i, j, neuron_in] > 0 and self.refractory_active_view[i, j, neuron_out] > 0: continue

                    # 3 - If spike, Update the voltage with the weights
                    if math.fabsf(spike_state_sub_view[i, j, neuron_in]) == 1.0:
                        spike_value = spike_state_sub_view[i, j, neuron_in]
                        delay = self.delay_view[j, neuron_in, neuron_out] if self.is_delay == True else 0
                        if delay + current_time >= self.run_time_margin: continue

                        if self.is_energy == True:
                            voltage_view[i, j, neuron_out, delay] += (self.weight_view[j, neuron_in, neuron_out] * self.energy.energy_view[i, j, self.energy.energy_index_view[i, j, neuron_in], neuron_in]) * spike_value
                        else:
                            voltage_view[i, j, neuron_out, delay] += self.weight_view[j, neuron_in, neuron_out] * spike_value

                        # 4 - Update the voltage_decoder_view (if augmented_decoder used with or without energy)
                        if self.is_augmented == True and neuron_out >= self.augmented.neuron_decoder_start and neuron_out <= self.augmented.neuron_decoder_end:

                            if self.augmented.is_neurons_update_with_augmented == True and augmented_decoder_view[i, j, neuron_in, 0] > 0.0:
                                if self.is_energy == True:
                                    augmented_decoder_view[i, j, neuron_out, delay] += (self.weight_view[j, neuron_in, neuron_out] * augmented_decoder_view_prev[i, j, neuron_in] * self.energy.energy_view[i, j, self.energy.energy_index_view[i, j, neuron_in], neuron_in]) * spike_value
                                else:
                                    augmented_decoder_view[i, j, neuron_out, delay] += (self.weight_view[j, neuron_in, neuron_out] * augmented_decoder_view_prev[i, j, neuron_in]) * spike_value

                            else:
                                if self.is_energy == True:
                                    augmented_decoder_view[i, j, neuron_out, delay] += (self.weight_view[j, neuron_in, neuron_out] * self.energy.energy_view[i, j, self.energy.energy_index_view[i, j, neuron_in], neuron_in]) * spike_value
                                else:
                                    augmented_decoder_view[i, j, neuron_out, delay] += self.weight_view[j, neuron_in, neuron_out] * spike_value

    @cython.boundscheck(False)
    # @cython.wraparound(False)
    cdef void init_network_CPU(self, np.ndarray[np.int32_t, ndim=1] pop_idx):
        self.nb_networks = pop_idx.shape[0]
        self.nb_neurons  = self.population.nb_neurons # total number of neurons
        self.nb_neurons_actives = self.population.neuron_active_global_indexes.size # input + hidden_active_global + output
        cdef int i, j, k

        # 1 - Dynamic variables
        self.voltage = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons, self.run_time_delay_max), dtype=np.float32)

        # 2 - Static variables
        self.weight    = self.population.weight[pop_idx]
        self.tau       = self.population.tau[pop_idx]
        self.threshold = self.population.threshold[pop_idx]
        self.constant_current_kernel = self.population.constant_current[pop_idx]
        self.voltage_reset = self.population.voltage_init[pop_idx]

        # 2 - Indexes
        self.input_indexes  = self.population.input_indexes
        self.hidden_indexes = self.population.hidden_indexes
        self.output_indexes = self.population.output_indexes

        # NEGATIVE_SPIKES
        self.input_indexes_start = self.input_indexes[0]
        self.hidden_indexes_start = self.hidden_indexes[0]
        self.output_indexes_start = self.output_indexes[0]

        self.input_indexes_end = self.input_indexes[-1]
        self.hidden_indexes_end = self.hidden_indexes[-1]
        self.output_indexes_end = self.output_indexes[-1]

        # print("self.input_indexes", self.input_indexes, "self.hidden_indexes", self.hidden_indexes, "self.output_indexes", self.output_indexes)
        # print("self.input_indexes_end", self.input_indexes_end, "self.hidden_indexes_end", self.hidden_indexes_end, "self.output_indexes_end", self.output_indexes_end)
        # exit()

        self.neurons_actives_indexes_view = self.population.neuron_active_global_indexes
        self.synapses_actives_indexes_view = self.population.synapse_active_global_indexes

        # (STDP) Set hidden weight to positive
        if self.is_r_stdp == True: self.r_stdp.set_hidden_weight_positive(self.weight, self.output_indexes_start, self.output_indexes_end)

        # 2 - Check if there is delay or refractory or bad values
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
            self.refractory_view   = self.refractory
            self.is_refractory = np.any(self.refractory != 0)
            self.refractory_active = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons), dtype=np.int32)
            self.refractory_active_view = self.refractory_active
        
        if self.is_LIF_beta == True:
            self.tau = np.clip(self.tau, 0.0, 1.0) # in case tau < 0 or tau > 1
            self.tau = 1.0 - self.tau # because tau = 1/tau in the LIF equation
        else:
            self.tau[self.tau <= 0] = 1.0 # in case of tau <= 0

        if self.disable_output_threshold == True: self.threshold[:, self.output_indexes] = 1e14 # set very high voltage to disable output reset


        # 3 - Transfer numpy memory to cython memoryview
        self.voltage_view = self.voltage
        self.voltage_reset_view = self.voltage_reset
        self.tau_view = self.tau
        self.constant_current_view = self.constant_current_kernel
        self.threshold_view = self.threshold
        self.spike_state = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons, self.run_time_margin), dtype=np.int32)
        self.spike_state_view = self.spike_state
        self.weight_view = self.weight
        
        # 4 - Set the first time step to 0
        self.first_time_step == 0


        self.neuron_to_record_indexes = np.array([], dtype=np.int32)
        if "input" in self.record_layer: self.neuron_to_record_indexes = np.concatenate((self.neuron_to_record_indexes, self.input_indexes))
        if "output" in self.record_layer: self.neuron_to_record_indexes = np.concatenate((self.neuron_to_record_indexes, self.output_indexes))
        if "hidden" in self.record_layer: self.neuron_to_record_indexes = np.concatenate((self.neuron_to_record_indexes, self.hidden_indexes))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void init_network_GPU(self, np.ndarray[np.int32_t, ndim=1] pop_idx, bint is_re_alloc):
        if HAS_CUDA == False: exit("From Runner_RL_cython: CUDA is not available")
        self.nb_networks = pop_idx.shape[0]
        self.nb_neurons = self.population.nb_neurons
        cdef int i,j, delay_shape_0, delay_shape_1

        # 1 - Static variables
        self.weight = self.population.weight[pop_idx]
        self.threshold = self.population.threshold[pop_idx]
        self.tau = self.population.tau[pop_idx]
        self.constant_current_kernel = None

        # 2 - Indexes
        self.input_indexes  = self.population.input_indexes
        self.hidden_indexes = self.population.hidden_indexes
        self.output_indexes = self.population.output_indexes

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
            self.is_refractory = np.any(self.refractory != 0)
            self.refractory_active = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons), dtype=np.int32)
        
        if self.is_LIF_beta == True:
            self.tau = np.clip(self.tau, 0.0, 1.0) # in case tau < 0 or tau > 1
            self.tau = 1.0 - self.tau # because tau = 1/tau in the LIF equation
        else:
            self.tau[self.tau <= 0] = 1.0 # in case of tau <= 0

        if self.disable_output_threshold == True: self.threshold[:, self.output_indexes] = 1e14 # set very high voltage to disable output reset

                
        # 4 - Set the first time step to 0
        self.first_time_step == 0

        if self.is_energy == True: self.weight_view = self.weight

        if HAS_CUDA: self.snn_gpu_ptr = init_SNN_GPU(
            # NULL, # snn_gpu_ptr
            self.snn_gpu_ptr if is_re_alloc == False else NULL, # Passing NULL to do a complete init with a malloc

            # Info run
            self.nb_episode,
            self.nb_networks,
            self.nb_neurons,
            self.run_time_delay_max,

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
            self.online,
            is_SL=False
        )

        if self.is_record_spike:
            self.spike_state = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons, self.run_time_margin), dtype=np.int32)
        elif self.is_record_voltage:
            self.voltage = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons, self.run_time_delay_max), dtype=np.float32)

        self.neuron_to_record_indexes = np.array([], dtype=np.int32)
        if "input" in self.record_layer: self.neuron_to_record_indexes = np.concatenate((self.neuron_to_record_indexes, self.input_indexes))
        if "output" in self.record_layer: self.neuron_to_record_indexes = np.concatenate((self.neuron_to_record_indexes, self.output_indexes))
        if "hidden" in self.record_layer: self.neuron_to_record_indexes = np.concatenate((self.neuron_to_record_indexes, self.hidden_indexes))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void init_network_step(self):

        if self.is_r_stdp == True:
            self.r_stdp.init_step(self.first_time_step, self.online)

        if self.is_energy == True and self.energy.energy_update_method != 0: # Need to reset the energy cause it is updated at each step by the energy_update_method
            self.energy.init_step(self.population, self.population_ids, self.first_time_step, self.online)

        if self.is_augmented == True:
            self.augmented.init_step(self.first_time_step, self.online)

        # 0 - If it is the first time step, do nothing
        if self.first_time_step == 0: # here the code is executed only once at the first time step
            self.first_time_step = 1
            return

        # 1 - Reset the voltage and add the previous last step voltage to the new voltage
        cdef np.ndarray[np.float32_t, ndim=4] new_voltage = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons, self.run_time_delay_max), dtype=np.float32)
        
        if self.online == True: # ONLINE Keep the last voltage
            if self.delay_max == 1:
                new_voltage[:, :, :, 0] = self.voltage[:, :, :, self.time_step]
            else:
                new_voltage[:, :, :, :self.delay_max-1] = self.voltage[:, :, :, self.time_step+1:]
            new_voltage[:,:,:,0] = new_voltage[:,:,:,0] + (-new_voltage[:,:,:,0] / self.tau * self.dt) # LIF FIX BUG CAUSE the last voltage need a lif step in order to be correct

                    
        self.voltage = new_voltage
        self.voltage_view = self.voltage

        # 2 - Reset spike_state in order to record the new spikes
        self.spike_state = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons, self.run_time_margin), dtype=np.int32)
        self.spike_state_view = self.spike_state


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void record(self):
        cdef int nb_snns = self.population_ids.size
        cdef int i, j
        cdef long[:,:,:] spike_sumed
        cdef float[:,:,:] voltage_record
        cdef np.ndarray[np.float32_t, ndim=3] spike_sumed_decoder

        # nb_neurons_to_record = self.neuron_to_record_indexes.shape[0]

        if self.is_record_spike:
            spike_sumed = np.sum(self.spike_state[:,:, self.neuron_to_record_indexes], axis=3)
        if self.is_record_voltage:
            voltage_record = self.voltage[:, :, self.neuron_to_record_indexes, -1]

        if self.is_record_augmented:
            # spike_sumed_decoder = np.sum(self.augmented_decoder[:,:, self.neuron_decoder_to_record_indexes], axis=3) 
            spike_sumed_decoder = np.sum(self.augmented.augmented_decoder[:,:, self.augmented.neuron_to_record_indexes], axis=3) 

            if self.augmented.is_normalize == True:
                spike_sumed_decoder = np.clip(spike_sumed_decoder/self.augmented.spike_max, 0, 1)

                if self.augmented.is_interpolate == True:
                    spike_sumed_decoder = np.interp(spike_sumed_decoder, (0, 1), (self.augmented.interpolate_min, self.augmented.interpolate_max)).astype(np.float32)
         
        for i in range(nb_snns):
            if self.is_record_spike: # SPIKE
                self.record_spikes[self.population_ids_list[i]] = spike_sumed[:, i, :]

            if self.is_record_voltage: # VOLTAGE
                self.record_voltages[self.population_ids_list[i]] = voltage_record[:, i, :]

            if self.is_record_augmented: # AUGMENTED
                self.record_augmented[self.population_ids_list[i]] = spike_sumed_decoder[:, i, :]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef dict get_record_augmented_spikes(self):
        return self.augmented.record_dict()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef dict get_record_spikes(self):
        # self.record()
        cdef int nb_snns = self.population_ids.size
        spike_sumed = np.sum(self.spike_state[:,:, self.neuron_to_record_indexes], axis=3)

        for i in range(nb_snns):
            if self.is_record_spike: # SPIKE
                self.record_spikes[self.population_ids_list[i]] = spike_sumed[:, i, :]

        return self.record_spikes

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef dict get_record_voltages(self):
        # self.record()
        cdef int nb_snns = self.population_ids.size
        voltage_record = self.voltage[:, :, self.neuron_to_record_indexes, -1]

        for i in range(nb_snns):
            if self.is_record_voltage: # VOLTAGE
                self.record_voltages[self.population_ids_list[i]] = voltage_record[:, i, :]

        return self.record_voltages

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray get_record_spikes_raw(self):
        return np.sum(self.spike_state[:,:, self.neuron_to_record_indexes], axis=3)

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray get_record_voltages_raw(self):
        return self.voltage[:, :, self.neuron_to_record_indexes, -1]

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray get_record_augmented_raw(self):
        return self.augmented.record_numpy()
    

    cpdef np.ndarray get_weight(self):
        return self.weight

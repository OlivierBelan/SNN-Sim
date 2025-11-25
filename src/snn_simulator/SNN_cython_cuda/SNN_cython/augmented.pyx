cimport cython
import numpy as np
cimport numpy as np
np.import_array()
cimport libc.math as math

cdef class Augmented:

    # cdef void init(self, int run_time, int run_time_margin, bint is_LIF_beta):

    #     self.run_time = run_time
    #     self.run_time_margin = run_time_margin
    #     self.is_LIF_beta = is_LIF_beta


        # # Init spikes importance
        # self.spike_max_time_step = <int>max(np.rint(self.spike_distribution_run/(self.run_time_margin-1)), 1)
        # if self.spike_distribution_importance > 0:
        #     self.spikes_importance = self.build_spike_importance(self.spike_distribution_importance, self.run_time_margin-1, is_descending=self.linear_spike_importance_type)
        # else:
        #     self.spikes_importance = np.zeros(self.run_time_margin-1, dtype=np.float32)
        # print("from augmented spike_importance", self.spikes_importance, "shape", np.shape(self.spikes_importance))


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void init_param(self, bint is_neurons_all_to_decode, int spike_max = 10, int spike_distribution_run = 10, int spike_distribution_importance = 10, str importance_type = "nothing", str linear_spike_importance_type = "descending", str spike_type = "positive", bint is_normalize = False, bint is_interpolate = False, float interpolate_max = 1.0, float interpolate_min = 0.0, bint is_voltage_reset=True, bint is_neurons_update_with_augmented = False):
        self.spike_max = spike_max
        self.spike_distribution_run = spike_distribution_run
        self.spike_distribution_importance = spike_distribution_importance
        self.linear_spike_importance_type = True if linear_spike_importance_type == "descending" else False

        self.is_neurons_all_to_decode = is_neurons_all_to_decode # If True augmented on all neuron ; If False on output layer only
        self.neuron_decoder_indexes = np.array([], dtype=np.int32)
        self.is_neurons_update_with_augmented = is_neurons_update_with_augmented # If True the voltage will be updated with the augmented spikes

        self.is_normalize = (is_normalize == True or is_interpolate == True)
        self.is_interpolate = is_interpolate
        self.interpolate_max = interpolate_max
        self.interpolate_min = interpolate_min
        self.is_voltage_reset = is_voltage_reset

        if importance_type == "first_index":
            self.importance_type = 0
        elif importance_type == "by_index":
            self.importance_type = 1
        # elif importance_type == "all":
        #     self.importance_type = 2
        elif importance_type == "nothing":
            self.importance_type = 3

        if spike_type == "positive":
            self.spike_format_type = 0
        elif spike_type == "absolute":
            self.spike_format_type = 1
        elif spike_type == "raw":
            self.spike_format_type = 2
        
        self.is_first_init = True



    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # cdef void init_importance(self, int run_time, int run_time_margin):
    #     self.run_time = run_time
    #     self.run_time_margin = run_time_margin

    #     # Init spikes importance
    #     self.spike_max_time_step = <int>max(np.rint(self.spike_distribution_run/(self.run_time_margin-1)), 1)
    #     if self.spike_distribution_importance > 0:
    #         self.spikes_importance = self.build_spike_importance(self.spike_distribution_importance, self.run_time_margin-1, is_descending=self.linear_spike_importance_type)
    #     else:
    #         self.spikes_importance = np.zeros(self.run_time_margin-1, dtype=np.float32)
    #     # print("from augmented spike_importance", self.spikes_importance, "shape", np.shape(self.spikes_importance))
    #     # exit()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void init_run(self, int run_time, int run_time_margin, bint is_LIF_beta, int nb_episodes, int nb_networks, int nb_neurons, np.ndarray[np.int32_t, ndim=1] population_ids, np.ndarray[np.int32_t, ndim=1] input_indexes, np.ndarray[np.int32_t, ndim=1] output_indexes, np.ndarray[np.int32_t, ndim=1] neuron_to_record_indexes, np.ndarray[np.float32_t, ndim=2] threshold, np.ndarray[np.float32_t, ndim=2] tau, np.ndarray[np.int32_t, ndim=2] refractory, bint is_refractory = False):

        self.nb_episodes = nb_episodes
        self.nb_networks = nb_networks
        self.nb_neurons = nb_neurons
        self.population_ids = population_ids

        if self.is_first_init == True:
            self.is_first_init = False
            self.run_time = run_time
            self.run_time_margin = run_time_margin
            self.is_LIF_beta = is_LIF_beta

            self.input_indexes = input_indexes
            self.output_indexes = output_indexes
            self.record_augmented = {}

            
            if self.is_neurons_all_to_decode == True:
                self.index_start_decoder_update = self.input_indexes.shape[0]
                self.neuron_decoder_indexes = np.arange(self.nb_neurons, dtype=np.int32)
                self.neuron_to_record_indexes = neuron_to_record_indexes
            else: # If False augmented only on output layer
                self.index_start_decoder_update = 0
                self.neuron_decoder_indexes = self.output_indexes
                self.neuron_to_record_indexes = self.output_indexes - self.input_indexes.shape[0]

            self.neuron_decoder_start = self.neuron_decoder_indexes[0]
            self.neuron_decoder_end = self.neuron_decoder_indexes[self.neuron_decoder_indexes.shape[0]-1]


            # Init spikes importance
            self.spike_max_time_step = <int>max(np.rint(self.spike_distribution_run/(self.run_time_margin-1)), 1)
            if self.spike_distribution_importance > 0:
                self.spikes_importance = self.build_spike_importance(self.spike_distribution_importance, self.run_time_margin-1, is_descending=self.linear_spike_importance_type)
            else:
                self.spikes_importance = np.zeros(self.run_time_margin-1, dtype=np.float32)
            # print("from augmented spike_importance", self.spikes_importance, "shape", np.shape(self.spikes_importance))
            # exit()

        # Init new decoder
        self.augmented_decoder = np.zeros((self.nb_episodes, self.nb_networks, len(self.neuron_decoder_indexes), self.run_time_margin), dtype=np.float32)
        self.voltage_decoder = np.zeros((self.nb_episodes, self.nb_networks, len(self.neuron_decoder_indexes)), dtype=np.float32)
        self.threshold_decoder = threshold[:, self.neuron_decoder_indexes]
        self.tau_decoder = tau[:, self.neuron_decoder_indexes]

        self.is_refractory = is_refractory
        if self.is_refractory == True:
            self.refractory_decoder = refractory[:, self.neuron_decoder_indexes]
            self.refractory_decoder_view = self.refractory_decoder

        self.voltage_decoder_view = self.voltage_decoder
        self.threshold_decoder_view = self.threshold_decoder
        self.tau_decoder_view = self.tau_decoder


    cdef void init_step(self, int first_time_step, bint is_online):
        if first_time_step == 0: # here the code is executed only once at the first time step
            return

        if is_online == False:
            # here it's offline (network is stateless), so we have to reset the voltage any new run
            # In the case of online (state_full network) == True, I have to keep the previous voltage_decoder, which is already done at this stage
            self.voltage_decoder = np.zeros((self.nb_episodes, self.nb_networks, len(self.neuron_decoder_indexes)), dtype=np.float32)
            self.voltage_decoder_view = self.voltage_decoder
        
        self.reset_record()

    cdef void reset_record(self):
        self.augmented_decoder = np.zeros((self.nb_episodes, self.nb_networks, len(self.neuron_decoder_indexes), self.run_time_margin), dtype=np.float32)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void update(self, int current_time):
        if current_time + 1 >= self.run_time_margin: return
        cdef int time_step = current_time + 1
        cdef float spike_floor
        cdef int i, j, k
        cdef float[:,:,:] augmented_decoder_view = self.augmented_decoder[:, :, :, time_step]
        cdef int nb_neurons_decoder = self.neuron_decoder_indexes.shape[0]
        cdef float voltage_decoder, threshold_decoder, tau_decoder

        for i in range(self.nb_episodes):
            for j in range(self.nb_networks):
                for k in range(self.index_start_decoder_update, nb_neurons_decoder):
                    self.voltage_decoder_view[i, j, k] += augmented_decoder_view[i, j, k]
                    voltage_decoder = self.voltage_decoder_view[i, j, k]
                    threshold_decoder = self.threshold_decoder_view[j, k]
                    tau_decoder = self.tau_decoder_view[j, k]
                    augmented_decoder_view[i, j, k] = 0.0 # reset the delayed information

                    # 1 - Update the augmented_decoder with the voltage_decoder (LIF Version)  -> (Σ_W/threshold) * (1 - (1/tau) * (1/(1+Refractory)) ; tau -> [1, +inf], refractory -> [0, +inf]
                    if self.is_LIF_beta == False and self.is_refractory == False and voltage_decoder > threshold_decoder:
                        augmented_decoder_view[i, j, k] = (voltage_decoder/threshold_decoder) * (1-(1/(tau_decoder))) 
                    
                    elif self.is_LIF_beta == False and self.is_refractory == True and voltage_decoder > threshold_decoder:
                        augmented_decoder_view[i, j, k] = (voltage_decoder/threshold_decoder) * (1-(1/(tau_decoder))) * (1/(1+self.refractory_decoder_view[i, j]))


                    # 1 - Update the augmented_decoder with the voltage_decoder (Beta Version) -> (Σ_W/threshold) * (tau) * (1/(1+Refractory)) ; tau -> [0, 1], refractory -> [0, +inf]
                    elif self.is_LIF_beta == True and self.is_refractory == False and voltage_decoder > threshold_decoder:
                        augmented_decoder_view[i, j, k] = (voltage_decoder/threshold_decoder) * tau_decoder

                    elif self.is_LIF_beta == True and self.is_refractory == True and voltage_decoder > threshold_decoder:
                        augmented_decoder_view[i, j, k] = (voltage_decoder/threshold_decoder) * tau_decoder * (1/(1+self.refractory_decoder_view[i, j]))
                    

                    spike_floor = math.floorf(augmented_decoder_view[i, j, k])


                    # 2 - Update the voltage_decoder with the augmented_decoder (if there is spikes) otherwise decay the voltage
                    # 2.1 - Hard-Reset the voltage if there is a spike
                    if self.is_voltage_reset == True and spike_floor >= 1.0:
                        self.voltage_decoder_view[i, j, k] = 0

                    # 2.2 - Soft-Reset the voltage if there is a spike (keep the remaining exceding voltage above the threshold)
                    elif self.is_voltage_reset == False and spike_floor >= 1.0:
                        self.voltage_decoder_view[i, j, k] = augmented_decoder_view[i, j, k] - spike_floor

                    # 2.3 - LIF - Decay the voltage if there is no spike
                    elif self.is_LIF_beta == False:
                        # self.voltage_decoder_view[i, j, k] = (self.voltage_decoder_view[i, j, k]) * (1-(1/(tau_decoder))) 
                        self.voltage_decoder_view[i, j, k] *= (1.0-(1.0/(tau_decoder))) 

                    # 2.4 - Beta - Decay the voltage if there is no spike
                    elif self.is_LIF_beta == True:
                        # self.voltage_decoder_view[i, j, k] = self.voltage_decoder_view[i, j, k] * tau_decoder
                        self.voltage_decoder_view[i, j, k] *= tau_decoder



                    # 3 - Add the rounded spike to the augmented_decoder
                    # 3.1 - Positive spike only
                    if self.spike_format_type == 0:
                        augmented_decoder_view[i, j, k] = spike_floor if spike_floor >= 0 else 0

                    # 3.2 - Abosolute spike only                    
                    elif self.spike_format_type == 1:
                        augmented_decoder_view[i, j, k] = math.fabs(spike_floor)

                    # 3.3 - Both positive and negative spikes
                    elif self.spike_format_type == 2:
                        augmented_decoder_view[i, j, k] = spike_floor
                    
                    # 4 - Clip the augmented_decoder to the maximum time step
                    if augmented_decoder_view[i, j, k] > self.spike_max_time_step: augmented_decoder_view[i, j, k] = self.spike_max_time_step



    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void apply_importance(self):
        cdef int i, j, k, l, index
        cdef int nb_neurons_decoder = self.neuron_decoder_indexes.shape[0]
        cdef float[:] spikes
        cdef float[:] spikes_importance_view = self.spikes_importance
        cdef float[:,:,:,:] augmented_decoder_view = self.augmented_decoder
        cdef bint is_first_index_found = False
        cdef int spikes_len = len(augmented_decoder_view[0, 0, 0])

        for i in range(self.nb_episodes):
            for j in range(self.nb_networks):
                for k in range(self.index_start_decoder_update, nb_neurons_decoder):
                # for k in range(nb_neurons_decoder):
                    spikes = augmented_decoder_view[i, j, k]
                    
                    if self.importance_type == 0: # first index
                        index = 0
                        is_first_index_found = False
                        for l in range(spikes_len):
                            if spikes[l] != 0:
                                is_first_index_found = True
                            if is_first_index_found == True:
                                if spikes[l] > self.spike_max_time_step: spikes[l] = self.spike_max_time_step
                                spikes[l] = spikes[l] + ((spikes[l]/self.spike_max_time_step) * spikes_importance_view[index])
                                index += 1

                    elif self.importance_type == 1: # by index
                        index = 0
                        for l in range(spikes_len):
                            if spikes[l] != 0:
                                if spikes[l] > self.spike_max_time_step: spikes[l] = self.spike_max_time_step
                                spikes[l] = spikes[l] + ((spikes[l]/self.spike_max_time_step) * spikes_importance_view[index])
                                index += 1

                    # elif self.importance_type == 2: # all ---> Due to the layer delay, it's impossible for the farthest neurons to reach their maximum spiking potential
                    #     for l in range(spikes_len):
                    #         if spikes[l] > self.spike_max_time_step: spikes[l] = self.spike_max_time_step
                    #         spikes[l] = spikes[l] + ((spikes[l]/self.spike_max_time_step) * spikes_importance_view[l])

                    elif self.importance_type == 3: # nothing
                        for l in range(spikes_len):
                            if spikes[l] > self.spike_max_time_step: spikes[l] = self.spike_max_time_step                    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef np.ndarray build_spike_importance(self, int spikes_distribution, int run_len, bint is_descending=False):

        cdef np.ndarray[np.float32_t, ndim=1] linespace, linespace_normalized, linespace_spikes
        # 1 - Create a linspace from spikes_distribution to 0 or inversement
        if is_descending == False:
            linespace = np.linspace(0, spikes_distribution, run_len).astype(np.float32)
        else:
            linespace = np.linspace(spikes_distribution, 0, run_len).astype(np.float32)

        # 2 - Normalize the linespace sum to make it sum to 1
        linespace_normalized = linespace / linespace.sum()

        # 3 - Scale the linespace on the scale of spikes
        linespace_spikes = spikes_distribution * linespace_normalized

        return linespace_spikes


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef dict record_dict(self):
        spike_sumed_decoder = np.sum(self.augmented_decoder[:,:, self.neuron_to_record_indexes], axis=3) 

        if self.is_normalize == True:
            spike_sumed_decoder = np.clip(spike_sumed_decoder/self.spike_max, 0, 1)

            if self.is_interpolate == True:
                spike_sumed_decoder = np.interp(spike_sumed_decoder, (0, 1), (self.interpolate_min, self.interpolate_max)).astype(np.float32)

        for i in range(self.nb_networks):
                self.record_augmented[self.population_ids[i]] = spike_sumed_decoder[:, i, :]

        return self.record_augmented


    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef np.ndarray record_numpy(self):
        cdef np.ndarray[np.float32_t, ndim=3] record_augmented
        # print("self.augmented_decoder\n", self.augmented_decoder, "shape", np.shape(self.augmented_decoder))
        record_augmented = np.sum(self.augmented_decoder[:,:, self.neuron_to_record_indexes], axis=3) 

        if self.is_normalize == True:
            record_augmented = np.clip(record_augmented/self.spike_max, 0, 1)

            if self.is_interpolate == True:
                record_augmented = np.interp(record_augmented, (0, 1), (self.interpolate_min, self.interpolate_max))
        return record_augmented

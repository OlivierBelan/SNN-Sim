cimport cython
import numpy as np
cimport numpy as np
np.import_array()
cimport libc.math as math
from SNN_cython_cuda.SNN_cython.tools_cython cimport get_time, norm_min_max_all, norm_min_max_rows, norm_min_max_columns, norm_L1_all, norm_L1_rows, norm_L1_columns, norm_L1_sum_all, norm_L1_sum_rows, norm_L1_sum_columns, norm_L2_all, norm_L2_rows, norm_L2_columns, norm_L2_sum_all, norm_L2_sum_rows, norm_L2_sum_columns
from SNN_cython_cuda.SNN_cython.snn_cython cimport SNN_cython_population
from .augmented cimport Augmented

cdef class Energy:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef void init_param(self, str energy_update_method = "constant", str energy_norm = "min_max_column", int energy_length = 1, bint is_energy_battery = False, bint energy_is_interp = False, float energy_interp_min = -1.0, float energy_interp_max = 1.0, bint energy_keep_sign = False, int energy_decimal = -1):
        self.energy_norm = energy_norm
        self.energy_length = energy_length
        self.energy_is_interp = energy_is_interp
        self.energy_interp_min = energy_interp_min
        self.energy_interp_max = energy_interp_max
        self.energy_keep_sign = energy_keep_sign
        self.is_energy_battery = is_energy_battery
        self.energy_decimal = energy_decimal

        if energy_update_method == "constant":    self.energy_update_method = 0
        elif energy_update_method == "ascending": self.energy_update_method = 1
        elif energy_update_method == "descending":self.energy_update_method = 2
        elif energy_update_method == "rate":      self.energy_update_method = 3
        elif energy_update_method == "weight_acceleration": self.energy_update_method = 4
        else: self.energy_update_method = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void init_network_run(self, SNN_cython_population population, np.ndarray[np.int32_t, ndim=1] pop_idx, float[:,:,:] weight_view, int run_time, int run_time_margin, int nb_episodes, int nb_networks, int nb_neurons, int nb_inputs):
        self.run_time = run_time
        self.run_time_margin = run_time_margin
        self.nb_episodes = nb_episodes
        self.nb_networks = nb_networks
        self.nb_neurons = nb_neurons
        self.nb_inputs = nb_inputs


        self.energy = np.empty((self.nb_episodes, self.nb_networks, self.energy_length, self.nb_neurons), dtype=np.float32)
        self.energy[:] = population.energy[pop_idx]
        self.energy_index = np.zeros((self.nb_episodes, self.nb_networks, self.nb_neurons), dtype=np.int32)
        if self.energy_update_method == 4: # weight_acceleration
            self.weight_acceleration_record = np.zeros((self.nb_episodes, self.nb_networks, self.nb_neurons), dtype=np.float32)

        # if self.is_energy_battery == True:
        #     self.energy_battery = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons), dtype=np.float32)
        #     self.energy_battery[:] = self.population.energy_battery[pop_idx]
        #     self.energy_battery_view = self.energy_battery

        self.energy_view = self.energy
        self.energy_index_view = self.energy_index

        # print("0 - self.energy\n", self.energy, "shape", np.shape(self.energy))
        # print("0 - self.energy_index\n", self.energy_index, "shape", np.shape(self.energy_index))
        # print("energy_battery", self.energy_battery, "shape", np.shape(self.energy_battery))
        # exit()


        # print("before norm weight\n", self.weight, "shape", np.shape(self.weight))
        # print("norm_FROM_NUMPY\n", norm_L2_rows_numpy(self.weight.copy()))
        # return
        if   self.energy_norm == "none": return
        elif self.energy_norm == "min_max_all": norm_min_max_all(weight_view, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, round_decimal=self.energy_decimal)
        elif self.energy_norm == "min_max_row": norm_min_max_rows(weight_view, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, round_decimal=self.energy_decimal)
        elif self.energy_norm == "min_max_column": norm_min_max_columns(weight_view, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, round_decimal=self.energy_decimal)

        elif self.energy_norm == "L1_all": norm_L1_all(weight_view, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, is_abs=True, round_decimal=self.energy_decimal)
        elif self.energy_norm == "L1_row": norm_L1_rows(weight_view, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, is_abs=True, round_decimal=self.energy_decimal)
        elif self.energy_norm == "L1_column": norm_L1_columns(weight_view, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, is_abs=True, round_decimal=self.energy_decimal)

        elif self.energy_norm == "L1_sum_all": norm_L1_sum_all(weight_view, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, keep_sign=self.energy_keep_sign, round_decimal=self.energy_decimal)
        elif self.energy_norm == "L1_sum_row": norm_L1_sum_rows(weight_view, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, keep_sign=self.energy_keep_sign, round_decimal=self.energy_decimal)
        elif self.energy_norm == "L1_sum_column": norm_L1_sum_columns(weight_view, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, keep_sign=self.energy_keep_sign, round_decimal=self.energy_decimal)

        elif self.energy_norm == "L2_all": norm_L2_all(weight_view, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, round_decimal=self.energy_decimal)
        elif self.energy_norm == "L2_row": norm_L2_rows(weight_view, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, round_decimal=self.energy_decimal)
        elif self.energy_norm == "L2_column": norm_L2_columns(weight_view, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, round_decimal=self.energy_decimal)

        elif self.energy_norm == "L2_sum_all": norm_L2_sum_all(weight_view, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, keep_sign=self.energy_keep_sign, round_decimal=self.energy_decimal)
        elif self.energy_norm == "L2_sum_row": norm_L2_sum_rows(weight_view, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, keep_sign=self.energy_keep_sign, round_decimal=self.energy_decimal)
        elif self.energy_norm == "L2_sum_column": norm_L2_sum_columns(weight_view, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, keep_sign=self.energy_keep_sign, round_decimal=self.energy_decimal)

        # print("0 - after norm weight\n", self.weight, "shape", np.shape(self.weight))
        # print("sum all weight\n", np.sum(np.abs(self.weight), axis=(1, 2), keepdims=True))
        # print("sum col weight\n", np.sum(np.abs(self.weight), axis=1, keepdims=True))
        # print("sum row weight\n", np.sum(np.abs(self.weight), axis=2, keepdims=True))
        # exit()
        # self.weight_view = self.weight

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void init_step(self, SNN_cython_population population, np.ndarray[np.int32_t, ndim=1] pop_idx, int first_time_step, bint is_online):
        if first_time_step == 0: # code here will be call only at the first time step on the run
            return

        if is_online == True: return

        # On OFFLINE MODE Need to reset the energy cause it is updated at each step by the energy_update_method
        
        self.energy = np.empty((self.nb_episodes, self.nb_networks, self.energy_length, self.nb_neurons), dtype=np.float32)
        self.energy[:] = population.energy[pop_idx]

        if self.is_energy_battery == True:
            self.energy_battery = np.empty((self.nb_episodes, self.nb_networks, self.nb_neurons), dtype=np.float32)
            self.energy_battery[:] = population.energy_battery[pop_idx]
                
        # 2 - Reset the energy index if the energy_length > 1
        if self.energy_length > 1:
            self.energy_index = np.zeros((self.nb_episodes, self.nb_networks, self.nb_neurons), dtype=np.int32)

        # 3 - Reset the weight acceleration record if the energy_update_method == weight_acceleration
        if self.energy_update_method == 4: # weight_acceleration
            self.weight_acceleration_record = np.zeros((self.nb_episodes, self.nb_networks, self.nb_neurons), dtype=np.float32)

        self.energy_view = self.energy
        self.energy_index_view = self.energy_index
        # self.energy_battery_view = self.energy_battery


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void update(self, int current_time, float[:,:,:,:] spike_state, Augmented augmented = None, bint is_augmented = False):
        if current_time + 1 >= self.run_time_margin: return
        cdef int energy_index
        cdef int energy_index_max = self.energy_length - 1
        # cdef float[:, :, :] spike_state_view = self.augmented_decoder[:, :, :, current_time] if self.is_augmented == True else self.spike_state[:, :, :, current_time]
        cdef float[:, :, :] spike_state_view = augmented.augmented_decoder[:, :, :, current_time] if is_augmented == True else spike_state[:, :, :, current_time]
        cdef float[:, :, :] spike_state_current_time_view = spike_state[:, :, :, current_time]
        # cdef int[:, :, :] input_spike_state_view = self.spike_state[:, :, :input_size, current_time]
            

        # print("weight\n", np.array(self.weight_view), "shape", np.shape(self.weight_view))
        # print("voltages\n", np.array(self.voltage_view), "shape", np.shape(self.voltage_view))
        # print("energy\n", np.array(self.energy_view), "shape", np.shape(self.energy_view))
        # print("energy_index\n", np.array(self.energy_index_view), "shape", np.shape(self.energy_index_view))
        # print("energy_battery\n", np.array(self.energy_battery_view), "shape", np.shape(self.energy_battery_view))
        # print("spike_state", self.spike_state, "shape", np.shape(self.spike_state))
        # print("augmented_decoder_view", self.augmented_decoder, "shape", np.shape(self.augmented_decoder))
        # print("spike_state_view", np.array(spike_state_view), "shape", np.shape(spike_state_view))
        # print("input_spike_state_view", np.array(input_spike_state_view), "shape", np.shape(input_spike_state_view))
        # exit()


        # 0 - Update the energy
        cdef int i, j, k
        for i in range(self.nb_episodes):
            for j in range(self.nb_networks):
                for k in range(self.nb_neurons):
                    
                    # 1 - If input neuron has spike and is augmented_decoder, add the input spike to the decoder record spike_state
                    if is_augmented and (k < self.nb_inputs and spike_state_current_time_view[i, j, k] >= 1):
                       spike_state_view[i, j, k] = spike_state_current_time_view[i, j, k]

                    # 2 - If neuron has spike, Update the energy (have to think to integrate augmented spikes in the energy!!!!!!)
                    if spike_state_view[i, j, k] >= 1.0:
                        energy_index = self.energy_index_view[i, j, k]

                        # 3 - if energy_battery, subtract the energy
                        # if self.is_energy_battery== True:
                        #     self.energy_battery_view[i, j, k] -= self.energy_view[i, j, energy_index, k]
                        #     if self.energy_battery_view[i, j, k] < 0: self.energy_battery_view[i, j, k] = 0

                        # 4 - Update the energy_index
                        energy_index = energy_index + 1  if energy_index < energy_index_max else 0
                        self.energy_index_view[i, j, k] = energy_index

                        # 5 - Energy Update method when there is a spike
                        if self.energy_update_method   == 1: # ascending
                            self.energy_view[i, j, energy_index, k] += self.energy_view[i, j, energy_index, k] * 0.10

                        elif self.energy_update_method == 2: # descending
                            self.energy_view[i, j, energy_index, k] -= self.energy_view[i, j, energy_index, k] * 0.10

                        elif self.energy_update_method == 3: # rate
                            pass

                        elif self.energy_update_method == 4: # weight_acceleration
                            if self.weight_acceleration_record[i, j, k] > 0:
                                self.energy_view[i, j, energy_index, k] *= spike_state_view[i, j, k] / self.weight_acceleration_record[i, j, k]
                            self.weight_acceleration_record[i, j, k] = spike_state_view[i, j, k]

                        else: # constant
                            continue

        # if current_time > 3:
        #     exit()
        # exit()
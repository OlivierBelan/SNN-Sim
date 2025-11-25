
cimport cython
import numpy as np
cimport numpy as np
np.import_array()

from libc.stdlib cimport rand
from libc.stdlib cimport RAND_MAX
cimport libc.math as math
from .encoder cimport EncoderType
cdef class Encoder:

    cpdef void init_RL(self, int run_time, int run_time_margin):
        self.run_time = run_time
        self.run_time_margin = run_time_margin
        self.is_RL = True

        # Combinatorial Encoder
        self.is_combinatorial_table_init = False
        self.combinatorial_encoder_table = None

        # Derivative Encoder
        self.prev_inputs_data = None
        self.prev_inputs_spike = None
        

    cpdef void init_SL(self, int run_time, int run_time_margin):
        self.run_time = run_time
        self.run_time_margin = run_time_margin
        self.is_RL = False

        # Combinatorial Encoder
        self.is_combinatorial_table_init = False
        self.combinatorial_encoder_table = None

        # Derivative Encoder
        self.prev_inputs_data = None
        self.prev_inputs_spike = None



    cdef np.ndarray[np.float32_t, ndim=4] encode(self, np.ndarray data_raw, bint is_encoded = False):
        if self.encoder_type == EncoderType.POISSON:
            return self.poisson_encoder(data_raw, self.rate, self.max_nb_spikes)

        elif self.encoder_type == EncoderType.BINOMIAL:
            return self.binomial_encoder(data_raw, self.max_nb_spikes, self.reduce_noise)

        elif self.encoder_type == EncoderType.RATE:
            return self.rate_encoder(data_raw)

        elif self.encoder_type == EncoderType.EXACT:
            return self.exact_encoder(data_raw, self.max_nb_spikes)

        elif self.encoder_type == EncoderType.RAW or is_encoded == True:
            return self.raw_encoder(data_raw)

        elif self.encoder_type == EncoderType.DIRECT:
            return self.direct_encoder(data_raw, self.direct_min, self.direct_max)

        elif self.encoder_type == EncoderType.COMBINATORIAL:
            return self.combinatorial_encoder(data_raw, combinatorial_factor=self.combinatorial_factor, combinaison_size=self.combinaison_size, combinaison_size_max=self.combinaison_size_max, combinaison_noise=self.combinaison_noise, combinatorial_roll=self.combinatorial_roll, combinatorial_filter=self.combinatorial_filter, combinatorial_modulo=self.combinatorial_modulo, combinatorial_print_table_debug=self.combinatorial_print_table_debug)

        elif self.encoder_type == EncoderType.DERIVATIVE:
            return self.derivative_encoder(data_raw, threshold=self.derivative_threshold, is_latency=self.derivative_is_latency, is_latency_positional=self.derivative_is_latency_positional, use_prev_input=self.derivative_use_prev_input, max_delta_latency=self.derivative_max_delta_latency)
        else:
            raise ValueError("Unknown encoder type or encoder type not set. Please set the encoder type before encoding data.")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float32_t, ndim=4] poisson_encoder_3(self, np.ndarray data_raw, int rate, int max_nb_spikes = 3):
        cdef int i, j, k, m, len_indexes, idx, offset, episode_batch, nb_population, nb_inputs
        cdef int[:] indexes

        cdef np.ndarray[np.int32_t, ndim=1] poisson
        cdef np.ndarray[np.float32_t, ndim=4] data_encoded
        cdef np.ndarray[np.int32_t, ndim=3] inputs_data_poisson                

        if self.is_RL == True:
            offset = <int>np.rint(0.12*self.run_time) # little hack to make the distribution more centered
            data_encoded = np.zeros([data_raw.shape[0], data_raw.shape[1], data_raw.shape[2], self.run_time_margin], dtype=np.float32)
            inputs_data_poisson = np.random.poisson(lam=data_raw[0].T*self.run_time, size=(self.run_time, data_raw.shape[2], data_raw.shape[1])).T.astype(np.int32)
        else: # SL
            offset = <int>np.rint(0.0*self.run_time) # little hack to make the distribution more centered
            data_encoded = np.zeros([data_raw.shape[0], 1, data_raw.shape[1], self.run_time_margin], dtype=np.float32)
            inputs_data_poisson = np.random.poisson(lam=data_raw.T*self.run_time, size=(self.run_time, data_raw.shape[1], data_raw.shape[0])).T.astype(np.int32)


        episode_batch = <int>inputs_data_poisson.shape[0] # number of episodes
        nb_population = <int>inputs_data_poisson.shape[1] # number of populations
        max_nb_spikes = max_nb_spikes if max_nb_spikes < self.run_time else self.run_time
        for i in range(episode_batch):
            for m in range(nb_population):

                # indexes = self.n_most_frequent_np(inputs_data_poisson[i, m], max_nb_spikes) - offset # little hack to make the distribution more centered
                indexes = self.n_most_frequent_np_2(inputs_data_poisson[i, m], max_nb_spikes, offset) # little hack to make the distribution more centered
                # indexes = self.n_most_frequent_np_3(inputs_data_poisson[i, m], max_nb_spikes) - offset # little hack to make the distribution more centered

                len_indexes = <int>len(indexes)
                for k in range(len_indexes):
                    idx = indexes[k]

                    if self.is_RL == True:
                        if idx >= self.run_time:
                            # data_encoded[0, i, m, self.run_time-1] = 1.0
                            continue
                        elif idx < 0:
                            # data_encoded[0, i, m, 0] = 1.0
                            continue
                        else:
                            data_encoded[0, i, m, idx] = 1.0

                    else: # SL
                        if idx >= self.run_time:
                            data_encoded[i, 0, m, self.run_time-1] = 1.0
                        elif idx < 0:
                            data_encoded[i, 0, m, 0] = 1.0
                        else:
                            data_encoded[i, 0, m, idx] = 1.0
        return data_encoded

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float32_t, ndim=4] poisson_encoder_2(self, np.ndarray data_raw, int rate, int max_nb_spikes = 3):
        exit("I suspect poisson_encoder_2 does not work properly for RL mode or SL mode, use poisson_encoder_3 or poisson instead")
        cdef int i, j, k, m, n, len_indexes, idx, offset, shape_0, shape_1
        cdef int[:] indexes
        cdef np.ndarray[np.float32_t, ndim=4] data_encoded
        cdef np.ndarray[np.int32_t, ndim=1] poisson
        max_nb_spikes = max_nb_spikes if max_nb_spikes < self.run_time else self.run_time

        
        if self.is_RL == True: 
            offset = <int>np.rint(0.12*self.run_time) # little hack to make the distribution more centered
            shape_0 = data_raw.shape[1] # number of episodes
            shape_1 = data_raw.shape[2] # number of populations
            data_encoded = np.zeros([data_raw.shape[0], data_raw.shape[1], data_raw.shape[2], self.run_time_margin], dtype=np.float32)
        else:
            offset = <int>np.rint(0.0*self.run_time) # little hack to make the distribution more centered
            shape_0 = data_raw.shape[0] # number of episodes
            shape_1 = data_raw.shape[1] # number of populations
            data_encoded = np.zeros([data_raw.shape[0], 1, data_raw.shape[1], self.run_time_margin], dtype=np.float32)


        for i in range(shape_0):
            for j in range(shape_1):
                if self.is_RL == True: 
                    m = 0; n = i
                    poisson = np.random.poisson(lam=data_raw[0, i, j]*self.run_time, size=rate).astype(np.int32)
                    indexes = self.n_most_frequent_np_2(poisson, max_nb_spikes, offset)
                else: 
                    m = i; n = 0
                    poisson = np.random.poisson(lam=data_raw[i, j]*self.run_time, size=rate).astype(np.int32)
                    indexes = self.n_most_frequent_np_2(poisson, max_nb_spikes, offset)

                len_indexes = <int>len(indexes)
                for k in range(len_indexes):
                    idx = indexes[k]
                    if idx >= self.run_time:
                        data_encoded[m, n, j, self.run_time-1] = 1.0
                    elif idx < 0:
                        data_encoded[m, n, j, 0] = 1.0
                    else:
                        data_encoded[m, n, j, idx] = 1.0

        return data_encoded

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float32_t, ndim=4] poisson_encoder(self, np.ndarray data_raw, int rate, int max_nb_spikes = 3):
        if self.is_RL == False: data_raw = data_raw[:, np.newaxis, :]  # add a new axis for the population dimension
        cdef int i, j, k, m, len_indexes, shape_0, shape_1, shape_2, episode_batch, nb_population, nb_inputs
        cdef int[:] indexes
        cdef np.ndarray[np.int32_t, ndim=1] poisson
        cdef float[:, :, :] data_raw_view = data_raw
        cdef np.ndarray[np.float32_t, ndim=4] data_encoded = np.zeros([data_raw.shape[0], data_raw.shape[1], data_raw.shape[2], self.run_time_margin], dtype=np.float32)


        episode_batch = data_raw.shape[0] # number of episodes
        nb_population = data_raw.shape[1] # number of populations
        nb_inputs = data_raw.shape[2] # number of inputs

        max_nb_spikes = max_nb_spikes if max_nb_spikes < self.run_time else self.run_time
        for i in range(episode_batch):
            for m in range(nb_population):
                for j in range(nb_inputs):
                    poisson = np.random.poisson(lam=data_raw_view[i, m, j]*self.run_time, size=rate).astype(np.int32)
                    indexes = self.n_most_frequent_np(poisson, max_nb_spikes)
                    len_indexes = len(indexes)
                    for k in range(len_indexes):
                        if indexes[k] < (<int>self.run_time):
                            data_encoded[i, m, j, indexes[k]] = 1.0
        return data_encoded


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float32_t, ndim=4] binomial_encoder(self, np.ndarray data_raw, int max_nb_spikes = 3, int reduce_noise = 100):
        if self.is_RL == False: data_raw = data_raw[:, np.newaxis, :]  # add a new axis for the population dimension
        cdef int run_time_binomial = self.run_time - 1
        cdef int[:] indexes
        cdef int i, j, k, m, len_indexes, episode_batch, nb_population, nb_inputs
        cdef np.ndarray[np.int32_t, ndim=1] binomial
        cdef float[:, :, :] data_raw_view = data_raw
        cdef np.ndarray[np.float32_t, ndim=4] data_encoded = np.zeros([data_raw.shape[0], data_raw.shape[1], data_raw.shape[2], <int>self.run_time_margin], dtype=np.float32)


        episode_batch = data_raw.shape[0] # number of episodes
        nb_population = data_raw.shape[1] # number of populations
        nb_inputs = data_raw.shape[2] # number of inputs

        max_nb_spikes = max_nb_spikes if max_nb_spikes < self.run_time else self.run_time
        for i in range(episode_batch):
            for m in range(nb_population):
                for j in range(nb_inputs):
                    binomial = np.random.binomial(n=run_time_binomial, p=data_raw_view[i, m, j], size=max_nb_spikes + reduce_noise).astype(np.int32)
                    indexes = self.n_most_frequent_np(binomial, max_nb_spikes)
                    len_indexes = len(indexes)
                    for k in range(len_indexes):
                        data_encoded[i, m, j, indexes[k]] = 1.0
        return data_encoded


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float32_t, ndim=4] rate_encoder(self, np.ndarray data_raw):
        if self.is_RL == False: data_raw = data_raw[:, np.newaxis, :]  # add a new axis for the population dimension
        cdef int i, j, k, m
        cdef int episode_batch, nb_population, nb_inputs
        cdef float[:, :, :] data_raw_view = data_raw
        cdef np.ndarray[np.float32_t, ndim=4] data_encoded = np.zeros([data_raw.shape[0], data_raw.shape[1], data_raw.shape[2], <int>self.run_time_margin], dtype=np.float32)

        episode_batch = data_raw.shape[0] # number of episodes (RL) / batch size (SL)
        nb_population = data_raw.shape[1] # number of populations
        nb_inputs     = data_raw.shape[2] # number of inputs

        for i in range(episode_batch):
            for m in range(nb_population):
                for j in range(nb_inputs):
                    for k in range(self.run_time):
                        if rand() < (data_raw_view[i, m, j] * RAND_MAX): # bernoulli_trial (binomial) C version
                            data_encoded[i, m, j, k] = 1.0
        return data_encoded

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float32_t, ndim=4] exact_encoder(self, np.ndarray data_raw, int max_nb_spikes = 3):
        if self.is_RL == False: data_raw = data_raw[:, np.newaxis, :]  # add a new axis for the population dimension
        cdef int shift_max, shift_min, middle
        cdef int shift_max_operation = (max_nb_spikes//2) +1
        cdef int shift_min_operation = (max_nb_spikes//2)
        cdef int i, j, k, episode_batch, nb_population, nb_inputs
        cdef float[:, :, :] data_raw_view = data_raw * self.run_time
        cdef np.ndarray[np.float32_t, ndim=4] data_encoded = np.zeros([data_raw.shape[0], data_raw.shape[1], data_raw.shape[2], self.run_time_margin], dtype=np.float32)

        episode_batch = data_raw.shape[0] # number of episodes (RL) / batch size (SL)
        nb_population = data_raw.shape[1] # number of populations
        nb_inputs = data_raw.shape[2] # number of inputs

        for i in range(episode_batch):
            for k in range(nb_population):
                for j in range(nb_inputs):
                    middle = <int>math.ceilf(data_raw_view[i, k, j]) - 1
                    shift_min = 0 if 0 > (middle - shift_min_operation) else (middle - shift_min_operation)
                    shift_max = <int>self.run_time if (<int>self.run_time) < (middle + shift_max_operation) else (middle + shift_max_operation)
                    data_encoded[i, k, j, shift_min:shift_max] = 1.0
        return data_encoded

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float32_t, ndim=4] raw_encoder(self, np.ndarray data_raw):
        # Add padding (if needed)
        if self.is_RL == False: data_raw = data_raw[:, np.newaxis, :]  # add a new axis for the population dimension
        data_raw = self.add_padding_input_data(data_raw, self.run_time, self.run_time_margin)
        return data_raw.astype(np.float32)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float32_t, ndim=4] direct_encoder(self, np.ndarray data_raw, float direct_min = -100_000, float direct_max = 100_000):

        if self.is_RL == False: data_raw = data_raw[:, np.newaxis, :]  # add a new axis for the population dimension
        cdef int i, j, k, m
        cdef float[:, :, :] inputs_data_view = data_raw
        cdef float value

        cdef np.ndarray[np.float32_t, ndim=4] data_encoded = np.zeros([data_raw.shape[0], data_raw.shape[1], data_raw.shape[2], self.run_time_margin], dtype=np.float32)
        episode_batch = data_raw.shape[0] # number of episodes (RL) / batch size (SL)
        nb_population = data_raw.shape[1] # number of populations
        nb_inputs = data_raw.shape[2] # number of inputs

        for i in range(episode_batch):
            for j in range(nb_population):
                for k in range(nb_inputs):
                    value = inputs_data_view[i, j, k]
                    for m in range(self.run_time): # add padding by using run_time_original
                        data_encoded[i, j, k, m] = max(direct_min, min(value, direct_max)) # clip the value between min and max
        # print("direct_encoder\n", np.array(self.data_encoded), "shape", np.shape(self.data_encoded))
        # exit()
        return data_encoded


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray[np.float32_t, ndim=4] combinatorial_encoder(self, np.ndarray data_raw, int combinatorial_factor = 1, int combinaison_size=1, int combinaison_size_max=1, float combinaison_noise=0.0, bint combinatorial_roll=True, str combinatorial_filter = "energy", float[:] combinatorial_modulo = None, bint combinatorial_print_table_debug = False):

        self.combinatorial_encoder_build_table(combinatorial_factor, combinaison_size, combinaison_size_max, combinatorial_roll, combinatorial_filter)
        if combinatorial_print_table_debug == True: print("combinatorial table:\n", self.combinatorial_encoder_table, "shape:", np.shape(self.combinatorial_encoder_table)); exit()
        if self.is_RL == False: 
            data_raw = data_raw[:, np.newaxis, :]
            if "dynamic" in combinatorial_filter: exit("Combinatorial modulo dynamic not yet available/implemented in SL mode..... but in RL mode yes")

        cdef int encoder_size = self.combinatorial_encoder_table.shape[0] - 1
        cdef int i, j, k, m, index, episode_batch, nb_population, nb_inputs

        data_raw = np.rint(np.interp(data_raw, [0, 1], [0, encoder_size])).astype(np.int32)
        cdef int[:, :, :] data_raw_view = data_raw
        cdef np.ndarray[np.float32_t, ndim=4] data_encoded = np.zeros([data_raw.shape[0], data_raw.shape[1], data_raw.shape[2], self.run_time_margin], dtype=np.float32)

        episode_batch = data_raw.shape[0] # number of episodes (RL) / batch size (SL)
        nb_population = data_raw.shape[1] # number of populations
        nb_inputs = data_raw.shape[2] # number of inputs


        # Modulo filter parameters
        cdef bint is_modulo_dynamic = True if "dynamic" in combinatorial_filter else False
        cdef float table_size = <float>self.combinatorial_encoder_table.shape[0]
        cdef float modulo = 0.0
        # cdef float mu = 0.0

        for i in range(episode_batch):
            for j in range(nb_population):
                if is_modulo_dynamic:

                    modulo = table_size - ((1-combinatorial_modulo[j]%1)**(1 - math.expf(-10/table_size)) * table_size)
                    # print("modulo", modulo, "mu", mu, "table_size", table_size)
                    # print("table_size - ((1 - mu%1)**(1 - np.exp(-10/table_size)) / 1.0 * table_size) % table_size -> ",  table_size - ((1 - mu%1)**(1 - np.exp(-10/table_size)) / 1.0 * table_size) % table_size)
                    # print("table_size - ((1 - mu%1)**(1 - np.exp(-10/table_size)) * table_size) ->                    ",  table_size - ((1 - mu%1)**(1 - np.exp(-10/table_size)) * table_size))
                    # print("table_size - ((1 - mu%1)**(1 - math.exp(-10/table_size)) * table_size) ->                  ",  table_size - ((1 - mu%1)**(1 - math.exp(-10/table_size)) * table_size))
                for k in range(nb_inputs):
                    index = data_raw_view[i, j, k]
                    if combinaison_noise != 0.0:
                        index = max(0, min(np.round(np.random.normal(index, combinaison_noise)).astype(np.int32), encoder_size))
                        # print("From ENCODER: index with noise", index)

                    if is_modulo_dynamic == True:
                        # index = np.random.randint(0, encoder_size+1)
                        # index = <int>math.rint(index_modulo - (index_modulo%modulo))
                        index = <int>math.rint(<float>index - math.fmodf(<float>index, modulo))
                        # print()
                    for m in range(self.run_time): # add padding by using run_time_original
                        data_encoded[i, j, k, m] = self.combinatorial_encoder_table_view[index, m]
                        # print("FROM ENCODER: data_encoded[i, j, k, m]", data_encoded[i, j, k, m], "index", index, "m", m, "self.combinatorial_encoder_table_view[index, m]", self.combinatorial_encoder_table_view[index, m])
        return data_encoded


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void combinatorial_encoder_build_table(self, int combinatorial_factor=1, int combinaison_size=1, int combinaison_size_max=1, bint combinatorial_roll=True, str combinatorial_filter = "energy"):
        if self.is_combinatorial_table_init == True: return
        cdef bint is_energy = True if "energy" in combinatorial_filter else False # check ["energy", "modulo_energy_static", "modulo_energy_dynamic"]
        cdef bint is_modulo_static = True if ("modulo" in combinatorial_filter) and ("static" in combinatorial_filter) else False # check ["modulo_static", "modulo_energy_static"]
        cdef bint is_binary = True if "binary" in combinatorial_filter else False
        cdef bint is_random = True if "random" in combinatorial_filter else False
        cdef bint is_number_ones = True if "number_ones" in combinatorial_filter else False
        cdef np.ndarray[np.float32_t, ndim=2] combinatorial_table_modulo
        cdef float modulo_filter

        cdef int i = 0
        cdef str binary_repr
        cdef int first_one_index
        cdef int run_len = self.run_time // combinatorial_factor
        cdef int combinatorial_encoder_len = min(2**run_len, combinaison_size_max) if run_len < 50 else min(2**50, combinaison_size_max) # to avoid memory error
        cdef np.ndarray[np.float32_t, ndim=1] row_sums 
        cdef long[:] sorted_indices, indices_to_keep

        # Binary Encoder -> Binary Order
        self.combinatorial_encoder_table = np.empty((combinatorial_encoder_len, run_len), dtype=np.float32)
        for i in range(combinatorial_encoder_len):
            binary_repr = format(i, f'0{run_len}b')
            self.combinatorial_encoder_table[i] = [float(x) for x in binary_repr]

        # self.combinatorial_encoder_table = self.combinatorial_encoder_table[np.lexsort((self.combinatorial_encoder_table @ (2**np.arange(run_len)), self.combinatorial_encoder_table.sum(axis=1)))]

        # Pruning (sort by the number of 1 and keep the n ccombinatorial_filterombinaison_size)
        if is_energy == True:
            row_sums = np.sum(self.combinatorial_encoder_table, axis=1)
            sorted_indices = np.lexsort((-np.arange(len(self.combinatorial_encoder_table)), -row_sums))[::-1]
            indices_to_keep = sorted_indices[:combinaison_size]
            self.combinatorial_encoder_table = self.combinatorial_encoder_table[indices_to_keep]

        if is_binary == True:
            self.combinatorial_encoder_table = self.combinatorial_encoder_table[:combinaison_size]

        # Repeat (if possible and asked)
        if combinatorial_factor > 1:
            self.combinatorial_encoder_table = np.repeat(self.combinatorial_encoder_table, combinatorial_factor, axis=1)

        # Sort (by the first 1 index)
        self.combinatorial_encoder_table = np.array(sorted(self.combinatorial_encoder_table, key=self.find_first_one_index, reverse=True))

        # Roll (if possible and asked)
        if combinatorial_roll == True:
            first_one_index = np.argmax(np.any(self.combinatorial_encoder_table==1, axis=0))
            self.combinatorial_encoder_table = np.roll(self.combinatorial_encoder_table, -first_one_index)

        # # Nothing
        # self.combinatorial_encoder_table = np.repeat(self.combinatorial_encoder_table, combinatorial_factor, axis=1)[:combinaison_size]

        # print("combinatorial_encoder\n", self.combinatorial_encoder_table, "shape", np.shape(self.combinatorial_encoder_table), "nb_one", np.sum(self.combinatorial_encoder_table))
        if is_number_ones == True:
            self.combinatorial_encoder_table = np.array(sorted(self.combinatorial_encoder_table,key=lambda row: (np.sum(row), *(-row[::-1])), reverse=False), dtype=np.float32)[:combinaison_size]


        self.combinatorial_encoder_table_view = self.combinatorial_encoder_table
        self.is_combinatorial_table_init = True
        # print("combinatorial_encoder\n", self.combinatorial_encoder_table, "shape", np.shape(self.combinatorial_encoder_table), "nb_one", np.sum(self.combinatorial_encoder_table))

        # Modulo filter
        if is_modulo_static == True:
            combinaison_size = combinaison_size if combinaison_size < <int>len(self.combinatorial_encoder_table) else len(self.combinatorial_encoder_table)
            modulo_filter = (<float>combinatorial_encoder_len/<float>combinaison_size) 
            combinaison_size = <int>np.rint(combinatorial_encoder_len/modulo_filter)
            # print("(combinatorial_encoder_len//modulo_filter)", combinaison_size)
            combinatorial_table_modulo = np.empty((combinaison_size, run_len), dtype=np.float32)
            for i in range(combinaison_size):
                # print("i*modulo_filter", <int>np.rint(i*modulo_filter))
                combinatorial_table_modulo[i] = self.combinatorial_encoder_table[<int>np.rint(i*modulo_filter)]

            # print("\ncombinatorial_encoder_modulo\n", combinatorial_table_modulo, "shape", np.shape(combinatorial_table_modulo), "nb_one", np.sum(combinatorial_table_modulo))
            self.combinatorial_encoder_table     = combinatorial_table_modulo
            self.combinatorial_encoder_table_view = combinatorial_table_modulo
            # # Dynamic Modudo -> Pour connaitre l'index -> : floor(index/modulo_filter) * modulo_filter ou index - index % modulo_filter pour donner le bon index pour sous-tableau
            # # Dynamic Modudo -> Pour calculer le modulo -> par rapport à la taille du tableau : intableau - np.rint(mu / sigma_init * intableau) % intableau ou np.rint(mu / sigma_init * intableau) % intableau
            # # Dynamic Modudo -> Pour calculer le modulu_2 -> par rapport à la taille du tableau: size - ((mu%1)**(1 - np.exp(-10/size)) / 1.0 * size) % size or size - ((1-mu%1)**(1 - np.exp(-10/size)) / 1.0 * size) % size
            # # pensez à ajouter un clip pour éviter les érreur par exemple le modulo doit être entre 1 et intableau - 1, et le mu doit être inférieur à sigma_init et supérieur à 0 (non negatif)

        if is_random == True:
            np.random.shuffle(self.combinatorial_encoder_table)
            self.combinatorial_encoder_table_view = self.combinatorial_encoder_table

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef np.ndarray[np.float32_t, ndim=4] derivative_encoder(self, np.ndarray data_raw, float threshold=0.1, bint is_latency=True, bint is_latency_positional=True, bint use_prev_input=True, float max_delta_latency=1.0):
        if self.is_RL == False: exit("Not implemented yet in SL mode")
        # print("use_prev_input", use_prev_input)
        # print("self.prev_inputs_data", np.array(self.prev_inputs_data), "shape", np.shape(self.prev_inputs_data))
        # print("self.prev_inputs_spike", np.array(self.prev_inputs_spike), "shape", np.shape(self.prev_inputs_spike))

        if self.prev_inputs_data == None: 
            # self.prev_inputs_data = np.zeros(np.shape(data_raw), dtype=np.float32)
            self.prev_inputs_data  = np.zeros(np.shape(data_raw), dtype=np.float32) + 0.50 # for test !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if use_prev_input == True:
                self.prev_inputs_spike = np.zeros([data_raw.shape[0], data_raw.shape[1], data_raw.shape[2], <int>self.run_time_margin], dtype=np.float32)

        cdef int run_time = self.run_time
        cdef float[:,:,:] inputs_data_view = data_raw.astype(np.float32)
        cdef np.ndarray[np.float32_t, ndim=4] data_encoded = np.zeros([data_raw.shape[0], data_raw.shape[1], data_raw.shape[2], <int>self.run_time_margin], dtype=np.float32)

        cdef int nb_episodes = data_raw.shape[0]
        cdef int pop_size    = data_raw.shape[1]
        cdef int nb_inputs   = data_raw.shape[2]
        cdef int i, j, k, m

        cdef float delta, abs_delta, ratio, latency_pos_round
        cdef float value = 0.0

        for i in range(nb_episodes):
            for j in range(pop_size):
                for k in range(nb_inputs):
                    delta = inputs_data_view[i,j,k] - self.prev_inputs_data[i,j,k]

                    # Positional DELTA VERSION
                    if is_latency == True and is_latency_positional == True: # latency (only one spike)
                        abs_delta = math.fabsf(delta)

                        if abs_delta <= threshold:
                            if use_prev_input == True:
                                for m in range(self.run_time): # add padding by using run_time_original
                                    data_encoded[i, j, k, m] = self.prev_inputs_spike[i, j, k, m] # keep the state previous value
                            continue

                        value = 1.0 if delta > 0.0 else -1.0

                        ratio = (abs_delta - threshold) / (max_delta_latency - threshold)
                        if ratio > 1.0: ratio = 1.0
                        latency_pos  = <int>(math.rintf((1.0 - ratio) * (run_time - 1)))

                        # print("latency_pos", latency_pos, "ratio", ratio, "delta", delta, "abs_delta", abs_delta, "data_raw[i,j,k]", data_raw[i, j, k], "self.prev_inputs_data[i,j,k]", self.prev_inputs_data[i, j, k], "value", value)
                        data_encoded[i, j, k, latency_pos] = value
                        self.prev_inputs_data[i, j, k] = inputs_data_view[i, j, k] # update the previous value

                    else:
                        # Classical DELTA VERSION
                        if   delta > threshold:
                            value = 1.0
                            self.prev_inputs_data[i,j,k] = inputs_data_view[i,j,k] # update the previous value

                        elif delta < -threshold:
                            value = -1.0
                            self.prev_inputs_data[i,j,k] = inputs_data_view[i,j,k] # update the previous value

                        else:
                            if use_prev_input == True:
                                value = self.prev_inputs_spike[i, j, k, 0] # keep the state previous value (0 index because either for the rate or latency the data is the same)
                            else:
                                value = 0.0

                        # Latency (only one spike)
                        if is_latency == True and is_latency_positional == False:
                            data_encoded[i,j,k,0] = value

                        else: # rate (repeat the same value for the run_time)
                            for m in range(self.run_time): # add padding by using run_time_original
                                data_encoded[i,j,k,m] = value
        if use_prev_input == True:
            self.prev_inputs_spike = np.copy(data_encoded) # keep the state previous value

        return data_encoded


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float32_t, ndim=4] latency_encoder(self, np.ndarray data_raw):
        if self.is_RL == False: data_raw = data_raw[:, np.newaxis, :]  # add a new axis for the population dimension

        cdef int episode_batch = data_raw.shape[0]
        cdef int nb_population = data_raw.shape[1]
        cdef int nb_inputs     = data_raw.shape[2]
        cdef int i, m, n, t
        cdef float v, scale

        cdef np.ndarray[np.float32_t, ndim=4] data_encoded = np.zeros((episode_batch, nb_population, nb_inputs, self.run_time), dtype=np.float32)
        cdef float[:, :, :] data_raw_view = data_raw
        cdef float[:, :, :, :] data_encoded_view = data_encoded


        scale = <float>(self.run_time - 1)

        for i in range(episode_batch):
            for m in range(nb_population):
                for n in range(nb_inputs):
                    v = data_raw_view[i, m, n]
                    if v > 0.0:
                        t = <int>( (1.0 - v) * scale + 0.5 )  # round

                        if t >= self.run_time: t = self.run_time - 1
                        elif t < 0:t = 0

                        data_encoded_view[i, m, n, t] = 1.0

        return data_encoded


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float32_t, ndim=4] burst_encoder(self, np.ndarray data_raw):
        cdef int input_data_len = data_raw[0].shape[0]
        cdef np.ndarray[np.float32_t, ndim=4] data_encoded = np.zeros([self.batch_features, 1, input_data_len, <int>self.run_time_margin], dtype=np.float32)
        cdef int i, j, k, nb_spikes, shift
        cdef int isi
 
        for i in range(self.batch_features):
            for j in range(input_data_len):
                nb_spikes = <int>math.ceil(data_raw[i, j]) # number of spikes
                # isi = <int>math.ceil(-((<float>self.run_time_orginal)*(<float>data_raw[i, j])) + ((<float>self.run_time_orginal))) # inter spike interval
                # print("nb_spikes", nb_spikes, "data_raw[i, j]", data_raw[i, j], "run_time_orginal", self.run_time_orginal, "max_nb_spikes", max_nb_spikes)
                shift = 0
                for k in range(nb_spikes):
                    if shift >= self.run_time: break
                    data_encoded[i, 0, j, k] = 1.0 # set the spike value
                    # shift += isi
                # self.input_data[i, 0, j, shift_min:shift_max] = spike_amplitude
                # print("self.input_data[i, 0, j, shift_min:shift_max]", np.array(self.input_data[i, 0, j]))
                # exit()
        return data_encoded



    # INIT AND CACHED FUNCTIONS

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef void raw_encoder_init(self):
        self.encoder_type = EncoderType.RAW
    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef void latency_encoder_init(self):
        self.encoder_type = EncoderType.LATENCY
    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef void rate_encoder_init(self):
        self.encoder_type = EncoderType.RATE


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef void poisson_encoder_init(self, int rate, int max_nb_spikes = 3):
        self.max_nb_spikes = max_nb_spikes
        self.rate = rate
        self.encoder_type = EncoderType.POISSON


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef void exact_encoder_init(self, int max_nb_spikes = 3):
        self.max_nb_spikes = max_nb_spikes
        self.encoder_type = EncoderType.EXACT

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef void direct_encoder_init(self, float direct_min = -100_000, float direct_max = 100_000):
        self.direct_min = direct_min
        self.direct_max = direct_max
        self.encoder_type = EncoderType.DIRECT
    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef void binomial_encoder_init(self, int max_nb_spikes = 3, int reduce_noise = 100):
        self.max_nb_spikes = max_nb_spikes
        self.reduce_noise = reduce_noise
        self.encoder_type = EncoderType.BINOMIAL


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef void combinatorial_encoder_init(self, int combinatorial_factor = 1, int combinaison_size=1, int combinaison_size_max=1, bint combinatorial_roll=True, str combinatorial_filter = "energy", bint combinatorial_print_table_debug = False):
        self.combinatorial_factor = combinatorial_factor
        self.combinaison_size = combinaison_size
        self.combinaison_size_max = combinaison_size_max
        self.combinatorial_roll = combinatorial_roll
        self.combinatorial_filter = combinatorial_filter
        self.combinatorial_print_table_debug = combinatorial_print_table_debug
        self.encoder_type = EncoderType.COMBINATORIAL
        # self.combinaison_noise = combinaison_noise
        self.combinatorial_modulo = None


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef void derivative_encoder_init(self, float threshold=0.1, bint is_latency=True, bint is_latency_positional=True, bint use_prev_input=True, float max_delta_latency=1.0):
        self.derivative_threshold = threshold
        self.derivative_is_latency = is_latency
        self.derivative_is_latency_positional = is_latency_positional
        self.derivative_use_prev_input = use_prev_input
        self.derivative_max_delta_latency = max_delta_latency
        self.encoder_type = EncoderType.DERIVATIVE


    # TOOLS

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef int find_first_one_index(self, float[:] row):
        cdef int i, row_len
        row_len = len(row)
        for i in range(row_len):
            if row[i] != 0:
                return i
        return row_len

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[np.float32_t, ndim=4] add_padding_input_data(self,  np.ndarray[np.float32_t, ndim=4] data, int run_time_original, int run_time_margin):
        # print("1 before padding - self.data\n", np.array(self.data[:2]), "shape", np.shape(self.data))
        if run_time_margin > run_time_original:
            data = np.pad(data, ((0, 0), (0, 0), (0, 0), (0, <int>(run_time_margin - run_time_original))), 'constant')
        # print("2 after padding  - self.data\n", np.array(self.data[:2]), "shape", np.shape(self.data))
        return data


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int[:] n_most_frequent_np(self, np.ndarray[np.int32_t, ndim=1] arr, int n):
        cdef np.ndarray unique, counts, indices
        unique, counts = np.unique(arr, return_counts=True)
        indices = np.argsort(counts)[::-1][:n]
        return unique[indices].astype(np.int32)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int[:] n_most_frequent_np_2(self, np.ndarray[np.int32_t, ndim=1] arr, int n, int hack):
        cdef int i, arr_len = <int>arr.shape[0]
        cdef dict counts = {}
        cdef int[:] result
        cdef list top_n
        cdef int key
        cdef object value

        # Count the frequency of each element in arr
        for i in range(arr_len):
            key = arr[i]
            value = counts.get(key)
            if value is None:
                counts[key] = 1
            else:
                counts[key] = value + 1

        # Get the n most frequent elements
        top_n = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
        result = np.empty(len(top_n), dtype=np.int32)

        for i in range(len(top_n)):
            result[i] = top_n[i][0] - hack

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int[:] n_most_frequent_np_3(self, np.ndarray[np.int32_t, ndim=1] arr, int n):
        cdef np.ndarray[np.int32_t, ndim=1] counts = np.bincount(arr)
        cdef np.ndarray[np.int32_t, ndim=1] indices = np.argsort(counts)[::-1][:n]
        return indices.astype(np.int32)
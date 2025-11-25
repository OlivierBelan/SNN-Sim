
import numpy as np
cimport numpy as np
np.import_array()


cdef enum EncoderType:
    RAW = 1
    LATENCY = 2
    RATE = 3
    POISSON = 4
    BINOMIAL = 5
    EXACT = 6
    COMBINATORIAL = 7
    DERIVATIVE = 8
    DIRECT = 9

cdef class Encoder:
    

    cpdef void init_RL(self, int run_time, int run_time_margin)
    cpdef void init_SL(self, int run_time, int run_time_margin)

    cdef float[:,:,:] prev_inputs_data
    cdef float[:,:,:,:] prev_inputs_spike
    cdef bint is_RL


    cdef int run_time
    cdef int run_time_margin
    cdef int batch_features

    cdef np.ndarray combinatorial_encoder_table
    cdef float[:,:] combinatorial_encoder_table_view
    cdef bint is_combinatorial_table_init


    cdef EncoderType encoder_type
    cdef np.ndarray[np.float32_t, ndim=4] encode(self, np.ndarray data_raw, bint is_encoded =*)


    # Poisson Encoder
    cdef int rate
    cdef int max_nb_spikes
    cdef int reduce_noise
    cdef float direct_min
    cdef float direct_max

    cpdef void raw_encoder_init(self)
    cpdef void latency_encoder_init(self)
    cpdef void rate_encoder_init(self)

    
    cpdef np.ndarray[np.float32_t, ndim=4] raw_encoder(self, np.ndarray data_raw)
    cpdef np.ndarray[np.float32_t, ndim=4] latency_encoder(self, np.ndarray data_raw)
    cpdef np.ndarray[np.float32_t, ndim=4] rate_encoder(self, np.ndarray data_raw)


    cpdef void poisson_encoder_init(self, int rate, int max_nb_spikes = *)
    cpdef np.ndarray[np.float32_t, ndim=4] poisson_encoder(self, np.ndarray data_raw, int rate, int max_nb_spikes = *)
    cpdef np.ndarray[np.float32_t, ndim=4] poisson_encoder_3(self, np.ndarray data_raw, int rate, int max_nb_spikes = *)
    cpdef np.ndarray[np.float32_t, ndim=4] poisson_encoder_2(self, np.ndarray data_raw, int rate, int max_nb_spikes = *)

    cpdef void binomial_encoder_init(self, int max_nb_spikes = *, int reduce_noise = *)
    cpdef np.ndarray[np.float32_t, ndim=4] binomial_encoder(self, np.ndarray data_raw, int max_nb_spikes = *, int reduce_noise = *)


    cpdef void exact_encoder_init(self, int max_nb_spikes = *)
    cpdef np.ndarray[np.float32_t, ndim=4] exact_encoder(self, np.ndarray data_raw, int max_nb_spikes = *)

    cpdef void direct_encoder_init(self, float direct_min = *, float direct_max = *)
    cpdef np.ndarray[np.float32_t, ndim=4] direct_encoder(self, np.ndarray data_raw, float direct_min = *, float direct_max = *)


    cpdef np.ndarray[np.float32_t, ndim=4] burst_encoder(self, np.ndarray inputs_data)
    
    # Combinatorial Encoder
    cdef int combinatorial_factor
    cdef int combinaison_size
    cdef int combinaison_size_max
    cdef bint combinatorial_roll
    cdef str combinatorial_filter
    cdef public float combinaison_noise
    cdef public float[:] combinatorial_modulo
    cdef bint combinatorial_print_table_debug

    cpdef void combinatorial_encoder_init(self, int combinatorial_factor = *, int combinaison_size=*, int combinaison_size_max=*, bint combinatorial_roll=*, str combinatorial_filter = *, bint combinatorial_print_table_debug = *)
    cpdef np.ndarray[np.float32_t, ndim=4] combinatorial_encoder(self, np.ndarray data_raw, int combinatorial_factor = *, int combinaison_size=*, int combinaison_size_max=*, float combinaison_noise=*, bint combinatorial_roll=*, str combinatorial_filter = *, float[:] combinatorial_modulo = *, bint combinatorial_print_table_debug = *)
    cdef void combinatorial_encoder_build_table(self, int combinatorial_factor=*, int combinaison_size=*, int combinaison_size_max=*, bint combinatorial_roll=*, str combinatorial_filter = *)

    # Derivative Encoder
    cdef float derivative_threshold
    cdef bint  derivative_is_latency
    cdef bint  derivative_is_latency_positional
    cdef bint  derivative_use_prev_input
    cdef float derivative_max_delta_latency

    cpdef void derivative_encoder_init(self, float threshold=*, bint is_latency=*, bint is_latency_positional=*, bint use_prev_input=*, float max_delta_latency=*)
    cpdef np.ndarray[np.float32_t, ndim=4] derivative_encoder(self, np.ndarray data_raw, float threshold=*, bint is_latency=*, bint is_latency_positional=*, bint use_prev_input=*, float max_delta_latency=*)


    cdef int find_first_one_index(self, float[:] row)
    cdef np.ndarray[np.float32_t, ndim=4] add_padding_input_data(self,  np.ndarray[np.float32_t, ndim=4] input_data, int run_time_original, int run_time_margin)
    cdef int[:] n_most_frequent_np(self, np.ndarray[np.int32_t, ndim=1] arr, int n)
    cdef int[:] n_most_frequent_np_2(self, np.ndarray[np.int32_t, ndim=1] arr, int n, int hack)
    cdef int[:] n_most_frequent_np_3(self, np.ndarray[np.int32_t, ndim=1] arr, int n)
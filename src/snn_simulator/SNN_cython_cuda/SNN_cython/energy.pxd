import numpy as np
cimport numpy as np
np.import_array()
from SNN_cython_cuda.SNN_cython.snn_cython cimport SNN_cython_population
from .augmented cimport Augmented

cdef class Energy:

    cpdef void init_param(self, str energy_update_method = *, str energy_norm = *, int energy_length = *, bint is_energy_battery = *, bint energy_is_interp = *, float energy_interp_min = *, float energy_interp_max = *, bint energy_keep_sign = *, int energy_decimal = *)
    cdef void init_network_run(self, SNN_cython_population population, np.ndarray[np.int32_t, ndim=1] pop_idx, float[:,:,:] weight_view, int run_time, int run_time_margin, int nb_episodes, int nb_networks, int nb_neurons, int nb_inputs)
    cdef void init_step(self, SNN_cython_population population, np.ndarray[np.int32_t, ndim=1] pop_idx, int first_time_step, bint is_online)
    cdef void update(self, int current_time, float[:,:,:,:] spike_state, Augmented augmented=*, bint is_augmented=*)

    cdef np.ndarray energy
    cdef np.ndarray energy_battery
    cdef np.ndarray energy_index
    cdef float[:, :, :, :] energy_view
    cdef float[:, :, :] energy_battery_view
    cdef float[:,:,: ] weight_acceleration_record
    cdef int[:, :, :] energy_index_view

    cdef bint is_energy
    cdef str energy_norm
    cdef int energy_length
    cdef bint is_energy_battery
    cdef int energy_update_method
    cdef bint energy_keep_sign
    cdef float energy_interp_min
    cdef float energy_interp_max
    cdef bint energy_is_interp
    cdef int energy_decimal

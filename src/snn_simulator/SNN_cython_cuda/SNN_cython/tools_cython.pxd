#!python
# cython: embedsignature=True, binding=True


cimport cython
import numpy as np
cimport numpy as np
np.import_array()

cdef double get_time()

# from snn_cython cimport SNN_cython as SNN
# cpdef SNN generate_snn(size_t nb_input = *, size_t nb_hidden = *, size_t nb_output=*)


cdef float[:,:,:] norm_min_max_all(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, int round_decimal = *)
cdef float[:,:,:] norm_min_max_rows(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, int round_decimal = *)
cdef float[:,:,:] norm_min_max_columns(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, int round_decimal = *)

cdef float[:,:,:] norm_L1_all(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint is_abs=*, int round_decimal = *)
cdef float[:,:,:] norm_L1_rows(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint is_abs=*, int round_decimal = *)
cdef float[:,:,:] norm_L1_columns(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint is_abs=*, int round_decimal = *)

cdef float[:,:,:] norm_L1_sum_all(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint keep_sign=*, int round_decimal = *)
cdef float[:,:,:] norm_L1_sum_rows(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint keep_sign=*, int round_decimal = *)
cdef float[:,:,:] norm_L1_sum_columns(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint keep_sign=*, int round_decimal = *)


cdef float[:,:,:] norm_L2_all(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, int round_decimal = *)
cdef float[:,:,:] norm_L2_rows(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, int round_decimal = *)
cdef float[:,:,:] norm_L2_columns(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, int round_decimal = *)

cdef float[:,:,:] norm_L2_sum_all(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint keep_sign=*, int round_decimal = *)
cdef float[:,:,:] norm_L2_sum_rows(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint keep_sign=*, int round_decimal = *)
cdef float[:,:,:] norm_L2_sum_columns(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint keep_sign=*, int round_decimal = *)



cdef float L2_norm_array(float[:] arr)
cdef float L2_norm_matrix(float[:,:] arr)
cdef float sum_array_float(float[:] arr, bint is_abs=*)
cdef float sum_matrix_float(float[:, :] arr, bint is_abs=*)
cdef float[:] min_max_array_float(float[:] arr)
cdef float[:] min_max_matrix_float(float[:, :] arr)
cdef float interpolation(float value, float min_input, float max_input, float min_output, float max_output)


cdef np.ndarray norm_min_max_all_numpy(np.ndarray[np.float32_t, ndim=3] matrix)
cdef np.ndarray norm_min_max_rows_numpy(np.ndarray matrix)
cdef np.ndarray norm_min_max_columns_numpy(np.ndarray matrix)

cdef np.ndarray norm_L1_all_numpy(np.ndarray matrix)
cdef np.ndarray norm_L1_rows_numpy(np.ndarray matrix)
cdef np.ndarray norm_L1_columns_numpy(np.ndarray matrix)

cdef np.ndarray norm_L2_all_numpy(np.ndarray matrix)
cdef np.ndarray norm_L2_rows_numpy(np.ndarray matrix)
cdef np.ndarray norm_L2_columns_numpy(np.ndarray matrix)

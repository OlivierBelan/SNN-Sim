cimport cython
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
cimport libc.math as math

import numpy as np
cimport numpy as np
np.import_array()

cdef double get_time():
    cdef timespec ts
    clock_gettime(CLOCK_REALTIME, &ts)
    return ts.tv_sec + (ts.tv_nsec / 1000000000.)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:,:,:] norm_min_max_all(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, int round_decimal = -1):
    cdef size_t i, j, k, len_dim1, len_dim2, len_dim3
    cdef float[:] min_max_val
    cdef float value
    len_dim1 = matrix.shape[0]
    len_dim2 = matrix.shape[1]
    len_dim3 = matrix.shape[2]
    for i in range(len_dim1):
        min_max_val = min_max_matrix_float(matrix[i])
        for j in range(len_dim2):
            for k in range(len_dim3):
                value = matrix[i, j, k]
                if min_max_val[0] != min_max_val[1] and value != 0.0:
                    value = (value - min_max_val[0]) / (min_max_val[1] - min_max_val[0])

                    if is_interp:
                        matrix[i, j, k] = interpolation(value, 0, 1, interp_min, interp_max)
                    else:
                        matrix[i, j, k] = value

                    if round_decimal >= 0:
                        matrix[i, j, k] = round_float(matrix[i, j, k], round_decimal)

    return matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:,:,:] norm_min_max_rows(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, int round_decimal = -1):
    cdef size_t i, j, k, len_dim1, len_dim2, len_dim3
    cdef float[:] min_max_val
    cdef float value
    len_dim1 = matrix.shape[0]
    len_dim2 = matrix.shape[1]
    len_dim3 = matrix.shape[2]
    for i in range(len_dim1):
        for j in range(len_dim2):
            min_max_val = min_max_array_float(matrix[i, j]) # send the row and min_max_val[0] = min_val, min_max_val[1] = max_val
            for k in range(len_dim3):
                value = matrix[i, j, k]
                if min_max_val[0] != min_max_val[1] and value != 0.0:
                    value = (value - min_max_val[0]) / (min_max_val[1] - min_max_val[0])

                    if is_interp:
                        matrix[i, j, k] = interpolation(value, 0, 1, interp_min, interp_max)
                    else:
                        matrix[i, j, k] = value

                    if round_decimal >= -1:
                        matrix[i, j, k] = round_float(matrix[i, j, k], round_decimal)

    return matrix


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:,:,:] norm_min_max_columns(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, int round_decimal = -1):
    cdef size_t i, j, k, len_dim1, len_dim2, len_dim3
    cdef float[:] min_max_val
    cdef float value
    len_dim1 = matrix.shape[0]
    len_dim2 = matrix.shape[1]
    len_dim3 = matrix.shape[2]

    for i in range(len_dim1):
        for j in range(len_dim3):
            min_max_val = min_max_array_float(matrix[i, :, j]) # send the column and min_max_val[0] = min_val, min_max_val[1] = max_val
            for k in range(len_dim2):
                value = matrix[i, k, j]
                if min_max_val[0] != min_max_val[1] and value != 0.0:
                    value = (value - min_max_val[0]) / (min_max_val[1] - min_max_val[0])

                    if is_interp:
                        matrix[i, k, j] = interpolation(value, 0, 1, interp_min, interp_max)
                    else:
                        matrix[i, k, j] = value

                    if round_decimal >= 0:
                        matrix[i, j, k] = round_float(matrix[i, j, k], round_decimal)

    return matrix




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:,:,:] norm_L1_all(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint is_abs=True, int round_decimal = -1):
    cdef size_t i, j, k, len_dim1, len_dim2, len_dim3
    cdef float total_sums, value
    len_dim1 = matrix.shape[0]
    len_dim2 = matrix.shape[1]
    len_dim3 = matrix.shape[2]
    for i in range(len_dim1):
        total_sums = sum_matrix_float(matrix[i], is_abs=is_abs)
        for j in range(len_dim2):
            for k in range(len_dim3):
                value = matrix[i, j, k]
                if total_sums != 0 and value != 0.0:
                    value = value / total_sums

                    if is_interp:
                        matrix[i, j, k] = interpolation(value, -1, 1, interp_min, interp_max)
                    else:
                        matrix[i, j, k] = value

                    if round_decimal >= 0:
                        matrix[i, j, k] = round_float(matrix[i, j, k], round_decimal)

    return matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:,:,:] norm_L1_rows(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint is_abs=True, int round_decimal = -1):
    cdef size_t i, j, k, len_dim1, len_dim2, len_dim3
    cdef float row_sums, value
    len_dim1 = matrix.shape[0]
    len_dim2 = matrix.shape[1]
    len_dim3 = matrix.shape[2]
    for i in range(len_dim1):
        for j in range(len_dim2):
            row_sums = sum_array_float(matrix[i, j], is_abs=is_abs)
            for k in range(len_dim3):
                value = matrix[i, j, k]
                if row_sums != 0 and value != 0.0:
                    value = value / row_sums

                    if is_interp:
                        matrix[i, j, k] = interpolation(value, -1, 1, interp_min, interp_max)
                    else:
                        matrix[i, j, k] = value

                    if round_decimal >= 0:
                        matrix[i, j, k] = round_float(matrix[i, j, k], round_decimal)

    return matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:,:,:] norm_L1_columns(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint is_abs=True, int round_decimal = -1):
    cdef size_t i, j, k, len_dim1, len_dim2, len_dim3
    cdef float column_sums, value
    len_dim1 = matrix.shape[0]
    len_dim2 = matrix.shape[1]
    len_dim3 = matrix.shape[2]
    for i in range(len_dim1):
        for j in range(len_dim3):
            column_sums = sum_array_float(matrix[i, :, j], is_abs=is_abs)
            for k in range(len_dim2):
                value = matrix[i, k, j]
                if column_sums != 0 and value != 0.0:
                    value = value / column_sums

                    if is_interp:
                        matrix[i, k, j] = interpolation(value, -1, 1, interp_min, interp_max)
                    else:
                        matrix[i, k, j] = value

                    if round_decimal >= 0:
                        matrix[i, j, k] = round_float(matrix[i, j, k], round_decimal)

    return matrix




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:,:,:] norm_L1_sum_all(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint keep_sign=False, int round_decimal = -1):
    norm_L1_all(matrix, True, 0, 1, is_abs=True) # Normalize the matrix according to the L1 norm
    cdef size_t i, j, k, len_dim1, len_dim2, len_dim3
    cdef float total_sums, value
    len_dim1 = matrix.shape[0]
    len_dim2 = matrix.shape[1]
    len_dim3 = matrix.shape[2]
    for i in range(len_dim1):
        total_sums = sum_matrix_float(matrix[i], is_abs=False)
        for j in range(len_dim2):
            for k in range(len_dim3):
                value = matrix[i, j, k]
                if total_sums != 0 and value != 0.0:
                    if keep_sign == False:
                            matrix[i, j, k] = (value / total_sums)

                    else: # In case we want to keep the orignal sign (interpolation is possible again)
                        if value > 0.5: # greater than 0.5 means the original value was positive
                            value = (value / total_sums)
                        else: # less than 0.5 means the original value was negative
                            value = -(value / total_sums)

                        if is_interp:
                            matrix[i, j, k] = interpolation(value, -1, 1, interp_min, interp_max)                  
                        else:
                            matrix[i, j, k] = value

                    if round_decimal >= 0:
                        matrix[i, j, k] = round_float(matrix[i, j, k], round_decimal)

                    
    return matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:,:,:] norm_L1_sum_rows(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint keep_sign=False, int round_decimal = -1):
    norm_L1_rows(matrix, True, 0, 1, is_abs=True) # Normalize the matrix according to the L1 norm
    cdef float row_sums, value
    len_dim1 = matrix.shape[0]
    len_dim2 = matrix.shape[1]
    len_dim3 = matrix.shape[2]
    for i in range(len_dim1):
        for j in range(len_dim2):
            row_sums = sum_array_float(matrix[i, j], is_abs=False)
            for k in range(len_dim3):
                value = matrix[i, j, k]
                
                if row_sums != 0 and value != 0.0:
                    if keep_sign == False:
                            matrix[i, j, k] = (value / row_sums)

                    else: # In case we want to keep the orignal sign (interpolation is possible again)
                        if value > 0.5: # greater than 0.5 means the original value was positive
                            value = (value / row_sums)
                        else: # less than 0.5 means the original value was negative
                            value = -(value / row_sums)

                        if is_interp:
                            matrix[i, j, k] = interpolation(value, -1, 1, interp_min, interp_max)                  
                        else:
                            matrix[i, j, k] = value

                    if round_decimal >= 0:
                        matrix[i, j, k] = round_float(matrix[i, j, k], round_decimal)

    return matrix


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:,:,:] norm_L1_sum_columns(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint keep_sign=False, int round_decimal = -1):
    norm_L1_columns(matrix, True, 0, 1, is_abs=True) # Normalize the matrix according to the L1 norm
    cdef size_t i, j, k, len_dim1, len_dim2, len_dim3
    cdef float column_sums, value
    len_dim1 = matrix.shape[0]
    len_dim2 = matrix.shape[1]
    len_dim3 = matrix.shape[2]
    for i in range(len_dim1):
        for j in range(len_dim3):
            column_sums = sum_array_float(matrix[i, :, j], is_abs=False)
            for k in range(len_dim2):
                value = matrix[i, k, j]
                if column_sums != 0 and value != 0.0:
                    if keep_sign == False:
                        matrix[i, k, j] = (value / column_sums)

                    else: # In case we want to keep the orignal sign (interpolation is possible again)
                        if value > 0.5: # greater than 0.5 means the original value was positive
                            value = (value / column_sums)
                        else: # less than 0.5 means the original value was negative
                            value = -(value / column_sums)

                        if is_interp:
                            matrix[i, k, j] = interpolation(value, -1, 1, interp_min, interp_max)                  
                        else:
                            matrix[i, k, j] = value

                    if round_decimal >= 0:
                        matrix[i, j, k] = round_float(matrix[i, j, k], round_decimal)

    return matrix


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:,:,:] norm_L2_all(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, int round_decimal = -1):
    cdef size_t i, j, k, len_dim1, len_dim2, len_dim3
    cdef float L2_norm, value
    len_dim1 = matrix.shape[0]
    len_dim2 = matrix.shape[1]
    len_dim3 = matrix.shape[2]
    for i in range(len_dim1):
        L2_norm = L2_norm_matrix(matrix[i])
        for j in range(len_dim2):
            for k in range(len_dim3):
                value = matrix[i, j, k]
                if L2_norm != 0 and value != 0.0:
                    value = value / L2_norm

                    if is_interp:
                        matrix[i, j, k] = interpolation(value, -1, 1, interp_min, interp_max)
                    else:
                        matrix[i, j, k] = value

                    if round_decimal >= 0:
                        matrix[i, j, k] = round_float(matrix[i, j, k], round_decimal)

    return matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:,:,:] norm_L2_rows(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, int round_decimal = -1):
    cdef size_t i, j, k, len_dim1, len_dim2, len_dim3
    cdef float L2_norm, value
    len_dim1 = matrix.shape[0]
    len_dim2 = matrix.shape[1]
    len_dim3 = matrix.shape[2]
    for i in range(len_dim1):
        for j in range(len_dim2):
            L2_norm = L2_norm_array(matrix[i, j])
            for k in range(len_dim3):
                value = matrix[i, j, k]
                if L2_norm != 0 and value != 0.0:
                    value = value / L2_norm

                    if is_interp:
                        matrix[i, j, k] = interpolation(value, -1, 1, interp_min, interp_max)
                    else:
                        matrix[i, j, k] = value

                    if round_decimal >= 0:
                        matrix[i, j, k] = round_float(matrix[i, j, k], round_decimal)

    return matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:,:,:] norm_L2_columns(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, int round_decimal = -1):
    cdef size_t i, j, k, len_dim1, len_dim2, len_dim3
    cdef float L2_norm, value
    len_dim1 = matrix.shape[0]
    len_dim2 = matrix.shape[1]
    len_dim3 = matrix.shape[2]
    for i in range(len_dim1):
        for j in range(len_dim3):
            L2_norm = L2_norm_array(matrix[i, :, j])
            for k in range(len_dim2):
                value = matrix[i, k, j]
                if L2_norm != 0 and value != 0.0:
                    value = value / L2_norm

                    if is_interp:
                        matrix[i, k, j] = interpolation(value, -1, 1, interp_min, interp_max)
                    else:
                        matrix[i, k, j] = value

                    if round_decimal >= 0:
                        matrix[i, j, k] = round_float(matrix[i, j, k], round_decimal)

    return matrix


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:,:,:] norm_L2_sum_all(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint keep_sign=False, int round_decimal = -1):
    norm_L2_all(matrix, True, 0, 1) # Normalize the matrix according to the L2 norm
    cdef size_t i, j, k, len_dim1, len_dim2, len_dim3
    cdef float total_sums, value
    len_dim1 = matrix.shape[0]
    len_dim2 = matrix.shape[1]
    len_dim3 = matrix.shape[2]
    for i in range(len_dim1):
        total_sums = sum_matrix_float(matrix[i], is_abs=False)
        for j in range(len_dim2):
            for k in range(len_dim3):
                value = matrix[i, j, k]
                if total_sums != 0 and value != 0.0:
                    if keep_sign == False:
                        matrix[i, j, k] = (value / total_sums)
                    
                    else: # In case we want to keep the orignal sign (interpolation is possible again)
                        if value > 0.5: # greater than 0.5 means the original value was positive
                            value = (value / total_sums)
                        else: # less than 0.5 means the original value was negative
                            value = -(value / total_sums)

                        if is_interp:
                            matrix[i, j, k] = interpolation(value, -1, 1, interp_min, interp_max)                  
                        else:
                            matrix[i, j, k] = value

                    if round_decimal >= 0:
                        matrix[i, j, k] = round_float(matrix[i, j, k], round_decimal)

    return matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:,:,:] norm_L2_sum_rows(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint keep_sign=False, int round_decimal = -1):
    norm_L2_rows(matrix, True, 0, 1) # Normalize the matrix according to the L2 norm
    cdef size_t i, j, k, len_dim1, len_dim2, len_dim3
    cdef float row_sums, value
    len_dim1 = matrix.shape[0]
    len_dim2 = matrix.shape[1]
    len_dim3 = matrix.shape[2]
    for i in range(len_dim1):
        for j in range(len_dim2):
            row_sums = sum_array_float(matrix[i, j], is_abs=False)
            for k in range(len_dim3):
                value = matrix[i, j, k]
                if row_sums != 0 and value != 0.0:
                    if keep_sign == False:
                        matrix[i, j, k] = (value / row_sums)

                    else: # In case we want to keep the orignal sign (interpolation is possible again)
                        if value > 0.5: # greater than 0.5 means the original value was positive
                            value = (value / row_sums)
                        else: # less than 0.5 means the original value was negative
                            value = -(value / row_sums)

                        if is_interp:
                            matrix[i, j, k] = interpolation(value, -1, 1, interp_min, interp_max)                  
                        else:
                            matrix[i, j, k] = value

                    if round_decimal >= 0:
                        matrix[i, j, k] = round_float(matrix[i, j, k], round_decimal)

    return matrix


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:,:,:] norm_L2_sum_columns(float[:,:,:] matrix, bint is_interp, float interp_min, float interp_max, bint keep_sign=False, int round_decimal = -1):
    norm_L2_columns(matrix, True, 0, 1) # Normalize the matrix according to the L2 norm
    cdef size_t i, j, k, len_dim1, len_dim2, len_dim3
    cdef float column_sums, value
    len_dim1 = matrix.shape[0]
    len_dim2 = matrix.shape[1]
    len_dim3 = matrix.shape[2]
    for i in range(len_dim1):
        for j in range(len_dim3):
            column_sums = sum_array_float(matrix[i, :, j], is_abs=False)
            for k in range(len_dim2):
                value = matrix[i, k, j]
                if column_sums != 0 and value != 0.0:
                    if keep_sign == False:
                        matrix[i, k, j] = (value / column_sums)

                    else: # In case we want to keep the orignal sign (interpolation is possible again)
                        if value > 0.5: # greater than 0.5 means the original value was positive
                            value = (value / column_sums)
                        else:
                            value = -(value / column_sums)

                        if is_interp:
                            matrix[i, k, j] = interpolation(value, -1, 1, interp_min, interp_max)                  
                        else:
                            matrix[i, k, j] = value

                    if round_decimal >= 0:
                        matrix[i, j, k] = round_float(matrix[i, j, k], round_decimal)

    return matrix


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float L2_norm_array(float[:] arr) noexcept:
    """L2 norm of an array (euclidean norm)."""
    cdef float norm = 0.0
    cdef size_t len_arr = arr.shape[0]
    cdef size_t i
    
    for i in range(len_arr): 
        norm += arr[i] * arr[i]
    
    return math.sqrt(norm)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float L2_norm_matrix(float[:,:] arr) noexcept:
    """L2 norm of a matrix (euclidean norm)."""
    cdef float norm = 0.0
    cdef size_t dim1 = arr.shape[0]
    cdef size_t dim2 = arr.shape[1]
    cdef size_t i, j
    
    for i in range(dim1):
        for j in range(dim2):
            norm += arr[i, j] * arr[i, j]
    
    return math.sqrt(norm)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float sum_matrix_float(float[:, :] arr, bint is_abs=False) noexcept:
    cdef float sum_val = 0.0
    cdef size_t dim1 = arr.shape[0]
    cdef size_t dim2 = arr.shape[1]
    cdef float value
    cdef size_t i, j
    for i in range(dim1):
        for j in range(dim2):
            value = arr[i, j]
            if value != 0.0:
                sum_val += math.fabsf(value)
    return sum_val

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float sum_array_float(float[:] arr, bint is_abs=False) noexcept:
    cdef float sum_val = 0.0
    cdef size_t len_arr = arr.shape[0]
    cdef size_t i
    for i in range(len_arr):
        sum_val += math.fabsf(arr[i])
    return sum_val

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:] min_max_matrix_float(float[:, :] arr) noexcept:
    cdef float[2] min_max_val = [+1e15, -1e15]
    cdef size_t dim1 = arr.shape[0]
    cdef size_t dim2 = arr.shape[1]
    cdef float value
    cdef size_t i, j
    for i in range(dim1):
        for j in range(dim2):
            value = arr[i, j]
            if value != 0.0:
                if value < min_max_val[0]: min_max_val[0] = value # min_val = value
                if value > min_max_val[1]: min_max_val[1] = value # max_val = value
    return min_max_val


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float[:] min_max_array_float(float[:] arr) noexcept:
    cdef float[2] min_max_val = [+1e15, -1e15]
    cdef size_t len_arr = arr.shape[0]
    cdef float value
    cdef size_t i
    for i in range(len_arr):
        value = arr[i]
        if value != 0.0:
            if value < min_max_val[0]: min_max_val[0] = value # min_val = value
            if value > min_max_val[1]: min_max_val[1] = value # max_val = value
    return min_max_val

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef float interpolation(float value, float min_input, float max_input, float min_output, float max_output) noexcept:
    """Interpolates a value from one range to another."""
    if min_input == max_input: return min_output # Avoid division by zero
    if value < min_input: value = min_input # Clip the value to the minimum
    if value > max_input: value = max_input # Clip the value to the maximum
    if min_input == min_output and max_input == max_output: return value # Avoid unnecessary calculations
    return min_output + (( (value - min_input) * (max_output - min_output) ) / (max_input - min_input)) # Interpolation formula





####### NUMPY IMPLEMENTATION (CAN BE SLOWER THAN THE CYTHON VERSION) #######

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef np.ndarray norm_min_max_all_numpy(np.ndarray[np.float32_t, ndim=3] matrix):
    """ Normalise les colonnes de la matrice entre 0 et 1. """
    cdef np.ndarray[np.float32_t, ndim=3] min_vals = np.min(matrix, axis=(1,2), keepdims=True)
    cdef np.ndarray[np.float32_t, ndim=3] max_vals = np.max(matrix, axis=(1,2), keepdims=True)
    # matrix = (matrix - min_vals) / (max_vals - min_vals)
    # print("ma")
    return (matrix - min_vals) / (max_vals - min_vals)
    # return np.interp(matrix,  [0, 1], energy_interval)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef np.ndarray norm_min_max_rows_numpy(np.ndarray[np.float32_t, ndim=3] matrix):
    """ Normalise les lignes de la matrice entre 0 et 1. """
    cdef np.ndarray[np.float32_t, ndim=3] min_vals = np.min(matrix, axis=2, keepdims=True)  # Minima le long des deux dernières dimensions
    cdef np.ndarray[np.float32_t, ndim=3] max_vals = np.max(matrix, axis=2, keepdims=True)  # Maxima le long des deux dernières dimensions
    # matrix = (matrix - min_vals) / (max_vals - min_vals)
    return (matrix - min_vals) / (max_vals - min_vals)
    # return np.interp(matrix,  [0, 1], energy_interval)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef np.ndarray norm_min_max_columns_numpy(np.ndarray[np.float32_t, ndim=3] matrix):
    """ Normalise les colonnes de la matrice entre 0 et 1. """
    cdef np.ndarray[np.float32_t, ndim=3] min_vals = np.min(matrix, axis=1, keepdims=True)
    cdef np.ndarray[np.float32_t, ndim=3] max_vals = np.max(matrix, axis=1, keepdims=True)
    # matrix = (matrix - min_vals) / (max_vals - min_vals)
    return (matrix - min_vals) / (max_vals - min_vals)
    # return np.interp(matrix,  [0, 1], [energy_interval_min, energy_interval_max])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef np.ndarray norm_L1_all_numpy(np.ndarray[np.float32_t, ndim=3] matrix):
    """Normalise l'ensemble de chaque sous-matrice dans la matrice 3D selon la norme L1."""
    cdef np.ndarray[np.float32_t, ndim=3] total_sums = np.sum(np.abs(matrix), axis=(1, 2), keepdims=True)
    return matrix / total_sums
    # return np.interp(matrix,  [-1, 1], [energy_interval_min, energy_interval_max])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef np.ndarray norm_L1_rows_numpy(np.ndarray[np.float32_t, ndim=3] matrix):
    """Normalise chaque ligne de chaque sous-matrice dans la matrice 3D selon la norme L1."""
    cdef np.ndarray[np.float32_t, ndim=3] row_sums = np.sum(np.abs(matrix), axis=2, keepdims=True)
    # matrix = matrix / row_sums
    # return np.interp(matrix,  [-1, 1], energy_interval_min, energy_interval_max)
    return matrix / row_sums


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef np.ndarray norm_L1_columns_numpy(np.ndarray[np.float32_t, ndim=3] matrix):
    """ Normalise les colonnes de la matrice de sorte que la somme de chaque colonne soit 1. """
    cdef np.ndarray[np.float32_t, ndim=3] col_sums = np.sum(np.abs(matrix), axis=1, keepdims=True)
    # matrix = matrix / col_sums
    # return np.interp(matrix,  [-1, 1], energy_interval_min, energy_interval_max)
    return matrix / col_sums


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef np.ndarray norm_L2_all_numpy(np.ndarray[np.float32_t, ndim=3] matrix):
    """Normalise l'ensemble de chaque sous-matrice dans la matrice 3D selon la norme L2."""
    cdef np.ndarray[np.float32_t, ndim=3] l2_norms = np.linalg.norm(matrix, axis=(1, 2), keepdims=True)  # Calcul des normes L2 pour chaque sous-matrice
    # matrix = matrix / l2_norms
    # return np.interp(matrix,  [-1, 1], energy_interval_min, energy_interval_max)
    return matrix / l2_norms

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef np.ndarray norm_L2_rows_numpy(np.ndarray[np.float32_t, ndim=3] matrix):
    """Normalise chaque ligne de chaque sous-matrice 3x3 dans la matrice 3D selon la norme L2."""
    cdef np.ndarray[np.float32_t, ndim=3] l2_norms = np.linalg.norm(matrix, axis=2, keepdims=True)  # Calcul des normes L2 pour chaque ligne
    # matrix = matrix / l2_norms
    # return np.interp(matrix,  [-1, 1], energy_interval_min, energy_interval_max)
    return matrix / l2_norms

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef np.ndarray norm_L2_columns_numpy(np.ndarray[np.float32_t, ndim=3] matrix):
    """Normalise chaque colonne de chaque sous-matrice dans la matrice 3D selon la norme L2."""
    cdef np.ndarray[np.float32_t, ndim=3] l2_norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # Calcul des normes L2 pour chaque colonne
    # matrix = matrix / l2_norms
    # return np.interp(matrix,  [-1, 1], energy_interval_min, energy_interval_max)
    return matrix / l2_norms


cdef float round_float(float value, int decimals):
    if decimals < 0:
        return value
    cdef pow_val = 10 ** decimals
    value = value * pow_val
    return math.roundf(value) / pow_val

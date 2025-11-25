# from cython.cimports.libcpp import bint

import cython
import numpy as np
cimport numpy as np
np.import_array()

cdef class SNN_cython:    
    # PULBIC VARIABLE
    cdef public int id

    # Neurons indexes
    cdef public np.ndarray input_indexes
    cdef public np.ndarray output_indexes
    cdef public np.ndarray hidden_indexes

    # Neurons and Synapses Population indexes active
    cdef public np.ndarray neuron_active_indexes
    # cdef public np.ndarray synapse_active_indexes

    # Neurons and Synapses Population indexes active
    # cdef public np.ndarray neuron_active_global_indexes
    # cdef public np.ndarray synapse_active_global_indexes
        
    #  PRIVATE VARIABLE
    # Neurons variables
    cdef public np.ndarray voltage_init
    cdef public np.ndarray tau
    cdef public np.ndarray current
    cdef public np.ndarray threshold
    cdef public np.ndarray refractory
    cdef public np.ndarray constant_current
    cdef public np.ndarray energy
    cdef public np.ndarray energy_battery
    # Synapses variables
    cdef public np.ndarray weight
    cdef public np.ndarray delay


    cpdef void init_network(self,
                            # 0 - NN General Parameters
                            dict parameter, 

                            # 1 - Neurons indexes
                            np.ndarray input_indexes, 
                            np.ndarray output_indexes, 
                            np.ndarray hidden_indexes, 
                            # np.ndarray hidden_indexes_active, 

                            # 2- Neurons and Synapses Population indexes active
                            np.ndarray neuron_active_global_indexes, 
                            # np.ndarray synapse_active_global_indexes, 

                            # 3- Neurons and Synapses Population indexes unactive
                            np.ndarray synapse_unactive_indexes, 
                            np.ndarray neuron_unactive_indexes,
                            
                            # 4 - Other Parameters that can optionnaly be used
                            bint is_delay,
                            bint is_refractory,
                            bint is_energy,
                            bint is_energy_battery
                            )


cdef class SNN_cython_population:
    cdef public int nb_population
    cdef public int nb_neurons
    cdef public np.ndarray population_genome_ids

    # Neurons indexes
    cdef public np.ndarray input_indexes
    cdef public np.ndarray output_indexes
    cdef public np.ndarray hidden_indexes

    # Neurons and Synapses Population indexes active
    cdef public np.ndarray neuron_active_indexes
    # cdef public np.ndarray synapse_active_indexes

    # Neurons and Synapses Population indexes active
    cdef public np.ndarray neuron_active_global_indexes
    cdef public np.ndarray synapse_active_global_indexes
        
    #  PRIVATE VARIABLE
    # Neurons variables
    cdef public np.ndarray voltage_init
    cdef public np.ndarray tau
    cdef public np.ndarray current
    cdef public np.ndarray threshold
    cdef public np.ndarray refractory
    cdef public np.ndarray constant_current
    cdef public np.ndarray energy
    cdef public np.ndarray energy_battery
    # Synapses variables
    cdef public np.ndarray weight
    cdef public np.ndarray delay

    cdef public np.ndarray weight_rows
    cdef public np.ndarray weight_cols


    cdef public bint is_delay
    cdef public bint is_refractory
    cdef public bint is_energy
    cdef public bint is_energy_battery
    cdef public bint is_dynamic_topology


    cpdef void init_network(self,
                            # 0 - NN General Parameters
                            dict parameter, 

                            np.ndarray population_genome_ids,

                            # 1 - Neurons indexes
                            np.ndarray input_indexes, 
                            np.ndarray output_indexes, 
                            np.ndarray hidden_indexes, 
                            # np.ndarray hidden_indexes_active, 

                            # 2- Neurons and Synapses Population indexes active
                            np.ndarray neuron_active_global_indexes, 
                            np.ndarray synapse_active_global_indexes,

                            # 3- Neurons and Synapses Population indexes unactive
                            # np.ndarray synapse_unactive_indexes, 
                            # np.ndarray neuron_unactive_indexes,
                            
                            # 4 - Other Parameters that can optionnaly be used
                            bint is_delay,
                            bint is_refractory,
                            bint is_energy,
                            bint is_energy_battery,
                            bint is_dynamic_topology,
                            bint is_disable_output_threshold
                            )

    cpdef void init_network_unactive_indexes(self, pop_idx, np.ndarray[np.int32_t, ndim=1] neuron_unactive_indexes, np.ndarray[np.int32_t, ndim=2] synapse_unactive_indexes)
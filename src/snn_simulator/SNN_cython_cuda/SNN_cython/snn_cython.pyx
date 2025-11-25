#!python
# cython: embedsignature=True, binding=True

import cython
import numpy as np
cimport numpy as np
np.import_array()


cdef class SNN_cython:    

    def __init__(self, int id = 0):
        # PULBIC VARIABLE
        self.id = id
        

    #  PUBLIC METHODE     
    cpdef void init_network(self, 
        # 0 - NN General Parameters
        dict parameter, 

        # 1 - Neurons indexes
        np.ndarray[np.int32_t, ndim=1] input_indexes, 
        np.ndarray[np.int32_t, ndim=1] output_indexes, 
        np.ndarray[np.int32_t, ndim=1] hidden_indexes, 

        # 2 - Neurons and Synapses Population indexes active
        np.ndarray[np.int32_t, ndim=1] neuron_active_global_indexes, 
        # np.ndarray[np.int32_t, ndim=2] synapse_active_global_indexes,

        # 3 - Neurons and Synapses Population indexes unactive
        np.ndarray[np.int32_t, ndim=2] synapse_unactive_indexes,
        np.ndarray[np.int32_t, ndim=1] neuron_unactive_indexes,

        # 4 - Other Parameters that can optionnaly be used
        bint is_delay,
        bint is_refractory,
        bint is_energy,
        bint is_energy_battery
        ):

        cdef np.ndarray flat_indices, rows, cols

        # 0 - Init info indexes
        self.input_indexes = input_indexes
        self.output_indexes = output_indexes
        self.hidden_indexes = hidden_indexes

        weight = parameter["weight"].copy()
        threshold = parameter["threshold"].copy()
        weight[synapse_unactive_indexes[0], synapse_unactive_indexes[1]] = 0.0 # set unactive synapses to 0.0
        threshold[neuron_unactive_indexes] = 1e14 # (just in case) set very big threshold to not have any spikes on unactive neurons

        # 1- Init Neurons
        self.voltage_init = parameter["voltage"][neuron_active_global_indexes]
        self.threshold = threshold[neuron_active_global_indexes]
        self.tau = parameter["tau"][neuron_active_global_indexes]
        self.constant_current = parameter["constant_current"][neuron_active_global_indexes]

        #2 - Init Synapses
        flat_indices = np.unique(neuron_active_global_indexes)
        rows, cols = np.meshgrid(flat_indices, flat_indices)
        self.weight = weight[cols, rows]
        # self.weight = parameter["weight"][cols, rows]


        # 3 - Init Time varying variables
        if is_refractory == True:
            self.refractory = parameter["refractory"][neuron_active_global_indexes]
        if is_delay == True:
            # self.delay = parameter["delay"][rows, cols]
            self.delay = parameter["delay"][cols, rows]
        if is_energy == True:
            self.energy = parameter["energy"][:, neuron_active_global_indexes]
            if is_energy_battery == True:
                self.energy_battery = parameter["energy_battery"][neuron_active_global_indexes]



cdef class SNN_cython_population:    

    cpdef void init_network(self,
                # 0 - NN General Parameters
                dict parameter, 

                np.ndarray[np.int32_t, ndim=1] population_genome_ids,

                # 1 - Neurons indexes
                np.ndarray[np.int32_t, ndim=1] input_indexes, 
                np.ndarray[np.int32_t, ndim=1] output_indexes, 
                np.ndarray[np.int32_t, ndim=1] hidden_indexes, 

                # 2 - Neurons and Synapses Population indexes active
                np.ndarray[np.int32_t, ndim=1] neuron_active_global_indexes, 
                np.ndarray[np.int32_t, ndim=2] synapse_active_global_indexes,

                # 3 - Neurons and Synapses Population indexes unactive
                # np.ndarray[np.int32_t, ndim=1] neuron_unactive_indexes,

                # 4 - Other Parameters that can optionnaly be used
                bint is_delay,
                bint is_refractory,
                bint is_energy,
                bint is_energy_battery,
                bint is_dynamic_topology,
                bint is_disable_output_threshold
                ):

        cdef np.ndarray flat_indices, rows, cols, threshold, weight

        # 0 - Init info indexes
        self.input_indexes = input_indexes
        self.output_indexes = output_indexes
        self.hidden_indexes = hidden_indexes

        # 1- Init Neurons
        self.voltage_init = parameter["voltage"]
        self.tau = parameter["tau"]
        self.constant_current = parameter["constant_current"]

        if is_dynamic_topology == True or is_disable_output_threshold == True:
            self.threshold = parameter["threshold"].copy()
        else:
            self.threshold = parameter["threshold"]


        #2 - Init Synapses
        if is_dynamic_topology == True: 
            self.weight = parameter["weight"].copy()
        else:
            self.weight = parameter["weight"]


        # 3 - Init Time varying variables
        if is_refractory == True:
            self.refractory = parameter["refractory"]
        if is_delay == True:
            self.delay = parameter["delay"]
        if is_energy == True:
            self.energy = parameter["energy"][:, :, neuron_active_global_indexes]
            if is_energy_battery == True:
                self.energy_battery = parameter["energy_battery"]

        # 4 - Init info indexes
        self.neuron_active_global_indexes = neuron_active_global_indexes
        self.synapse_active_global_indexes = synapse_active_global_indexes


        # 5 - Init info population
        self.population_genome_ids = population_genome_ids
        self.is_delay = is_delay
        self.is_refractory = is_refractory
        self.is_energy = is_energy
        self.is_energy_battery = is_energy_battery
        self.is_dynamic_topology = is_dynamic_topology

        self.nb_population = parameter["threshold"].shape[0]
        self.nb_neurons = parameter["threshold"].shape[1]

    cpdef void init_network_unactive_indexes(self, pop_idx, np.ndarray[np.int32_t, ndim=1] neuron_unactive_indexes, np.ndarray[np.int32_t, ndim=2] synapse_unactive_indexes):
        self.weight[pop_idx, synapse_unactive_indexes[0], synapse_unactive_indexes[1]] = 0.0 # set unactive synapses to 0.0
        self.threshold[pop_idx, neuron_unactive_indexes] = 1e14 # (just in case) set very big threshold to not have any spikes on unactive neurons
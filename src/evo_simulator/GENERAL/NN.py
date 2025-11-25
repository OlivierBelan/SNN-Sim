import numpy as np
from typing import Dict, Tuple, List
import numba as nb
from evo_simulator.GENERAL.Attribute import Attribute_Paramaters

class NN:
    def __init__(self, nb_inputs:int, nb_hiddens:int, nb_outputs:int, nb_hiddens_active:int, hiddens_config:List[int], hiddens_layer_names:List[str],  architecture:List[List[str]], is_self_neuron_connection:bool, inputs_multiplicator=1, outputs_multiplicator:int = 1, network_type:str = "SNN", attributes_manager:Attribute_Paramaters=None):

        # order of neurons: inputs -> outputs -> hiddens
        self.attributes_manager:Attribute_Paramaters = attributes_manager

        inputs_multiplicator:int = max(1, inputs_multiplicator)
        outputs_multiplicator:int = max(1, outputs_multiplicator)
        
        self.nb_inputs:int = nb_inputs * inputs_multiplicator if network_type == "SNN" else nb_inputs
        self.nb_inputs_original:int = nb_inputs

        self.nb_hiddens:int = nb_hiddens
        self.hiddens_config:List[int] = hiddens_config
        self.hiddens_layer_names:List[str] = hiddens_layer_names


        self.nb_outputs:int = nb_outputs * outputs_multiplicator if network_type == "SNN" else nb_outputs
        self.nb_outputs_original:int = nb_outputs
        self.outputs_multiplicator:int = outputs_multiplicator

        self.architecture_layers:List[List[str]] = architecture
        self.architecture_neurons:Dict[str, Dict[str, np.ndarray]] = {}
        self.is_self_neuron_connection:bool = is_self_neuron_connection


        self.nb_neurons:int = self.nb_inputs + self.nb_hiddens + self.nb_outputs
        self.nb_hiddens_active:int = nb_hiddens_active
        self.nb_neurons_active:int = self.nb_inputs + self.nb_hiddens_active + self.nb_outputs
        
        self.input_first_id:int = 0
        self.input_last_id:int = self.nb_inputs - 1
        self.output_first_id:int = self.nb_inputs
        self.output_last_id:int = self.nb_inputs + self.nb_outputs - 1
        self.hidden_first_id:int = self.nb_inputs + self.nb_outputs
        self.hidden_last_id:int = self.nb_inputs + self.nb_outputs + self.nb_hiddens - 1


        # Parameters
        self.parameters:Dict[str, np.ndarray] = {}

        # Build matrices
        self.neurons_status:np.ndarray = np.zeros(self.nb_neurons, dtype=np.bool_)
        self.neurons_status[:self.nb_inputs + self.nb_outputs] = True
        neurons_counter:int = self.nb_inputs + self.nb_outputs
        for layer_name in self.hiddens_layer_names:
            self.neurons_status[neurons_counter : neurons_counter + self.hiddens_config[layer_name]["nb_neurons_active"]] = True
            neurons_counter += self.hiddens_config[layer_name]["nb_neurons"]

        self.synapses_status:np.ndarray = np.zeros((self.nb_neurons, self.nb_neurons), dtype=np.bool_)
        self.synapses_indexes:Tuple[np.ndarray, np.ndarray] = np.where(self.synapses_status == False)

        self.neuron_deactived_status:np.ndarray = np.zeros(self.nb_hiddens, dtype=np.bool_)

        self.network_type:str = network_type
        if network_type == "SNN": self.__init_SNN()
        elif network_type == "ANN": self.__init_ANN()
        else: raise Exception("Network type '", network_type,"' type not found, the available types are: 'SNN', 'ANN'")
        
        self.inputs:Dict[str, np.ndarray] = {}
        self.hiddens:Dict[str, np.ndarray] = {}
        self.outputs:Dict[str, np.ndarray] = {}
        self.__inputs()
        self.__hiddens()
        self.__outputs()

        # print("nb_inputs: ", self.nb_inputs)
        # print("nb_hiddens: ", self.nb_hiddens)
        # print("nb_outputs: ", nb_outputs)
        # print("nb_hiddens_active: ", nb_hiddens_active)
        # print("hiddens_config: ", hiddens_config)
        # print("architecture_layers: ", self.architecture_layers)
        # print("architecture_neurons: ", self.architecture_neurons)
        # exit()

        # remove self connections
        np.fill_diagonal(self.synapses_status, False)

        # Connect layers
        self.__build_topology()

        self.neuron_actives_indexes:np.ndarray = None
        self.neuron_unactives_indexes:np.ndarray = None
        self.synapses_actives_indexes:Tuple[np.ndarray, np.ndarray] = None
        self.synapses_unactives_indexes:Tuple[np.ndarray, np.ndarray] = None
        self.synapses_unactives_weight_indexes:Tuple[np.ndarray, np.ndarray] = None

        self.update_indexes()


    def __init_SNN(self):
        # Neurons Parameters
        # Voltages
        self.parameters["voltage"] = np.zeros(self.nb_neurons, dtype=np.float32)

        # Thresholds
        self.parameters["threshold"] = np.zeros(self.nb_neurons, dtype=np.float32)

        # Tau (decay/leak)
        self.parameters["tau"] = np.zeros(self.nb_neurons, dtype=np.float32)

        # Constant current
        self.parameters["constant_current"] = np.zeros(self.nb_neurons, dtype=np.float32)

        # Refractory
        self.is_refractory:bool = True if "refractory" in self.attributes_manager.min_parameters else False
        if self.is_refractory == True:
            self.parameters["refractory"] = np.zeros(self.nb_neurons, dtype=np.float32)

        # Synapses Parameters
        # Weights
        self.parameters["weight"] = np.zeros((self.nb_neurons, self.nb_neurons), dtype=np.float32)

        # Delay
        self.is_delay:bool = True if "delay" in self.attributes_manager.min_parameters else False
        if self.is_delay == True:
            self.parameters["delay"] = np.zeros((self.nb_neurons, self.nb_neurons), dtype=np.float32)


        # Energy
        self.is_energy:bool = self.attributes_manager.is_energy
        self.is_energy_battery:bool = self.attributes_manager.is_energy_battery
        if self.is_energy == True:
            self.energy_length:int = self.attributes_manager.energy_length
            self.parameters["energy"] = np.zeros((self.energy_length, self.nb_neurons), dtype=np.float32)
            if self.is_energy_battery == True:
                self.parameters["energy_battery"] = np.zeros(self.nb_neurons, dtype=np.float32)

    def __init_ANN(self):
        # Neurons Parameters
        # bias
        self.parameters["bias"] = np.zeros(self.nb_neurons, dtype=np.float32)

        # Synapses Parameters
        # Weights
        self.parameters["weight"] = np.zeros((self.nb_neurons, self.nb_neurons), dtype=np.float32)

    def set_arbitrary_parameters(self) -> None:
        neuron_parameters_names:List[str] = self.attributes_manager.parameters_neuron_names
        synapse_parameters_names:List[str] = self.attributes_manager.parameters_synapse_names
        for param_name in self.attributes_manager.parameters_all_names:
            if param_name in self.parameters: # Parameters are set from the attributes_manager which contains information from your config file
                if param_name == "energy":
                    for i in range(self.attributes_manager.energy_length):
                        self.parameters[param_name][i] = self.epsilon_mu_sigma_jit(
                                                                                    parameter=self.parameters[param_name][i],  
                                                                                    mu_parameter=self.attributes_manager.mu_parameters[param_name],
                                                                                    sigma_paramater=self.attributes_manager.sigma_parameters[param_name],
                                                                                    min=self.attributes_manager.min_parameters[param_name],
                                                                                    max=self.attributes_manager.max_parameters[param_name],
                    )

                elif param_name in synapse_parameters_names: # synapses parameters
                    self.parameters[param_name][self.synapses_actives_indexes] = self.epsilon_mu_sigma_jit(
                                                                                    parameter=self.parameters[param_name][self.synapses_actives_indexes],  
                                                                                    mu_parameter=self.attributes_manager.mu_parameters[param_name],
                                                                                    sigma_paramater=self.attributes_manager.sigma_parameters[param_name],
                                                                                    min=self.attributes_manager.min_parameters[param_name],
                                                                                    max=self.attributes_manager.max_parameters[param_name],
                    )
                elif param_name in neuron_parameters_names and param_name != "energy": # neuron parameters
                    self.parameters[param_name] = self.epsilon_mu_sigma_jit(
                                                                                    parameter=self.parameters[param_name],  
                                                                                    mu_parameter=self.attributes_manager.mu_parameters[param_name],
                                                                                    sigma_paramater=self.attributes_manager.sigma_parameters[param_name],
                                                                                    min=self.attributes_manager.min_parameters[param_name],
                                                                                    max=self.attributes_manager.max_parameters[param_name],
                )

    def update_indexes(self):
        self.neuron_actives_indexes, self.neuron_unactives_indexes, self.hiddens["neurons_indexes_active"], self.synapses_actives_indexes, self.synapses_unactives_indexes, self.synapses_unactives_weight_indexes = self.update_indexes_jit(
            self.neurons_status,
            self.synapses_status,
            self.hiddens["neurons_indexes"],
            self.hiddens["neurons_status"],
            self.parameters["weight"],
        )
        self.nb_hiddens_active = self.hiddens["neurons_indexes_active"].shape[0]
        self.nb_neurons_active = self.nb_inputs + self.nb_hiddens_active + self.nb_outputs
        self.synapses_actives_indexes_unique, self.synapses_actives_indexes_first_indices, synapses_actives_indexes_count = np.unique(self.synapses_actives_indexes[0], return_index=True, return_counts=True)
        self.neuron_least_synapses_index:int = np.argmin(synapses_actives_indexes_count)
        self.neuron_least_synapses_count:int = synapses_actives_indexes_count[self.neuron_least_synapses_index]        

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def update_indexes_jit(
        neurons_status:np.ndarray,
        synapses_status:np.ndarray,
        hidden_neurons_indexes:np.ndarray,
        hidden_neurons_status:np.ndarray,
        weight_matrix:np.ndarray,
        ):
        # 1- Find the active neurons' indices (all and hidden)
        neuron_actives_indexes:np.ndarray = np.where(neurons_status)[0].astype(np.int32)
        neuron_unactives_indexes:np.ndarray = np.where(~neurons_status)[0].astype(np.int32)
        hidden_neurons_indexes_active:np.ndarray = hidden_neurons_indexes[np.where(hidden_neurons_status)[0]].astype(np.int32)
        
        # 2 - Find the active and unactive synapses' indices for the given active neurons
        neuron_actives_len:int = neuron_actives_indexes.shape[0]
        neuron_actives_synapses_status:np.ndarray = np.empty((neuron_actives_len, neuron_actives_len), dtype=synapses_status.dtype)
        for i in nb.prange(neuron_actives_len):
            for j in nb.prange(neuron_actives_len):
                neuron_actives_synapses_status[i, j] = synapses_status[neuron_actives_indexes[i], neuron_actives_indexes[j]]
       
        # Find the active and inactive synapses' indices for the given active neurons
        synapses_actives_local_indexes:Tuple[np.ndarray, np.ndarray] = np.where(neuron_actives_synapses_status)
        synapses_unactives_local_indexes:Tuple[np.ndarray, np.ndarray] = np.where(~neuron_actives_synapses_status)

        # Map the local indices back to the original indices in the 'synapses_status' matrix
        synapses_actives_indexes: Tuple[np.ndarray, np.ndarray] = (neuron_actives_indexes[synapses_actives_local_indexes[0]], neuron_actives_indexes[synapses_actives_local_indexes[1]])
        synapses_unactives_indexes: Tuple[np.ndarray, np.ndarray] = (neuron_actives_indexes[synapses_unactives_local_indexes[0]], neuron_actives_indexes[synapses_unactives_local_indexes[1]])

        # 3 - Find the active and unactive synapses' indices given the active weight (weight != 0)
        synapses_unactives_len:int = synapses_unactives_indexes[0].shape[0]
        synapses_unactives_weight_indexes:Tuple[np.ndarray, np.ndarray] = (np.empty(synapses_unactives_len, dtype=np.int32), np.empty(synapses_unactives_len, dtype=np.int32))

        weight, row, col, current_size = 0.0, 0, 0, 0
        for i in nb.prange(synapses_unactives_len):
            row:int = synapses_unactives_indexes[0][i]
            col:int = synapses_unactives_indexes[1][i]
            weight:float = weight_matrix[row, col]
            if weight > 0.001 or weight < -0.001:
                synapses_unactives_weight_indexes[0][current_size] = row
                synapses_unactives_weight_indexes[1][current_size] = col
                current_size += 1
        
        # Trim the arrays to the actual size
        synapses_unactives_weight_indexes:Tuple[np.ndarray, np.ndarray] = (synapses_unactives_weight_indexes[0][:current_size], synapses_unactives_weight_indexes[1][:current_size])
        
        return neuron_actives_indexes, neuron_unactives_indexes, hidden_neurons_indexes_active, synapses_actives_indexes, synapses_unactives_indexes, synapses_unactives_weight_indexes

    def __build_topology(self):
        # Build the topology of the network
        for layer in self.architecture_layers:
            self.connect_layers(self.architecture_neurons[layer[0]], self.architecture_neurons[layer[1]])
            


    def __set_synapses_indexes(self):
        np.fill_diagonal(self.synapses_status, True)
        self.inputs["synapses_indexes"] = np.where(self.synapses_status[:self.nb_inputs] == False)
        self.inputs["synapses_indexes"] = tuple(np.array([self.inputs["synapses_indexes"][0], self.inputs["synapses_indexes"][1]]))
        self.outputs["synapses_indexes"] = np.where(self.synapses_status[self.nb_inputs:self.nb_inputs+self.nb_outputs] == False)
        self.outputs["synapses_indexes"] = tuple(np.array([self.outputs["synapses_indexes"][0] + self.output_first_id, self.outputs["synapses_indexes"][1]]))
        self.hiddens["synapses_indexes"] = np.where(self.synapses_status[self.nb_inputs+self.nb_outputs:] == False)
        self.hiddens["synapses_indexes"] = tuple(np.array([self.hiddens["synapses_indexes"][0] + self.hidden_first_id, self.hiddens["synapses_indexes"][1]]))
        np.fill_diagonal(self.synapses_status, False)

    def __set_synapses_actives_indexes(self):
        self.inputs["synapses_actives_indexes"] = np.where(self.synapses_status[:self.nb_inputs] == True)
        self.inputs["synapses_actives_indexes"] = tuple(np.array([self.inputs["synapses_actives_indexes"][0], self.inputs["synapses_actives_indexes"][1]]))
        self.outputs["synapses_actives_indexes"] = np.where(self.synapses_status[self.nb_inputs:self.nb_inputs+self.nb_outputs] == True)
        self.outputs["synapses_actives_indexes"] = tuple(np.array([self.outputs["synapses_actives_indexes"][0] + self.output_first_id, self.outputs["synapses_actives_indexes"][1]]))
        self.hiddens["synapses_actives_indexes"] = np.where(self.synapses_status[self.nb_inputs+self.nb_outputs:] == True)
        self.hiddens["synapses_actives_indexes"] = tuple(np.array([self.hiddens["synapses_actives_indexes"][0] + self.hidden_first_id, self.hiddens["synapses_actives_indexes"][1]]))

    def connect_layers(self, source:Dict[str, np.ndarray], target:Dict[str, np.ndarray]):
        
        # Get active neurons indexes
        source_active_neurons_indexes = source["neurons_indexes"][source["neurons_status"] == True]
        target_active_neurons_indexes = target["neurons_indexes"][target["neurons_status"] == True]

        # Create meshgrid of active neuron indexes
        source_mesh, target_mesh = np.meshgrid(source_active_neurons_indexes, target_active_neurons_indexes)

        # Reshape meshgrid to 1D arrays
        source_synapses_indexes = source_mesh.ravel()
        target_synapses_indexes = target_mesh.ravel()

        # Connect neurons
        self.synapses_status[source_synapses_indexes, target_synapses_indexes] = True

        # deactivate self-connection
        if self.is_self_neuron_connection == False:
            np.fill_diagonal(self.synapses_status, False)

    def __inputs(self):
        if self.network_type == "SNN":
            self.inputs["voltage"] = self.parameters["voltage"][:self.nb_inputs]
            self.inputs["threshold"] = self.parameters["threshold"][:self.nb_inputs]
            self.inputs["tau"] = self.parameters["tau"][:self.nb_inputs]
            self.inputs["constant_current"] = self.parameters["constant_current"][:self.nb_inputs]
            self.inputs["weight"] = self.parameters["weight"][:self.nb_inputs]

            if self.is_refractory == True:
                self.inputs["refractory"] = self.parameters["refractory"][:self.nb_inputs]
            if self.is_delay == True:
                self.inputs["delay"] = self.parameters["delay"][:self.nb_inputs]
            if self.is_energy == True:
                self.inputs["energy"] = self.parameters["energy"][:, :self.nb_inputs]
            if self.is_energy_battery == True:
                self.inputs["energy_battery"] = self.parameters["energy_battery"][:self.nb_inputs]

        elif self.network_type == "ANN":
            self.inputs["bias"] = self.parameters["bias"][:self.nb_inputs]
            self.inputs["weight"] = self.parameters["weight"][:self.nb_inputs]

        self.inputs["neurons_status"] = self.neurons_status[:self.nb_inputs]
        self.inputs["synapses_status"] = self.synapses_status[:self.nb_inputs]
        self.inputs["neurons_indexes"] = np.arange(self.nb_inputs, dtype=np.int32)

        self.architecture_neurons["I"] = {}
        self.architecture_neurons["I"]["neurons_indexes"] = self.inputs["neurons_indexes"]
        self.architecture_neurons["I"]["neurons_status"] = self.inputs["neurons_status"]
        self.architecture_neurons["I"]["size"] = self.nb_inputs

    def __hiddens(self):
        if self.network_type == "SNN":
            self.hiddens["voltage"] = self.parameters["voltage"][self.nb_inputs+self.nb_outputs:]
            self.hiddens["threshold"] = self.parameters["threshold"][self.nb_inputs+self.nb_outputs:]
            self.hiddens["tau"] = self.parameters["tau"][self.nb_inputs+self.nb_outputs:]
            self.inputs["constant_current"] = self.parameters["constant_current"][self.nb_inputs+self.nb_outputs:]
            self.hiddens["weight"] = self.parameters["weight"][self.nb_inputs+self.nb_outputs:]

            if self.is_refractory == True:
                self.hiddens["refractory"] = self.parameters["refractory"][self.nb_inputs+self.nb_outputs:]
            if self.is_delay == True:
                self.inputs["delay"] = self.parameters["delay"][self.nb_inputs+self.nb_outputs:]
            if self.is_energy == True:
                self.hiddens["energy"] = self.parameters["energy"][:, self.nb_inputs+self.nb_outputs:]
            if self.is_energy_battery == True:
                self.hiddens["energy_battery"] = self.parameters["energy_battery"][self.nb_inputs+self.nb_outputs:]

        elif self.network_type == "ANN":
            self.hiddens["bias"] = self.parameters["bias"][self.nb_inputs+self.nb_outputs:]
            self.hiddens["weight"] = self.parameters["weight"][self.nb_inputs+self.nb_outputs:]

        self.hiddens["neurons_status"] = self.neurons_status[self.nb_inputs+self.nb_outputs:]
        self.hiddens["synapses_status"] = self.synapses_status[self.nb_inputs+self.nb_outputs:]
        self.hiddens["neurons_indexes"] = np.arange(self.nb_inputs+self.nb_outputs, self.nb_inputs+self.nb_outputs+self.nb_hiddens, dtype=np.int32)
        self.hiddens["neurons_indexes_active"] = self.hiddens["neurons_indexes"][np.where(self.hiddens["neurons_status"] == True)[0]]

        prev_indexes:int = 0
        prev_indexes_active:int = 0
        # for layer_name, nb_neurons in zip(self.hiddens_layer_names, self.hiddens_config):
        for layer_name in self.hiddens_layer_names:
            nb_neurons:int = self.hiddens_config[layer_name]["nb_neurons"]
            nb_neurons_active:int = self.hiddens_config[layer_name]["nb_neurons_active"]
            self.architecture_neurons[layer_name] = {}
            self.architecture_neurons[layer_name]["neurons_indexes"] = self.hiddens["neurons_indexes"][prev_indexes:prev_indexes+nb_neurons]
            self.architecture_neurons[layer_name]["neurons_indexes_active"] = self.hiddens["neurons_indexes_active"][prev_indexes_active:prev_indexes_active+nb_neurons_active]
            self.architecture_neurons[layer_name]["neurons_status"] = self.hiddens["neurons_status"][prev_indexes:prev_indexes+nb_neurons]
            self.architecture_neurons[layer_name]["size"] = nb_neurons
            prev_indexes += nb_neurons
            prev_indexes_active += nb_neurons_active
        # print("hidden_neurons_status: ", self.hiddens["neurons_status"])
        # print("hidden_neurons_indexes: ", self.hiddens["neurons_indexes"])
        # print("hidden_neurons_indexes_active: ", self.hiddens["neurons_indexes_active"])
        # print("self.architecture_neurons: ", self.architecture_neurons)
        # exit()

    def __outputs(self):
        if self.network_type == "SNN":
            self.outputs["voltage"] = self.parameters["voltage"][self.nb_inputs:self.nb_inputs+self.nb_outputs]
            self.outputs["threshold"] = self.parameters["threshold"][self.nb_inputs:self.nb_inputs+self.nb_outputs]
            self.outputs["tau"] = self.parameters["tau"][self.nb_inputs:self.nb_inputs+self.nb_outputs]
            self.inputs["constant_current"] = self.parameters["constant_current"][self.nb_inputs:self.nb_inputs+self.nb_outputs]
            self.outputs["weight"] = self.parameters["weight"][self.nb_inputs:self.nb_inputs+self.nb_outputs]

            if self.is_refractory == True:
                self.outputs["refractory"] = self.parameters["refractory"][self.nb_inputs:self.nb_inputs+self.nb_outputs]
            if self.is_delay == True:
                self.inputs["delay"] = self.parameters["delay"][self.nb_inputs:self.nb_inputs+self.nb_outputs]
            if self.is_energy == True:
                self.outputs["energy"] = self.parameters["energy"][:, self.nb_inputs:self.nb_inputs+self.nb_outputs]
            if self.is_energy_battery == True:
                self.outputs["energy_battery"] = self.parameters["energy_battery"][self.nb_inputs:self.nb_inputs+self.nb_outputs]
        
        elif self.network_type == "ANN":
            self.outputs["bias"] = self.parameters["bias"][self.nb_inputs:self.nb_inputs+self.nb_outputs]
            self.outputs["weight"] = self.parameters["weight"][self.nb_inputs:self.nb_inputs+self.nb_outputs]

        self.outputs["neurons_status"] = self.neurons_status[self.nb_inputs:self.nb_inputs+self.nb_outputs]
        self.outputs["synapses_status"] = self.synapses_status[self.nb_inputs:self.nb_inputs+self.nb_outputs]
        self.outputs["neurons_indexes"] = np.arange(self.nb_inputs, self.nb_inputs+self.nb_outputs,  dtype=np.int32)
        
        self.architecture_neurons["O"] = {}
        self.architecture_neurons["O"]["neurons_indexes"] = self.outputs["neurons_indexes"]
        self.architecture_neurons["O"]["neurons_status"] = self.outputs["neurons_status"]
        self.architecture_neurons["O"]["size"] = self.nb_outputs


        # In case of population coding, slice the output to split the output neurons for each classs 
        # e.g for 2 classes -> 2 output neurons is needed for each class but we have 10 output neurons in total, 
        # therefore we need to split the output neurons for each class (5 neurons for each class in this case)
        self.outputs["neurons_indexes_formated"] = np.array(np.split(self.outputs["neurons_indexes"], self.nb_outputs_original))

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def epsilon_mu_sigma_jit(parameter:np.ndarray, mu_parameter:np.ndarray, sigma_paramater:np.ndarray, min:np.ndarray, max:np.ndarray, mu_bias:float=0.0, sigma_coef:float=1.0) -> np.ndarray:
        '''
        Jit function for the epsilon computation
        '''
        # 1- set Epislon with the Gaussian (from randn) distribution (Mu -> center of the distribution, Sigma -> width of the distribution)
        mu:np.ndarray = mu_parameter + mu_bias
        sigma:np.ndarray = sigma_paramater * sigma_coef
        epsilon:np.ndarray = np.random.randn(1, parameter.size) * sigma + mu
        # 2- clip Epsilon and apply it to the neurons parameters
        return np.clip(epsilon.astype(np.float32)[0], min, max)

    def update_synpases_layers_indexes(self):
        self.synapses_layers_size:List[int] = []
        self.synapses_layers_indexes:Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        # nb_synapses_weights_layer_indexes:int = 0

        for source_name, target_name in self.architecture_layers:
            self.synapses_layers_indexes[source_name + "->" + target_name] = self.get_weight_layers_indexes(self.synapses_indexes, self.architecture_neurons[source_name]["neurons_indexes"], self.architecture_neurons[target_name]["neurons_indexes"])
            # nb_synapses_weights_layer_indexes += self.synapses_layers_indexes[source_name + "->" + target_name][0].size

        # for connection_name, synapses_indexes in self.synapses_layers_indexes.items():
        for synapses_indexes in self.synapses_layers_indexes.values():
            # print(connection_name, ":", synapses_indexes, "size", synapses_indexes[0].size)
            self.synapses_layers_size.append(synapses_indexes[0].size)

        # print("nb_synapses_weights_layer_indexes:", nb_synapses_weights_layer_indexes, "nb_synapses:", self.synapse_parameters_size, "nb_layers", self.layer_parameters_size,"\n")
        # print("synapses_layers_size", synapses_layers_size, "nn_layer_least_synapses_count", np.min(synapses_layers_size))
        # exit()

    def get_weight_layers_indexes(self, synapses_indexes:Tuple[np.ndarray, np.ndarray], source_indexes:np.ndarray, target_indexes:np.ndarray) -> np.ndarray:
        input_synpases_indexes = np.where(np.isin(synapses_indexes[0], source_indexes))[0]
        target_synapse_indexes = np.where(np.isin(synapses_indexes[1][input_synpases_indexes], target_indexes))[0]
        return (synapses_indexes[0][input_synpases_indexes[target_synapse_indexes]], synapses_indexes[1][input_synpases_indexes[target_synapse_indexes]])

from typing import Dict, Any, List, Callable
from evo_simulator.GENERAL.NN import NN
from evo_simulator.GENERAL.Fitness import Fitness
from evo_simulator.GENERAL.Attribute import Attribute_Paramaters
import numpy as np
import evo_simulator.TOOLS as TOOLS

class Genome:
    def __init__(self, id:int, config_genome:Dict[str, Any], attributes_manager:Attribute_Paramaters):
        self.id:int = id
        self.fitness:Fitness = Fitness()
        self.info:Dict[str, Any] = {"is_elite":False}
        self.config_genome:Dict[str, Any] = config_genome
        self.attributes_manager:Attribute_Paramaters = attributes_manager

class Genome_NN(Genome):
    def __init__(self, id:int, config_genome:Dict[str, Any], attributes_manager:Attribute_Paramaters, hiddens_active:float=None) -> None:
        Genome.__init__(self, id, config_genome, attributes_manager)

        self.__init_config(config_genome)
        if hiddens_active is not None:
            self.hiddens_active = hiddens_active
        self.nn:NN = NN(
                    nb_inputs=self.inputs, 
                    nb_outputs=self.outputs,
                    nb_hiddens=self.hiddens,
                    nb_hiddens_active=self.hiddens_active, 
                    hiddens_config=self.hiddens_config,
                    hiddens_layer_names=self.hiddens_layer_names,
                    architecture=self.architecture,
                    is_self_neuron_connection=self.is_self_neuron_connection,
                    inputs_multiplicator=self.inputs_multiplicator,
                    outputs_multiplicator=self.outputs_multiplicator, 
                    network_type=self.network_type, 
                    attributes_manager=self.attributes_manager
                    )

        # Population index
        self.population_idx:int = None
        self.sync_with_population_func:Callable = None

        #  Torch network
        self.net_torch = None

        # Combinatorial modulo
        self.combinatorial_modulo:float = 1.0
        self.combinatorial_modulo_sigma:float = 1.0
        self.combinatorial_modulo_ocillator_noise:float = 0.0

        # Action dynamic RL
        self.action_offset_max:np.ndarray = None # np.zeros(self.outputs)
        self.action_offset_min:np.ndarray = None # np.zeros(self.outputs)
        self.action_offset_sigma_max:np.ndarray = None # np.zeros(self.outputs)
        self.action_offset_sigma_min:np.ndarray = None # np.zeros(self.outputs)
        self.action_offset_ocillator_noise:np.ndarray = None # np.zeros(self.outputs)
        self.action_std_range_factor:float = 2.0

        self.action_local_min:np.ndarray = None # np.zeros(self.outputs)
        self.action_local_max:np.ndarray = None # np.zeros(self.outputs)
    
    def __init_config(self, config_genome) -> None:
        self.config_genome:Dict[str, Any] = config_genome

        self.inputs:int = int(self.config_genome["inputs"])
        self.inputs_multiplicator:int = max(1, int(self.config_genome["inputs_multiplicator"])) # number of neurons used to represent one input

        self.hiddens_config:Dict[str, Any] = TOOLS.hiddens_from_config(self.config_genome["hiddens"])
        self.hiddens:int = self.hiddens_config["nb_hiddens"]
        self.hiddens_active:int = self.hiddens_config["nb_hiddens_active"]


        self.outputs:int = int(self.config_genome["outputs"])
        self.outputs_multiplicator:int = max(1, int(self.config_genome["outputs_multiplicator"])) # number of neurons used to represent one output

        self.architecture, self.hiddens_layer_names = TOOLS.architecture_from_config(self.config_genome["architecture"], len(self.hiddens_config["layer_names"]))
        self.architecture_print_orignal:str = self.config_genome["inputs"] + "x" + self.config_genome["hiddens"] + "x" + self.config_genome["outputs"]
        self.architecture_print_multiplied:str = str(self.inputs*self.inputs_multiplicator) + "x" + self.config_genome["hiddens"] + "x" + str(self.outputs*self.outputs_multiplicator)

        self.is_self_neuron_connection:bool = True if self.config_genome["is_self_neuron_connection"] == "True" else False

        self.network_type:str = self.config_genome["network_type"]

        # if self.network_type == "ANN":
        #     self.is_inter_hidden_feedback:bool = True if self.config_genome["is_inter_hidden_feedback"] == "True" else False
        #     self.is_layer_normalization:bool = True if self.config_genome["is_layer_normalization"] == "True" else False

    def update_hiddens(self, hiddens:Dict[str, int]) -> None:
        nb_hidden:int = 0
        for hidden_name, nb_neurons in hiddens.items():
            if hidden_name in self.hiddens_config["layer_names"]:
                self.hiddens_config[hidden_name]["nb_neurons"] = nb_neurons
                self.hiddens_config[hidden_name]["nb_neurons_active"] = nb_neurons
                nb_hidden += nb_neurons


        self.hiddens_config["nb_hiddens"] = nb_hidden
        self.hiddens_config["nb_hiddens_active"] = nb_hidden
        self.hiddens = nb_hidden
        self.hiddens_active = nb_hidden

        self.nn:NN = NN(
                    nb_inputs=self.inputs, 
                    nb_outputs=self.outputs,
                    nb_hiddens=self.hiddens,
                    nb_hiddens_active=self.hiddens_active, 
                    hiddens_config=self.hiddens_config,
                    hiddens_layer_names=self.hiddens_layer_names,
                    architecture=self.architecture,
                    is_self_neuron_connection=self.is_self_neuron_connection,
                    inputs_multiplicator=self.inputs_multiplicator,
                    outputs_multiplicator=self.outputs_multiplicator, 
                    network_type=self.network_type, 
                    attributes_manager=self.attributes_manager
                    )

    def get_hiddens_config_nb_neurons(self) -> Dict[str, int]:
        hiddens:Dict[str, int] = {}
        layers:List[str] = self.hiddens_config["layer_names"]
        for hidden_name in layers:
            hiddens[hidden_name] = self.hiddens_config[hidden_name]["nb_neurons"]
        return hiddens
    
    def sync_with_population(self) -> None:
        if self.sync_with_population_func is None: raise Exception("The sync_with_population_func is not defined")
        self.sync_with_population_func(self)
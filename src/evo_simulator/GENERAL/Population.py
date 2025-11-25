from evo_simulator.GENERAL.Attribute import Attribute_Paramaters
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.Fitness import Fitness
from evo_simulator.GENERAL.NN import NN
from evo_simulator.GENERAL.Index_Manager import get_new_genome_id

from typing import Dict, Any, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from evo_simulator import TOOLS

class Population:
    def __init__(self, id:int, algo_name_config:str, config_path_file:str, attribute_manager:Attribute_Paramaters, extra_info:Dict[Any, Any] = {}) -> None:
        # General
        self.id:int = id
        self.population:Dict[int, Genome_NN] = {}
        self.population_running:Dict[int, Genome_NN] = {}
        self.population_reproduction:Dict[int, Genome_NN] = {}
        self.best_genome:Genome_NN = None
        self.extra_info:Dict[str, Any] = extra_info
        self.config_path_file:str = config_path_file
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["NEURO_EVOLUTION"])
        self.optimization_type:str = self.config["NEURO_EVOLUTION"]["optimization_type"] # maximize, minimize, closest_to_zero
        self.is_population_initialized:bool = False

        self.keys:List[str] = None
        self.is_action_local:bool = False
        self.is_action_mean:bool = False

        # Parameters
        self.age: int = 0
        self.stagnation: int = 0
        self.reproduction_size: int = 1
        self.size = 0

        # Fitness
        self.fitness: Fitness = Fitness()
        self.extra_info["genome_idx_ranked"] = None
        self.extra_info["action_max_step"]   = None
        self.extra_info["action_min_step"]   = None

    def update(self, population:"Population" | List["Population"]) -> None:
        if isinstance(population, List):
            for pop in population:
                self.population.update(pop.population)
        else:
            self.population.update(population.population)

    def update_array(self, results:np.ndarray | List[np.ndarray] | List[Dict[str, np.ndarray]] | List[Dict[str, List[int]]], is_RL:bool=False, obs_max:np.ndarray=None, obs_min:np.ndarray=None, seed:np.ndarray=None) -> None:
        if is_RL == True:

            if self.keys is None: self.keys = ["best_episode_raw_score", "mean_episode_raw_score", "action_local_min", "action_local_max"]

            if isinstance(results, list):
                genome_idx: int = 0
                self.is_action_step   = "action_max_step" in results[0]
                self.is_action_mean   = "action_max_mean" in results[0]

                if self.is_action_step == True:
                    self.extra_info["action_max_step"] = []
                    self.extra_info["action_min_step"] = []

                for res in results:
                    # print("\n\n")
                    # for k in self.keys:
                    #     print("results[" + k + "]:", res[k], "\n")
                    # print("\n")

                    if self.is_action_step == True:
                        self.extra_info["action_max_step"].extend(res["action_max_step"])
                        self.extra_info["action_min_step"].extend(res["action_min_step"])

                    nb_values = len(res["best_episode_raw_score"])
                    for i in range(nb_values):
                        genome: Genome_NN = self.population[genome_idx]
                        genome_idx += 1

                        for k in self.keys: genome.info[k] = res[k][i] # add the information to the genome info (e.g. best_episode_raw_score, mean_episode_raw_score, etc.)

                        genome.fitness.score = genome.info["mean_episode_raw_score"]
                        genome.info["obs_max"] = obs_max
                        genome.info["obs_min"] = obs_min
                        genome.info["seed"] = seed
                        genome.action_local_max = genome.info["action_local_max"]
                        genome.action_local_min = genome.info["action_local_min"]

                        if self.is_action_mean == True:
                            genome.info["action_max_mean"] = res["action_max_mean"]
                            genome.info["action_min_mean"] = res["action_min_mean"]
                            genome.info["action_max_std"]  = res["action_max_std"]
                            genome.info["action_min_std"]  = res["action_min_std"]
                            

                        # print("genome_info:", genome.info)
                        # print("genome_offset_max:", genome.action_offset_max)
                        # print("genome_offset_min:", genome.action_offset_min)

                if self.is_action_step == True:
                    self.extra_info["genome_idx_ranked"] = np.array(sorted(self.population, key=lambda x: self.population[x].fitness.score, reverse=True), dtype=np.int32)
                    self.extra_info["action_max_step"] = np.array(self.extra_info["action_max_step"], dtype=np.float32)
                    self.extra_info["action_min_step"] = np.array(self.extra_info["action_min_step"], dtype=np.float32)
                    # print("self.extra_info:", self.extra_info)
                # exit("from update_array in Population.py")

                # genome_idx:int = 0
                # for res_dict in results:
                #     nb_values = len(res_dict["best_episode_raw_score"])
                #     for i in range(nb_values):
                #         genome = self.population[genome_idx]
                #         for key, values_list in res_dict.items():
                #             genome.info[key] = values_list[i]
                #         if "best_episode_raw_score" in genome.info:
                #             genome.fitness.score = genome.info["best_episode_raw_score"]

                #         genome_idx += 1
            else:
                genome_idx: int = 0                    
                nb_values = len(results["best_episode_raw_score"])
                self.is_action_step = "action_max_step" in results
                self.is_action_mean = "action_max_mean" in results

                # print("\n\n")
                # for k in self.keys:
                #     print(k, ":", results[k])
                #     print("\n")
                # exit()

                if self.is_action_step == True:
                    self.extra_info["genome_idx_ranked"] = np.array(sorted(self.population, key=lambda x: self.population[x].fitness.score, reverse=True), dtype=np.int32)
                    self.extra_info["action_max_step"]   = np.array(results["action_max_step"], dtype=np.float32)
                    self.extra_info["action_min_step"]   = np.array(results["action_min_step"], dtype=np.float32)
                    # print("self.extra_info:", self.extra_info)
                # exit("from update_array in Population.py")

                for i in range(nb_values):
                    genome: Genome_NN = self.population[genome_idx]
                    genome_idx += 1

                    for k in self.keys: genome.info[k] = results[k][i] # add the information to the genome info (e.g. best_episode_raw_score, mean_episode_raw_score, etc.)

                    genome.fitness.score = genome.info["mean_episode_raw_score"]
                    genome.info["obs_max"] = obs_max
                    genome.info["obs_min"] = obs_min
                    genome.info["seed"] = seed
                    genome.action_local_max = genome.info["action_local_max"]
                    genome.action_local_min = genome.info["action_local_min"]

                    if self.is_action_mean == True:
                        genome.info["action_max_mean"] = results["action_max_mean"]
                        genome.info["action_min_mean"] = results["action_min_mean"]
                        genome.info["action_max_std"]  = results["action_max_std"]
                        genome.info["action_min_std"]  = results["action_min_std"]

                    # print("genome_info:", genome.info)
                    # print("genome_offset_max:", genome.action_offset_max)
                    # print("genome_offset_min:", genome.action_offset_min)
                # exit("from update_array in Population.py")

                # genome_idx:int = 0
                # nb_values = len(results["best_episode_raw_score"])
                # for i in range(nb_values):
                #     genome = self.population[genome_idx]
                #     for key, values_list in res_dict.items():
                #         genome.info[key] = values_list[i]
                #     if "best_episode_raw_score" in genome.info:
                #         genome.fitness.score = genome.info["best_episode_raw_score"]
                #     genome_idx += 1

        else:
            if isinstance(results, List):
                results = np.concatenate(results)
            for i, genome in enumerate(self.population.values()):
                genome.fitness.score = results[i]

    def replace(self, population:"Population") -> None:
        self.population = population.population

    def update_info(self, optimization_type:str = None) -> None:
        self.reset()
        if optimization_type is not None:
            if optimization_type not in ["maximize", "minimize", "closest_to_zero"]: raise ValueError("optimization_type must be 'maximize', 'minimize' or 'closest_to_zero'")
            self.optimization_type = optimization_type

        # 0 - Get current fitness and topology info of the population
        for index, genome in enumerate(self.population.values()):
            # 1 - Fitness  
            genome_score:float = genome.fitness.score if self.optimization_type != "closest_to_zero" else abs(genome.fitness.score)
            # 1.1 - Fitness max
            if index == 0 or genome_score > self.fitness.max:
                self.fitness.max = genome_score
                if self.optimization_type == "maximize":
                    self.best_genome = genome            
            # 1.2 - Fitness min
            if index == 0 or genome_score < self.fitness.min:
                self.fitness.min = genome_score
                if self.optimization_type in ["minimize", "closest_to_zero"]:
                    self.best_genome = genome
            # 1.3 - Fitness mean
            self.fitness.mean += genome_score

        # 3 - Update mean
        self.size:int = len(self.population)
        if self.size > 0:
            self.fitness.mean /= self.size

        # 4 - Fitness best score and history
        self.fitness.score = self.fitness.max if self.optimization_type == "maximize" else self.fitness.min
        self.fitness.history_best.append(self.fitness.score)
        self.fitness.history_mean.append(self.fitness.mean)
    
    def reset(self):
        self.fitness.reset()


class Population_NN(Population):
    def __init__(self, id:int, algo_name_config:str, config_path_file:str, attribute_manager:Attribute_Paramaters, extra_info:Dict[Any, Any] = {}) -> None:

        Population.__init__(self, id, algo_name_config, config_path_file, attribute_manager, extra_info)

        self.best_genome:Genome_NN = None

        # Topology
        self.neurons_max:int = 0
        self.neurons_mean:int = 0
        self.neurons_min:int = 0

        self.synapses_max:int = 0
        self.synapses_mean:int = 0
        self.synapses_min:int = 0

        self.parameters:Dict[str, np.ndarray] = {}

        # self.config.update(TOOLS.config_function(config_path_file, ["Genome_NN"]))

        self.config.update(TOOLS.config_function(config_path_file, [algo_name_config, "Genome_NN","NEURO_EVOLUTION", "Runner_Info"]))
        self.params_to_update:List[str] = list(set(self.config[algo_name_config]["params_to_update"].replace(" ", "").split(",")))
        self.attributes_manager:Attribute_Paramaters = attribute_manager

        self.neuron_params_names:set[str] = set(self.attributes_manager.parameters_neuron_names)
        self.synapse_params_names:set[str] = set(self.attributes_manager.parameters_synapse_names)

        self.pop_size:int = int(self.config[algo_name_config]["pop_size"])
        self.population_genome_ids:np.ndarray = np.arange(self.pop_size, dtype=np.int32)
        self.population_genome_ids_list:List[int] = self.population_genome_ids.tolist()
        self.network_type:str = self.config["Genome_NN"]["network_type"]
        self.is_dynamic_topology:bool = True if self.config["Genome_NN"]["is_dynamic_topology"] == "True" else False
        self.is_first_sync_topology:bool = True
        self.genome_core:Genome_NN = None

        if "SNN" in self.config["Genome_NN"]["network_type"]:
            self.is_SNN:bool = True
            self.is_refractory:bool = None
            self.is_delay:bool = None
            self.is_energy:bool = None
            self.is_energy_battery:bool = None
            if self.is_energy == True: self.energy_length:int = None
        elif "ANN" in self.config["Genome_NN"]["network_type"]:
            self.is_SNN:bool = False

    def sync_genomes_to_population(self, is_sync_population_to_genome:bool = False) -> None:

        self.__init_NN() # only init if not already initialized
        self.population_running = {}
        self.population_reproduction = {}
        for i, genome in enumerate(self.population.values()):
            nn:NN = genome.nn
            # genome.id = get_new_genome_id() # set that as same as the population index (I need to see if I can keep only one index)
            genome.population_idx = i
            genome.sync_with_population_func = self.sync_one_genome_to_population
            # get_new_genome_id() # I need to increment it as it could be used in the future by other functions (e.g NEAT in reproduction)

            if self.is_SNN == True: # SNN
                # 1 - Transfer Genome parameters to Population
                self.parameters["voltage"][i] = nn.parameters["voltage"]
                self.parameters["threshold"][i] = nn.parameters["threshold"]
                self.parameters["tau"][i] = nn.parameters["tau"]
                self.parameters["constant_current"][i] = nn.parameters["constant_current"]

                if self.is_refractory == True:
                    self.parameters["refractory"][i] = nn.parameters["refractory"]
                
                self.parameters["weight"][i] = nn.parameters["weight"]
                
                if self.is_delay == True:
                    self.parameters["delay"][i] = nn.parameters["delay"]

                if self.is_energy == True:
                    self.parameters["energy"][i] = nn.parameters["energy"]
                    if self.is_energy_battery == True:
                        self.parameters["energy_battery"][i] = nn.parameters["energy_battery"]
                
                # 2 - Transfer Population parameters to Genome
                if is_sync_population_to_genome == True:
                    nn.parameters["voltage"] = self.parameters["voltage"][i]
                    nn.parameters["threshold"] = self.parameters["threshold"][i]
                    nn.parameters["tau"] = self.parameters["tau"][i]
                    nn.parameters["constant_current"] = self.parameters["constant_current"][i]

                    if self.is_refractory == True:
                        nn.parameters["refractory"] = self.parameters["refractory"][i]

                    nn.parameters["weight"] = self.parameters["weight"][i]

                    if self.is_delay == True:
                        nn.parameters["delay"] = self.parameters["delay"][i]
                    
                    if self.is_energy == True:
                        nn.parameters["energy"] = self.parameters["energy"][i]
                        if self.is_energy_battery == True:
                            nn.parameters["energy_battery"] = self.parameters["energy_battery"][i]

            else: # ANN
                # 1 - Transfer Genome parameters to Population
                self.parameters["bias"][i] = nn.parameters["bias"]
                self.parameters["weight"][i] = nn.parameters["weight"]

                # 2 - Transfer Population parameters to Genome
                if is_sync_population_to_genome:
                    nn.parameters["bias"] = self.parameters["bias"][i]
                    nn.parameters["weight"] = self.parameters["weight"][i]

            # 3 - Get Topology status
            if self.is_dynamic_topology == True:
                self.parameters["neurons_status"][i] = nn.neurons_status
                self.parameters["synapses_status"][i] = nn.synapses_status
            else:
                self.parameters["neurons_status"][0] = nn.neurons_status
                self.parameters["synapses_status"][0] = nn.synapses_status
            
            # 4 - Add to running and reproducing population
            self.population_running[genome.population_idx] = genome
            self.population_reproduction[genome.id] = genome

    def sync_population_to_genomes(self) -> None:
        for i, genome in enumerate(self.population.values()):
            nn:NN = genome.nn
            genome.population_idx = i
            genome.sync_with_population_func = self.sync_one_genome_to_population

            # 1 - Transfer Population parameters to Genome
            if self.is_SNN == True: # SNN
                nn.parameters["voltage"] = self.parameters["voltage"][i]
                nn.parameters["threshold"] = self.parameters["threshold"][i]
                nn.parameters["tau"] = self.parameters["tau"][i]
                nn.parameters["constant_current"] = self.parameters["constant_current"][i]

                if self.is_refractory == True:
                    nn.parameters["refractory"] = self.parameters["refractory"][i]

                nn.parameters["weight"] = self.parameters["weight"][i]

                if self.is_delay == True:
                    nn.parameters["delay"] = self.parameters["delay"][i]
                
                if self.is_energy == True:
                    nn.parameters["energy"] = self.parameters["energy"][i]
                    if self.is_energy_battery == True:
                        nn.parameters["energy_battery"] = self.parameters["energy_battery"][i]
            else: # ANN
                nn.parameters["bias"] = self.parameters["bias"][i]
                nn.parameters["weight"] = self.parameters["weight"][i]

    def sync_one_genome_to_population(self, genome:Genome_NN) -> None:
        if genome.population_idx is None: raise ValueError("The genome has not been added to the population")

        nn:NN = genome.nn
        i:int = genome.population_idx

        # 1 - Transfer Genome parameters to Population
        if self.is_SNN == True: # SNN
            self.parameters["voltage"][i] = nn.parameters["voltage"]
            self.parameters["threshold"][i] = nn.parameters["threshold"]
            self.parameters["tau"][i] = nn.parameters["tau"]
            self.parameters["constant_current"][i] = nn.parameters["constant_current"]

            if self.is_refractory == True:
                self.parameters["refractory"][i] = nn.parameters["refractory"]

            self.parameters["weight"][i] = nn.parameters["weight"]

            if self.is_delay == True:
                self.parameters["delay"][i] = nn.parameters["delay"]
            
            if self.is_energy == True:
                self.parameters["energy"][i] = nn.parameters["energy"]
                if self.is_energy_battery == True:
                    self.parameters["energy_battery"][i] = nn.parameters["energy_battery"]

        else: # ANN
            self.parameters["bias"][i] = nn.parameters["bias"]
            self.parameters["weight"][i] = nn.parameters["weight"]
        
        # 3 - Get Topology status
        if self.is_dynamic_topology == True:
            self.parameters["neurons_status"][i] = nn.neurons_status
            self.parameters["synapses_status"][i] = nn.synapses_status

    def sync_topology_status(self) -> None:   
        if self.is_first_sync_topology == True or self.is_dynamic_topology == True:
            self.parameters["neurons_actives_indexes"]              = np.where(np.sum(self.parameters["neurons_status"], axis=0) > 0)[0].astype(np.int32)
            self.parameters["hidden_neurons_actives_indexes"]       = np.array(np.where(np.any(self.parameters["neurons_status"][:, self.nb_inputs + self.nb_outputs:], axis=0)), dtype=np.int32)[0] + (self.nb_inputs + self.nb_outputs)
            self.parameters["synapses_actives_indexes"]             = np.array(np.where(np.any(self.parameters["synapses_status"], axis=0)), dtype=np.int32)
            self.is_first_sync_topology = False
        # self.parameters["hidden_synapses_actives_indexes"]    = np.array(np.where(np.any(self.parameters["synapses_status"][:, self.nb_inputs + self.nb_outputs:], axis=0)))
        # self.parameters["hidden_synapses_actives_indexes"][0] += self.nb_inputs + self.nb_outputs

    def sync_population_to_run(self) -> None:
        self.population = self.population_running
    
    def sync_population_to_reproduction(self) -> None:
        self.population = self.population_reproduction

    def __init_NN(self, nn:NN = None) -> None:
        if self.is_population_initialized == True: return
        self.is_population_initialized = True

        if nn is None:
            if self.population is None or len(self.population) == 0: raise ValueError("Impossible to initialize the Population, the population is empty or nn is not provided")
            first_key:int = next(iter(self.population))
            self.nb_population:int = len(self.population)
            nn:NN = self.population[first_key].nn

        self.network_type:str = nn.network_type
        self.nb_neurons:int = nn.nb_neurons
        self.nb_neurons_population:int = self.nb_population * self.nb_neurons
        self.nb_inputs:int = nn.nb_inputs
        self.nb_outputs:int = nn.nb_outputs
        self.nb_outputs_original:int = nn.nb_outputs_original
        self.nb_hiddens:int = nn.nb_hiddens
        self.input_idx:np.ndarray = nn.inputs["neurons_indexes"]
        self.output_idx:np.ndarray = nn.outputs["neurons_indexes"]

        self.population_genome_ids:np.ndarray = np.arange(self.pop_size, dtype=np.int32)
        self.population_genome_ids_list:List[int] = self.population_genome_ids.tolist()

        if self.network_type == "SNN":   self.__init_SNN(nn)

        elif self.network_type == "ANN": self.__init_ANN()

        else: raise Exception("Network type '", self.network_type,"' type not found, the available types are: 'SNN', 'ANN'")

        # Topology
        if self.is_dynamic_topology == True:
            self.parameters["neurons_status"] = np.zeros((self.nb_population, self.nb_neurons), dtype=np.bool_)
            self.parameters["synapses_status"] = np.zeros((self.nb_population, self.nb_neurons, self.nb_neurons), dtype=np.bool_)
        else:
            self.parameters["neurons_status"] = np.zeros((1, self.nb_neurons), dtype=np.bool_)
            self.parameters["synapses_status"] = np.zeros((1, self.nb_neurons, self.nb_neurons), dtype=np.bool_)
        
    def __init_SNN(self, nn:NN) -> None:

        self.is_refractory:bool = nn.is_refractory
        self.is_delay:bool = nn.is_delay
        self.is_energy:bool = nn.is_energy
        self.is_energy_battery:bool = nn.is_energy_battery
        if self.is_energy == True: self.energy_length:int = nn.energy_length

        # Neurons Parameters
        # Voltages
        self.parameters["voltage"] = np.zeros((self.nb_population, self.nb_neurons), dtype=np.float32)

        # Thresholds
        self.parameters["threshold"] = np.zeros((self.nb_population, self.nb_neurons), dtype=np.float32)

        # Tau (decay/leak)
        self.parameters["tau"] = np.zeros((self.nb_population, self.nb_neurons), dtype=np.float32)

        # Constant current
        self.parameters["constant_current"] = np.zeros((self.nb_population, self.nb_neurons), dtype=np.float32)

        # Refractory
        if self.is_refractory == True:
            self.parameters["refractory"] = np.zeros((self.nb_population, self.nb_neurons), dtype=np.float32)

        # Synapses Parameters
        # Weights
        self.parameters["weight"] = np.zeros((self.nb_population, self.nb_neurons, self.nb_neurons), dtype=np.float32)

        # Delay
        if self.is_delay == True:
            self.parameters["delay"] = np.zeros((self.nb_population, self.nb_neurons, self.nb_neurons), dtype=np.float32)


        # Energy
        if self.is_energy == True:
            self.parameters["energy"] = np.zeros((self.nb_population, self.energy_length, self.nb_neurons), dtype=np.float32)
            self.energy_size:int = self.energy_length*self.nb_neurons
            if self.is_energy_battery == True:
                self.parameters["energy_battery"] = np.zeros((self.nb_population, self.nb_neurons), dtype=np.float32)
                self.is_energy_battery_size:int = self.parameters["energy_battery"].size

    def __init_ANN(self):
        # Neurons Parameters
        # bias
        self.parameters["bias"] = np.zeros((self.nb_population, self.nb_neurons), dtype=np.float32)

        # Synapses Parameters
        # Weights
        self.parameters["weight"] = np.zeros((self.nb_population, self.nb_neurons, self.nb_neurons), dtype=np.float32)



    def update_info(self, optimization_type:str = None) -> None:
        self.reset()
        if optimization_type is not None:
            if optimization_type not in ["maximize", "minimize", "closest_to_zero"]: raise ValueError("optimization_type must be 'maximize', 'minimize' or 'closest_to_zero'")
            self.optimization_type = optimization_type

        nb_fitness_mean:int = 0
        # 0 - Get current fitness and topology info of the population
        for index, genome in enumerate(self.population.values()):
            nn:NN = genome.nn

            # 1 - Fitness  
            genome_score:float = genome.fitness.score if self.optimization_type != "closest_to_zero" else abs(genome.fitness.score)
            # 1.1 - Fitness max
            if index == 0 or genome_score > self.fitness.max:
                self.fitness.max = genome_score
                if self.optimization_type == "maximize":
                    self.best_genome = genome            
            # 1.2 - Fitness min
            if index == 0 or genome_score < self.fitness.min:
                self.fitness.min = genome_score
                if self.optimization_type in ["minimize", "closest_to_zero"]:
                    self.best_genome = genome
            # 1.3 - Fitness mean
            if genome_score != -np.inf and genome_score != np.inf:
                self.fitness.mean += genome_score
                nb_fitness_mean += 1

            # 2 - Topology
            # 2.1 - Neurons
            nb_neurons:int = len(nn.hiddens["neurons_indexes_active"])
            if nb_neurons > self.neurons_max:
                self.neurons_max = nb_neurons
            if nb_neurons < self.neurons_min:
                self.neurons_min = nb_neurons
            self.neurons_mean += nb_neurons

            # 2.2 - Synapses
            nb_synapses:int = len(nn.synapses_actives_indexes[0])
            if nb_synapses > self.synapses_max:
                self.synapses_max = nb_synapses
            if nb_synapses < self.synapses_min:
                self.synapses_min = nb_synapses
            self.synapses_mean += nb_synapses 

        # 3 - Update mean
        self.size:int = len(self.population)
        if self.size > 0 and nb_fitness_mean > 0:
            self.fitness.mean /= nb_fitness_mean
            self.neurons_mean /= self.size
            self.synapses_mean /= self.size

        # 4 - Fitness best score and history
        self.fitness.score = self.fitness.max if self.optimization_type == "maximize" else self.fitness.min
        self.fitness.history_best.append(self.fitness.score)
        self.fitness.history_mean.append(self.fitness.mean)
    
    def reset(self):
        self.fitness.reset()
        self.neurons_max:int = 0
        self.neurons_mean:int = 0
        self.neurons_min:int = 0

        self.synapses_max:int = 0
        self.synapses_mean:int = 0
        self.synapses_min:int = 0



    def first_generation(self, size:int=None, extra_info:Dict[str, Any] = None, is_same_fixed_weight:bool = False) -> None:
        self.is_first_generation = False
        population_idx:int = 0

        self.population_running = {}
        self.population_reproduction = {}
        if size is not None: self.pop_size = size
        while len(self.population) < self.pop_size:
            new_genome:Genome_NN = Genome_NN(get_new_genome_id(), self.config["Genome_NN"], self.attributes_manager)
            # new_genome:Genome_NN = Genome_NN(population_idx, self.config["Genome_NN"], self.attributes_manager)
            population_idx += 1
            new_genome.nn.set_arbitrary_parameters() # Parameters are set from the attributes_manager which contains information from your config file
            self.population[new_genome.id] = new_genome

            
            if (self.genome_core == None and len(self.population) > 0): # Get the first genome and init modules
                self.genome_core:Genome_NN = new_genome

                self.neuron_parameters_size:int = self.genome_core.nn.nb_neurons
                self.synapse_parameters_size:int = self.genome_core.nn.synapses_actives_indexes[0].size
                self.layer_parameters_size:int = len(self.genome_core.nn.architecture_layers)
                self.synapses_actives_indexes:np.ndarray = self.genome_core.nn.synapses_actives_indexes

                self.init_combinatorial(self.config_path_file)
                self.init_action_dynamic_RL(self.config_path_file)
                self.init_energy(self.config_path_file)
                fixed_weights:np.ndarray = self.genome_core.nn.parameters["weight"][self.genome_core.nn.synapses_actives_indexes]

                if self.is_dynamic_topology == False:
                    self.parameters["neurons_status"]  = np.array([self.genome_core.nn.neurons_status])
                    self.parameters["synapses_status"] = np.array([self.genome_core.nn.synapses_status])

            if is_same_fixed_weight == True: # Set the same fixed weight for all genomes (need to find a better way to do this)
                new_genome.nn.parameters["weight"][new_genome.nn.synapses_actives_indexes] = fixed_weights

            if extra_info is not None:
                if "weight_random_fixed" in extra_info:
                    new_genome.nn.parameters["weight"][new_genome.nn.synapses_actives_indexes] = extra_info["weight_random_fixed"] # need to find a better way to do this


        # Check if the parameters to update are correct
        for param in self.params_to_update:
            if param not in self.attributes_manager.parameters_all_names or param not in self.genome_core.nn.parameters:
                raise Exception(f"Parameter \"{param}\" not found in the parameters_all_names or in the genome parameters")
        if len(self.params_to_update) == 0: raise Exception("params_to_update is empty")
        self.init_param_size()

        self.population_running = self.population
        self.population_reproduction = self.population

    def init_param_size(self) -> None:
        self.parameters_size:int = 0

        for param in self.params_to_update:
            
            # 1 - add Neuron parameters
            if param in self.attributes_manager.parameters_neuron_names:
                if param == "energy":
                    self.parameters_size += self.neuron_parameters_size * self.energy_length
                else:
                    self.parameters_size += self.neuron_parameters_size
            # 2 - add Synapse parameters
            elif param in self.attributes_manager.parameters_synapse_names:
                self.parameters_size += self.synapse_parameters_size


        end_text:str = ""
        if self.is_combinatorial_modulo: 
            self.parameters_size += 1
            end_text = " with combinatorial_modulo"
        
        if self.is_action_dynamic_RL_evolution:
            if self.is_action_population_elite_evolution == True: # population/elite evolution
                self.parameters_size += 1
            else:                                  # Evolution
                self.parameters_size += self.genome_core.nn.nb_outputs_original*2 # one time for the max bound action space and one time for the min bound action space
            end_text = " with action_dynamic_RL"

        print("EvoSax-Total: parameters_size", self.parameters_size, end_text)


    def init_combinatorial(self, config_path_file:str) -> None:
        self.is_combinatorial_modulo:bool = False
        if self.is_SNN == False: return
        if (self.config["Runner_Info"]["encoder"] == "combinatorial"):
            self.config.update(TOOLS.config_function(config_path_file, ["Combinatorial_Encoder"]))
            print("combinatorial_filter", self.config["Combinatorial_Encoder"]["combinatorial_filter"])
            self.is_combinatorial_modulo:bool = True if all(word in self.config["Combinatorial_Encoder"]["combinatorial_filter"] for word in ["modulo", "dynamic"]) else False
            print("is_combinatorial_modulo", self.is_combinatorial_modulo)
            self.combinatorial_ocillator_noise_speed:float = float(self.config["Combinatorial_Encoder"]["combinatorial_ocillator_noise"])
            self.combinatorial_ocillator_noise_decay:float = float(self.config["Combinatorial_Encoder"]["combinatorial_ocillator_noise_decay"]) 
            self.combinatorial_modulo:np.ndarray = np.zeros((self.pop_size), dtype=np.float32)

    def init_action_dynamic_RL(self, config_path_file:str) -> None:
        self.is_action_dynamic_RL_evolution:bool = False
        self.action_dynamic_type:int = 0
        if "action_type" in self.config["Runner_Info"] and "evolution" in self.config["Runner_Info"]["action_type"]:
            self.is_action_dynamic_RL_evolution:bool = True
            action_type_config:str = self.config["Runner_Info"]["action_type"]
            self.is_action_abs:bool = "abs" in action_type_config
            if ("population" in action_type_config or "elite" in action_type_config) and "evolution" in action_type_config: 
                self.action_dynamic_type:int = 5 if "elite" in action_type_config else 4
            elif "population"  in action_type_config: self.action_dynamic_type:int = 1
            elif "elite"       in action_type_config: self.action_dynamic_type:int = 2
            elif "evolution"   in action_type_config: 
                self.action_dynamic_type:int = 3
                self.is_action_abs:bool = "abs" in action_type_config
            else: raise Exception("The action dynamic type is not specified in the config file, option are: population, elite, evolution -> e.g action_dynamic_population, action_dynamic_elite_0.1, action_dynamic_evolution, etc....")
            self.is_action_population_elite:bool = self.action_dynamic_type in [1, 2, 4, 5]
            self.is_action_population_elite_evolution:bool = self.action_dynamic_type in [4, 5]


    def init_energy(self, config_path_file:str) -> None:
        if ("energy" not in self.params_to_update and "energy" in self.config["Runner_Info"]["neuron_model"]) or ("energy" in self.params_to_update and "energy" not in self.config["Runner_Info"]["neuron_model"]): raise Exception("energy parameter must be updated with Energy_network, add energy to params_to_update or disable Energy in config file")
        self.is_energy:bool = True if "energy" in self.config["Runner_Info"]["neuron_model"] else False
        if self.is_energy == True:
            self.config.update(TOOLS.config_function(config_path_file, ["Energy_Network"]))
            self.energy_length:int = int(self.config["Energy_Network"]["energy_length"])
            self.energy_reset_value:float = float(self.config["Energy_Network"]["energy_init"])
        
    def update_modules(self, param_index:int, sigma_index:int, optimizer_params:np.ndarray, optimizer_sigma:np.ndarray, generation:int) -> Tuple[int, int]:

        # print("Combinatorial sigma", self.state.sigma[0], "shape", self.state.sigma.shape)
        # TEST: keep combinatorial modulo sigma higher
        # if self.state.sigma[0] < 0.12: self.state.sigma[0] = 0.12
        self.ocillator_noise:float = TOOLS.oscillation(generation, speed=self.combinatorial_ocillator_noise_speed)
        self.combinatorial_ocillator_noise_speed *= self.combinatorial_ocillator_noise_decay

        # if self.is_combinatorial_modulo: param_index = self.update_combinatorial_modulo_population(param_index, optimizer_params)
        for i, genome in enumerate(self.population.values()):
            param_index = 0
            if self.is_combinatorial_modulo:        param_index, sigma_index = self.update_combinatorial_modulo_genome(genome, param_index, optimizer_params[i], sigma_index, optimizer_sigma)
            if self.is_action_dynamic_RL_evolution: param_index, sigma_index = self.update_action_dynamic_RL(genome, param_index, optimizer_params[i], sigma_index, optimizer_sigma)

        return param_index, sigma_index

    def update_combinatorial_modulo_genome(self, genome:Genome_NN, param_index:int, optimizer_params:np.ndarray, sigma_index:int, sigma:np.ndarray) -> Tuple[int, int]:
        genome.combinatorial_modulo = optimizer_params[param_index] + self.ocillator_noise
        genome.combinatorial_modulo_sigma = sigma[sigma_index]
        genome.combinatorial_modulo_ocillator_noise = self.ocillator_noise
        self.combinatorial_modulo[genome.population_idx] = genome.combinatorial_modulo
        param_index += 1
        sigma_index += 1
        return param_index, sigma_index

    def update_combinatorial_modulo_population(self, param_index:int, optimizer_params:np.ndarray, sigma:float) -> int:
        self.combinatorial_modulo = optimizer_params[param_index + self.pop_size] + self.ocillator_noise
        self.combinatorial_modulo_sigma = sigma
        self.combinatorial_modulo_ocillator_noise = self.ocillator_noise
        param_index += 1
        return param_index

    def update_action_dynamic_RL(self, genome:Genome_NN, param_index:int, optimizer_params:np.ndarray, sigma_index:int, sigma:np.ndarray) -> Tuple[int, int]:
        if self.is_action_population_elite_evolution: # population/elite evolution
            if self.is_action_abs == True: genome.action_std_range_factor = float(np.clip(np.abs(2 + optimizer_params[param_index]), 0.5, 4.0)) # clip to keep the std range factor in a reasonable range
            else:                          genome.action_std_range_factor = float(np.clip(2 + optimizer_params[param_index], 0.5, 4.0))
            # print("Action std range factor", genome.action_std_range_factor)
            param_index += 1
            return param_index, sigma_index
        
        # Else -> evolution only
        # 1 - Get the action parameters MAX
        if self.is_action_abs == True: genome.action_offset_max = np.abs(optimizer_params[param_index:self.nb_outputs_original], dtype=np.float32)
        else:                          genome.action_offset_max = optimizer_params[param_index:self.nb_outputs_original]

        # 2 - Get the action parameters MIN
        if self.is_action_abs == True: genome.action_offset_min = np.abs(optimizer_params[param_index+self.nb_outputs_original:self.nb_outputs_original*2], dtype=np.float32)
        else:                          genome.action_offset_min = optimizer_params[param_index+self.nb_outputs_original:self.nb_outputs_original*2]

        param_index += self.nb_outputs_original*2
        
        # # 3 - Get the SIGMA of action MAX
        # genome.action_offset_sigma_max = sigma[sigma_index:sigma_index+self.nb_outputs_original]
        
        # # 4 - Get the SIGMA of action MIN
        # genome.action_offset_sigma_min = sigma[sigma_index+self.nb_outputs_original:sigma_index+self.nb_outputs_original*2]

        # sigma_index += self.nb_outputs_original*2

        return param_index, sigma_index

    def get_sub_population_from_indexes(self, population_genome_ids:np.ndarray) -> "Population_NN":
        sub_population:Population_NN = Population_NN(-1, self.algo_name_config, self.config_path_file, self.attributes_manager)
        sub_population.init_sub_population_from_another_population(self, population_genome_ids)
        return sub_population

    def init_sub_population_from_another_population(self, population:"Population_NN", population_genome_ids:np.ndarray):
        self.population_genome_ids = population_genome_ids
        self.nb_populations = self.population_genome_ids.size


        if self.is_population_initialized == False:
            self.is_population_initialized = True
            
            self.nb_population = population.nb_population
            self.network_type:str = population.network_type
            self.nb_neurons:int = population.nb_neurons
            self.nb_neurons_population:int = population.nb_neurons_population
            self.nb_inputs:int = population.nb_inputs
            self.nb_outputs:int = population.nb_outputs
            self.nb_hiddens:int = population.nb_hiddens
            self.input_idx:np.ndarray = population.input_idx
            self.output_idx:np.ndarray = population.output_idx

        if self.is_SNN == True:    self.__init_SNN_from_population(population, population_genome_ids)

        elif self.is_SNN == False: self.__init_ANN_from_population(population, population_genome_ids)

        else: raise Exception("Network type '", self.network_type,"' type not found, the available types are: 'SNN', 'ANN'")

        # Topology
        if self.is_dynamic_topology == True:
            self.parameters["neurons_status"] = population.parameters["neurons_status"][population_genome_ids]
            self.parameters["synapses_status"] = population.parameters["synapses_status"][population_genome_ids]
        else:
            self.parameters["neurons_status"] = population.parameters["neurons_status"]
            self.parameters["synapses_status"] = population.parameters["synapses_status"]



    def __init_SNN_from_population(self, population:"Population_NN", population_genome_ids:np.ndarray) -> None:
        self.is_refractory:bool = population.is_refractory
        self.is_delay:bool = population.is_delay
        self.is_energy:bool = population.is_energy
        self.is_energy_battery:bool = population.is_energy_battery
        if self.is_energy == True: self.energy_length:int = population.energy_length

        # Neurons Parameters
        # Voltages
        self.parameters["voltage"] = population.parameters["voltage"][population_genome_ids]

        # Thresholds
        self.parameters["threshold"] = population.parameters["threshold"][population_genome_ids]

        # Tau (decay/leak)
        self.parameters["tau"] = population.parameters["tau"][population_genome_ids]

        # Constant current
        self.parameters["constant_current"] = population.parameters["constant_current"][population_genome_ids]

        # Refractory
        if self.is_refractory == True:
            self.parameters["refractory"] = population.parameters["refractory"][population_genome_ids]

        # Synapses Parameters
        # Weights
        self.parameters["weight"] = population.parameters["weight"][population_genome_ids]

        # Delay
        if self.is_delay == True:
            self.parameters["delay"] = population.parameters["delay"][population_genome_ids]


        # Energy
        if self.is_energy == True:
            self.parameters["energy"] = population.parameters["energy"][population_genome_ids]
            self.energy_size:int = self.energy_length*self.nb_neurons
            if self.is_energy_battery == True:
                self.parameters["energy_battery"] = population.parameters["energy_battery"][population_genome_ids]
                self.is_energy_battery_size:int = self.parameters["energy_battery"].size
        

    def __init_ANN_from_population(self, population:"Population_NN", population_genome_ids:np.ndarray) -> None:
        # Neurons Parameters
        # bias
        self.parameters["bias"] = population.parameters["bias"][population_genome_ids]

        # Synapses Parameters
        # Weights
        self.parameters["weight"] = population.parameters["weight"][population_genome_ids]

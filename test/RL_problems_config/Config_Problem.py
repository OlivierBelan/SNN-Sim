from problem.RL.ENVIRONNEMENT import Environment
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator import TOOLS
from typing import  Dict, Any, Tuple
import numpy as np
import re

class Config_Problem():
    def __init__(self, name:str, nb_inputs:int, nb_outputs:int, config_path:str, obs_max_init_value:float = 5.0, obs_min_init_value:float = -5.0, termination_fitness:float = None, auto_obersvation:bool = False, auto_obersvation_ratio:float=0.01):
        self.name:str = name
        self.nb_inputs:int = nb_inputs
        self.nb_outputs:int = nb_outputs
        self.config_path:str = config_path
        self.termination_fitness:float = termination_fitness
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path, ["Genome_NN", "NEURO_EVOLUTION", "Runner_Info"])
        self.observation_max_global:np.ndarray = np.full(nb_inputs, obs_max_init_value, dtype=np.float64) # 5 is arbitrary
        self.observation_min_global:np.ndarray = np.full(nb_inputs, obs_min_init_value, dtype=np.float64) # -5 is arbitrary
        self.observation_max_global_init:np.ndarray = np.full(nb_inputs, obs_max_init_value, dtype=np.float64) # 5 is arbitrary
        self.observation_min_global_init:np.ndarray = np.full(nb_inputs, obs_min_init_value, dtype=np.float64) # -5 is arbitrary
        self.observation_scaling:np.ndarray = np.full(nb_inputs, 1.0, dtype=np.float64) 
        self.auto_obersvation:bool = auto_obersvation
        self.auto_obersvation_ratio:float = auto_obersvation_ratio
        self.observation_update_type:str = self.config["Runner_Info"]["observation_type"]
        self.is_raw_update_observation:bool = "raw" in self.observation_update_type
        self.is_abs_update_observation:bool = "abs" in self.observation_update_type
        self.is_scaling_update_observation:bool = "scaling" in self.observation_update_type
        self.is_scaling_dynamic:bool = False
        self.scaling_type:int = 0
        self.is_render:bool = False
        self.is_action_population_elite:bool = False
        self.is_action_population_elite_evolution:bool = False
        self.is_SNN:bool = self.config["Genome_NN"]["network_type"] == "SNN"


        if self.is_scaling_update_observation == True:
            if "dynamic" in self.observation_update_type: self.is_scaling_dynamic = True
            else: self.is_scaling_dynamic = False

            if "1" in self.observation_update_type: self.scaling_type = 1
            elif "2" in self.observation_update_type: self.scaling_type = 2
            elif "3" in self.observation_update_type: self.scaling_type = 3
            else: raise Exception("The scaling type is not specified in the config file, option are: 1, 2, 3 -> e.g scaling1, scaling2, scaling3 or scaling_1, etc....")

        # print("is_raw_update_observation:", self.is_raw_update_observation, "is_abs_update_observation:", self.is_abs_update_observation, "is_scaling_update_observation:", self.is_scaling_update_observation)
        if self.is_raw_update_observation == False and self.is_abs_update_observation == False and self.is_scaling_update_observation == False: raise Exception("The observation_type in the config file is not implemented option are: raw, abs, raw_scaling, abs_scaling")

        self.action_update_type:str  = self.config["Runner_Info"]["action_type"]
        self.action_max_ratio:float  = float(np.clip(float(self.config["Runner_Info"]["action_max_ratio"]), 0.0, 1.0))
        self.action_min_ratio:float  = float(np.clip(float(self.config["Runner_Info"]["action_min_ratio"]), 0.0, 1.0))
        self.is_action_dynamic:bool  = "dynamic" in self.action_update_type
        self.action_dynamic_type:int = 0
        
        if self.is_action_dynamic == True:
            if ("population" in self.action_update_type or "elite" in self.action_update_type) and "evolution" in self.action_update_type: 
                self.action_dynamic_type:int = 5 if "elite" in self.action_update_type else 4
                if self.action_dynamic_type == 5: self.action_elite_ratio:float = float(re.search(r"(\d+\.\d+)", self.action_update_type).group(1))

            elif "population" in self.action_update_type: self.action_dynamic_type:int = 1
            elif "elite"    in self.action_update_type: 
                self.action_dynamic_type:int = 2
                self.action_elite_ratio:float = float(re.search(r"(\d+\.\d+)", self.action_update_type).group(1))
            elif "evolution" in self.action_update_type: self.action_dynamic_type:int = 3
            else: raise Exception("The action dynamic type is not specified in the config file, option are: population, elite, evolution -> e.g action_dynamic_population, action_dynamic_elite_0.1, action_dynamic_evolution, etc....")
            self.action_absolute_min:np.ndarray = None
            self.action_absolute_max:np.ndarray = None
            self.action_init_max:np.ndarray = None
            self.action_init_min:np.ndarray = None
            self.action_local_max:np.ndarray = None
            self.action_local_min:np.ndarray = None

            self.action_max_mean:np.ndarray = 0.0
            self.action_min_mean:np.ndarray = 0.0
            self.action_max_std:np.ndarray = 2.0
            self.action_min_std:np.ndarray = 2.0

            self.is_action_population_elite:bool = self.action_dynamic_type in [1, 2, 4, 5]
            self.is_action_population_elite_evolution:bool = self.action_dynamic_type in [4, 5]

        nb_inputs_config:int = int(self.config["Genome_NN"]["inputs"])
        nb_outputs_config:int = int(self.config["Genome_NN"]["outputs"])
        if nb_inputs_config != nb_inputs:
            raise Exception("The number of inputs in the config file (", nb_inputs_config ,") is different from the number of inputs in the problem (", nb_inputs,")")
        if nb_outputs_config != nb_outputs:
            raise Exception("The number of outputs in the config file (", nb_outputs_config ,") is different from the number of outputs in the problem (", nb_outputs,")")
        
    def get_env(self) -> Environment:
        raise NotImplementedError("get_env() not implemented")
    
    def encoding_observation_to_input_network(self, observation: np.ndarray, observation_max:np.ndarray, observation_min:np.ndarray) -> np.ndarray:
        raise NotImplementedError("encoding_observation_to_input_network() not implemented")
    
    def decoding_output_network_to_action(self, action: np.ndarray) -> int:
        raise NotImplementedError("decoding_output_network_to_action() not implemented")

    def fitness_step(self, genome, episode:int, info:Dict[str, Any]) -> None:
        raise NotImplementedError("fitness_step() not implemented")

    def fitness_end(self, genome, episode:int, info:Dict[str, Any]) -> None:
        raise NotImplementedError("fitness_end() not implemented")
    
    def update_observation_min_max(self, obs_max:np.ndarray, obs_min:np.ndarray) -> None:
        self.observation_max_global = obs_max
        self.observation_min_global = obs_min


        if self.is_scaling_update_observation == True: 

            # Scaling 1
            if self.scaling_type == 1:
                self.observation_scaling:np.ndarray = ((np.abs(self.observation_max_global_init / self.observation_max_global) + np.abs(self.observation_min_global_init / self.observation_min_global)) / 2)

            # Scaling 2
            elif self.scaling_type == 2:
                self.observation_scaling:np.ndarray = (((self.observation_max_global - self.observation_max_global_init) / self.observation_max_global_init) + ((self.observation_min_global - self.observation_min_global_init) / self.observation_min_global_init) / 2)

            # Scaling 3
            elif self.scaling_type == 3:
                # obs_old_range = self.observation_max_global_init - self.observation_min_global_init
                # obs_new_range = self.observation_max_global - self.observation_min_global
                # self.observation_scaling = obs_new_range / obs_old_range
                self.observation_scaling:np.ndarray = (self.observation_max_global - self.observation_min_global) / (self.observation_max_global_init - self.observation_min_global_init)


            if self.is_scaling_dynamic == True: # for scaling
                self.observation_max_global_init = self.observation_max_global.copy()
                self.observation_min_global_init = self.observation_min_global.copy()

    
    def update_action_min_max(self, population:Dict[int, Genome_NN], extra_info:Dict[str, Any]) -> None:

        if self.is_render == False and (self.action_dynamic_type == 1 or self.action_dynamic_type == 2 or self.action_dynamic_type == 4 or self.action_dynamic_type == 5): # Population Version or Elite Version (1, 2, 4, 5)
            action_local_max, action_local_min = self.update_action_min_max_with_std_range_factor(extra_info["action_max_step"], extra_info["action_min_step"], extra_info["genome_idx_ranked"])

        for genome in population.values():
            if   self.is_render == True and (self.action_dynamic_type == 1 or self.action_dynamic_type == 2):  # (RENDER) Population Version or Elite Version (1, 2)
                self.update_action_min_max_render(genome)

            elif self.is_render == False and (self.action_dynamic_type == 1 or self.action_dynamic_type == 2): # Population Version or Elite Version (1, 2)
                genome.action_local_max = action_local_max
                genome.action_local_min = action_local_min

            elif self.action_dynamic_type == 3:                                                                # Dynamic action evolution (3)
                self.update_action_min_max_evolution(genome)

            elif self.action_dynamic_type == 4 or self.action_dynamic_type == 5:                               # Population/Elite Evolution Version (4, 5)
                self.update_action_min_max_with_std_range_factor_evolution(genome)

            else:
                genome.action_local_max = self.action_init_max
                genome.action_local_min = self.action_init_min

    def update_action_min_max_with_std_range_factor(self, action_max_mean:np.ndarray, action_min_mean:np.ndarray, genome_idx_fitness_ranked:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_render == True: return
        if action_max_mean is None or action_min_mean is None: return self.action_init_max, self.action_init_min

        if self.action_dynamic_type == 2 or self.action_dynamic_type == 5: # Elite/Elite_Evoluton Version that count only the elite population
            nb_elite = int(len(genome_idx_fitness_ranked) * self.action_elite_ratio)
            genome_idx_fitness_ranked = genome_idx_fitness_ranked[:nb_elite]
            action_max_mean = action_max_mean[genome_idx_fitness_ranked]
            action_min_mean = action_min_mean[genome_idx_fitness_ranked]

        # 1 - Population Version or Elite Version
        if self.is_action_population_elite: # Population (1) or Elite Version (2) or Population/Elite Evolution Version (4, 5)
            # print("action_max_mean:", action_max_mean, "action_min_mean:", action_min_mean, "np.mean(action_max_mean)", np.mean(action_max_mean, axis=0), "np.mean(action_min_mean)", np.mean(action_min_mean, axis=0),"np.std(action_max_mean):", np.std(action_max_mean, axis=0), "np.std(action_min_mean):", np.std(action_min_mean, axis=0))
            alpha = 0.1
            self.action_max_mean = alpha * np.mean(action_max_mean, axis=0) + (1 - alpha) * self.action_max_mean # Exponential moving average
            self.action_min_mean = alpha * np.mean(action_min_mean, axis=0) + (1 - alpha) * self.action_min_mean # Exponential moving average
            self.action_max_std  = alpha * np.std( action_max_mean, axis=0) + (1 - alpha) * self.action_max_std  # Exponential moving average
            self.action_min_std  = alpha * np.std( action_min_mean, axis=0) + (1 - alpha) * self.action_min_std  # Exponential moving average
            
            k = 2.0  # (std_range_factor) arbitrary value that will be used to scale the action_std
            action_local_min:np.ndarray = self.action_min_mean - k * self.action_min_std # Max bound
            action_local_max:np.ndarray = self.action_max_mean + k * self.action_max_std # Min bound

            # Clip the action between the min and max
            action_local_min = np.clip(action_local_min, self.action_absolute_min, self.action_init_min)
            action_local_max = np.clip(action_local_max, self.action_init_max, self.action_absolute_max)
            # exit("from update_action_min_max in Config_Problem.py")
            return action_local_max, action_local_min

    def update_action_min_max_with_std_range_factor_evolution(self, genome:Genome_NN) -> None:

        if self.is_render == True:
            genome.action_local_min = genome.info["action_min_mean"] - genome.action_std_range_factor * genome.info["action_min_std"]
            genome.action_local_max = genome.info["action_max_mean"] + genome.action_std_range_factor * genome.info["action_max_std"]
        else:
            genome.action_local_min = self.action_min_mean - genome.action_std_range_factor * self.action_min_std
            genome.action_local_max = self.action_max_mean + genome.action_std_range_factor * self.action_min_std

        # Clip the action between the min and max
        genome.action_local_min = np.clip(genome.action_local_min, self.action_absolute_min, self.action_init_min)
        genome.action_local_max = np.clip(genome.action_local_max, self.action_init_max, self.action_absolute_max)
        # print("action_local_min:", genome.action_local_min, "\naction_local_max:", genome.action_local_max, "\naction_std_range_factor:", genome.action_std_range_factor, "\naction_max_mean:", self.action_max_mean, "\naction_min_mean:", self.action_min_mean, "\naction_max_std:", self.action_max_std, "\naction_min_std:", self.action_min_std)

    def update_action_min_max_evolution(self, genome:Genome_NN) -> None:
        # Keep in mind that action_offset_min/max will come as an absolute value, so there is no negative as it come
        genome.action_local_min =  np.clip((self.action_init_min - genome.action_offset_min), self.action_absolute_min, self.action_init_min)
        genome.action_local_max =  np.clip((self.action_init_max + genome.action_offset_max),  self.action_init_max, self.action_absolute_max)

    def update_action_min_max_render(self, genome:Genome_NN) -> None:
        if self.is_action_dynamic == True:
            # info = genome.info
            # self.action_local_min = info["action_local_min"]
            # self.action_local_max = info["action_local_max"]
            self.action_local_min = genome.action_local_min
            self.action_local_max = genome.action_local_max
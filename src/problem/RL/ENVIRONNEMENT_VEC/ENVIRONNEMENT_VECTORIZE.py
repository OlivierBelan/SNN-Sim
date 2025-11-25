
from typing import List, Dict, Any, Callable, Tuple
from evo_simulator.GENERAL.Genome import Genome_NN
import numpy as np

class Environment_Vec:
    def __init__(self, env_name:str, encoding_observation_to_network_input_function:Callable, decoding_output_network_to_action_function:Callable, fitness_step_function:Callable, fitness_end_function:Callable = None, update_observation_min_max_function:Callable = None):
        self.env_name:str = env_name
        self.encoding_observation_to_network_input_function:Callable = encoding_observation_to_network_input_function
        self.decoding_output_network_to_action_function:Callable = decoding_output_network_to_action_function
        self.fitness_step_function:Callable = fitness_step_function
        self.fitness_end_function:Callable = fitness_end_function
        self.update_observation_min_max_function:Callable = update_observation_min_max_function
        self.check_functions()
        self.nb_envs:int = None

        self.id:int = None
        self.observation:Any = None
        self.reward:float = None
        self.terminated:bool = False 
        self.truncated: bool = False
        self.extra_info:Dict[str, Any] = {}
        self.seed:int = None

        self.observation_history:List[np.ndarray] = []
        self.observation_max_history:np.ndarray = None
        self.observation_min_history:np.ndarray = None
        self.reward_history:List[float] = []
        self.action_history:List[np.ndarray] = []
        self.reward_cumulative:np.ndarray = None

    def check_functions(self) -> None:
        if self.encoding_observation_to_network_input_function == None: raise Exception("The encoding_observation_to_network_input_function function is not implemented")
        if self.decoding_output_network_to_action_function == None: raise Exception("The decoding_output_network_to_action_function function is not implemented")
        if self.fitness_step_function == None: raise Exception("The fitness_step_function function is not implemented")
        if self.fitness_end_function == None: raise Exception("The fitness_end_function function is not implemented")

    def update(self, action:np.ndarray) -> bool:
        '''
            Update the environment with the action and update the fitness object with the reward
            Return True if the episode is active, else False if the episode is terminated or truncated
        '''
        raise NotImplementedError

    def reset(self, seed:int = None):
        '''
            Reset the environment and set the seed if it is not None
        '''
        raise NotImplementedError

    def encoding_observation_to_network_input(self) -> np.ndarray:
        '''
            Return the spikes encoding of the observation
        '''
        raise NotImplementedError
        
    def fitness_end(self, genomes:Dict[int, Genome_NN], episodes:List[int]):
        '''
            Update the fitness object of each genome with the fitness_end_function
        '''
        if self.fitness_end_function is not None:
            self.fitness_end_function(genomes, episodes)
        else:
            raise NotImplementedError("The fitness_end_function function is not implemented")

    def close(self):
        raise NotImplementedError

    def copy(self) -> "Environment":
        raise NotImplementedError

class Environment_Manager_Vec:
    def __init__(self, environment_builer:Callable):
        self.environement_builder:Callable = environment_builer
        self.envs:List[Environment_Vec] = [] # List of environments
        self.seeds:List[int] = []
        self.name:str = environment_builer.name
        self.input_size:int = environment_builer.nb_inputs

        self.seeds:int = None
        self.nb_envs:int = None
        self.nb_episodes:int = None

        self.fitness_end_function:Callable = None
        self.update_observation_min_max_function:Callable = None

        self.final_rewards:np.ndarray = None
        self.final_obs_max:np.ndarray = None
        self.final_obs_min:np.ndarray = None
        
    def create_environments(self, nb_envs:int, seeds:List[int]) -> List[Environment_Vec]:

        nb_episode:int = len(seeds)
        if nb_episode == 0 or len(np.unique(seeds)) != len(seeds): raise ValueError("The number of seeds must be equal to the number of episodes and the seeds must be unique")
        self.nb_episodes = nb_episode
        self.seeds = seeds
        self.nb_envs = nb_envs

        if (len(self.envs) == len(seeds) and self.envs[0].nb_envs == nb_envs):
            # # 3 - Init some variables
            self.final_rewards:np.ndarray = np.zeros((self.nb_episodes, self.nb_envs))
            self.final_obs_max:np.ndarray = np.zeros((self.nb_episodes, self.input_size))
            self.final_obs_min:np.ndarray = np.zeros((self.nb_episodes, self.input_size))
            # self.reset(seeds) # I did comment it cause I do reset just afterwards in the loop
            return self.envs
        
        # 1 - Create the environments
        self.envs:List[Environment_Vec] = []
        for seed in seeds:
            self.envs.append(self.create_new_env(nb_envs))
        
        self.fitness_end_function = self.envs[0].fitness_end_function
        self.update_observation_min_max_function = self.envs[0].update_observation_min_max_function
        
        # 2 - Reset & Update environment seeds
        # self.reset(self.seeds) # (I did comment it cause I do reset just afterwards in the loop)

        # 3 - Init some variables
        self.final_rewards:np.ndarray = np.zeros((self.nb_episodes, self.nb_envs))
        self.final_obs_max:np.ndarray = np.zeros((self.nb_episodes, self.input_size))
        self.final_obs_min:np.ndarray = np.zeros((self.nb_episodes, self.input_size))

        return self.envs


    def encoding_observation_to_snn_input(self) -> np.ndarray:
        inputs_array:np.ndarray = np.zeros((self.nb_episodes, self.nb_envs, self.input_size))
        for i, env_vec in enumerate(self.envs):
            inputs_array[i] += env_vec.encoding_observation_to_network_input()
        return inputs_array

    def encoding_observation_to_ann_input(self) -> Dict[int, np.ndarray]:
        observations_dict:Dict[int, np.ndarray] = {}
        env_observation:List[np.ndarray] = []

        for env_id, envs_list in self.envs_dict.items():
            env_observation = []
            for j in range(len(envs_list)):
                env_observation.append(envs_list[j].encoding_observation_to_network_input())
            observations_dict[env_id] = np.array(env_observation)
        return observations_dict


    def update_environments(self, actions:np.ndarray, output_indexes:np.ndarray=None) -> bool:
        is_active:int = 0
        for i, env_vec in enumerate(self.envs):
            if output_indexes is not None: # For SNN only (cause actions dict contain spikes of all neurons, then we need to select only the output neurons spikes)
                is_active += env_vec.update(actions[i][:, output_indexes])
            else:
                is_active += env_vec.update(actions[i][output_indexes])
        return is_active > 0 # Return True if at least one environment is active
        
    def fitness_end(self, genomes:Dict[int, Genome_NN]) -> Tuple[np.ndarray, np.ndarray]:
        if self.fitness_end_function is not None:
            self.get_final_observation_min_max()
            return self.fitness_end_function(genomes, self.get_final_reward(), self.final_obs_max, self.final_obs_min)
        else:
            raise NotImplementedError("The fitness_end_function function is not implemented")

    def get_final_reward(self) -> np.ndarray:
        for i, env_vec in enumerate(self.envs):
            self.final_rewards[i] = env_vec.reward_cumulative
        return self.final_rewards

    def get_final_observation_min_max(self) -> Tuple[np.ndarray, np.ndarray]:
        for i in range(self.nb_episodes):
            self.final_obs_max[i] = self.envs[i].observation_max_history
            self.final_obs_min[i] = self.envs[i].observation_min_history
        return self.final_obs_max, self.final_obs_min

    def update_observation_min_max(self, observation_max:np.ndarray, observation_min:np.ndarray) -> None:
        if self.update_observation_min_max_function is not None:
            for env_vec in self.envs:
                env_vec.update_observation_min_max_function(observation_max, observation_min)
        else:
            raise NotImplementedError("The update_observation_min_max_function function is not implemented")

    def reset(self, seeds:List[int]) -> None:
        # 1 - Check if the number of seeds is equal to the number of environments
        if len(np.unique(seeds)) != len(seeds): raise ValueError("The number of seeds must be unique")
        # 2 - Reset the environments with the seeds
        for i, env_vec in enumerate(self.envs):
            env_vec.seed = seeds[i]
            env_vec.reset(seeds[i])
        
    def create_new_env(self, nb_envs:int) -> Environment_Vec:
        new_env:Environment_Vec = self.environement_builder.get_env(nb_envs)
        return new_env
    
    # def __update_envs_dict_ids_with_genomes_ids(self, genomes_ids:List[int]) -> Dict[int, List[Environment]]:
    #     new_envs_dict:Dict[int, List[Environment]] = {}
    #     for index, envs_list in enumerate(self.envs_dict.values()):
    #         if index < len(genomes_ids):
    #             new_envs_dict[genomes_ids[index]] = envs_list
    #     return new_envs_dict

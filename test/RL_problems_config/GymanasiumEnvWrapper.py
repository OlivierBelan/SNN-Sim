
from evo_simulator.GENERAL.Fitness import Fitness
from evo_simulator.GENERAL.Genome import Genome_NN
from problem.RL.ENVIRONNEMENT import Environment
from problem.RL.GYM_ENV import Environment_Gym
from typing import List, Dict, Any, Tuple, Callable
from RL_problems_config.Config_Problem import Config_Problem
import gymnasium as gym

import numpy as np
import numba as nb

# @nb.njit(cache=True, fastmath=False, nogil=True)
def normalize_number(nb:float, max_value:float, min_value:float) -> float:
    res = (nb - min_value) / (max_value - min_value)
    if res > 1.0: return 1.0
    if res < 0.0: return 0.0
    return res





class EnvGym_Wrapper(Config_Problem):
    def __init__(self, name:str, config_path:str, nb_input:int, nb_output:int, build_env_func:Callable, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=True, auto_obersvation_ratio:float=0.01) -> None:
        self.build_env_func:Callable = build_env_func
        Config_Problem.__init__(self, name, nb_input, nb_output, config_path, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def get_env(self, render:bool=False) -> Environment_Gym:
        self.gym_env:gym.Env = self.build_env_func(render)

        # if hasattr(self.gym_env.action_space, "low"):
        self.action_absolute_min:np.ndarray = self.gym_env.action_space.low
        self.action_absolute_max:np.ndarray = self.gym_env.action_space.high
        self.action_init_max:np.ndarray = self.action_absolute_max * self.action_max_ratio
        self.action_init_min:np.ndarray = self.action_absolute_min * self.action_min_ratio
        self.action_nn_output_bound_max:np.ndarray = np.ones(self.nb_outputs, dtype=np.float32)
        self.action_nn_output_bound_min:np.ndarray = np.zeros(self.nb_outputs, dtype=np.float32)
        # else:
        #     self.action_init_max = None
        #     self.action_init_min = None

        # print("action absolute min: ", self.action_absolute_min, " action absolute max: ", self.action_absolute_max)
        # print("action init min: ", self.action_init_min, " action init max: ", self.action_init_max)
        # print("action nn output bound min: ", self.action_nn_output_bound_min, " action nn output bound max: ", self.action_nn_output_bound_max)
        # exit(0)

        return Environment_Gym(
            env_name=self.name,
            gym_env=self.gym_env,
            encoding_observation_to_network_input_function=self.encoding_observation_to_input_network,
            decoding_output_network_to_action_function=self.decoding_output_network_to_action, 
            fitness_step_function=self.fitness_step, 
            fitness_end_function=self.fitness_end,
            update_observation_min_max_function=self.update_observation_min_max,
            update_action_min_max_function=self.update_action_min_max,
            update_action_min_max_render_function=self.update_action_min_max_render,
            is_abs_update_observation=self.is_abs_update_observation
        )

    def encoding_observation_to_input_network(self, observation: np.ndarray, observation_max:np.ndarray, observation_min:np.ndarray) -> np.ndarray:
        observation_len:int = len(observation)
        if self.auto_obersvation == True and self.is_scaling_update_observation == True: observation += (observation * self.observation_scaling)
        # if self.auto_obersvation == True and self.is_scaling_update_observation == True: observation = self.observation_min_global_init + ((observation - self.observation_min_global) / (self.observation_max_global - self.observation_min_global)) * (self.observation_max_global_init - self.observation_min_global_init)
        for i in range(observation_len):
            if self.auto_obersvation == True:
                observation[i] = normalize_number(observation[i], self.observation_max_global[i], self.observation_min_global[i])
                #observation[i] = normalize_number(observation[i], self.observation_max_global_init[i], self.observation_min_global_init[i])
            else:
                observation[i] = normalize_number(observation[i], observation_max[i], observation_min[i])
                self.observation_max_global[i] = observation_max[i] # just to for the print in the end (used vs found)
                self.observation_min_global[i] = observation_min[i]
            if observation[i] > 1 or observation[i] < 0:
                 raise Exception("observation[i] > 1 or observation[i] < 0")
        return observation

    def rescale(self, value, min_input=0, max_input=1, min_output=-1, max_output=1):
        """Rescale a value from one range to another."""
        return min_output + (value - min_input) / (max_input - min_input) * (max_output - min_output)

    def decoding_output_network_to_action(self, action:np.ndarray) -> np.ndarray:        
        if self.is_SNN == True:
            return self.rescale(action, 0, 1, self.action_absolute_max, self.action_absolute_min)
        else:
            return self.rescale(action, -1, 1, self.action_absolute_max, self.action_absolute_min) # cause of the use of tanh activation function with ANN

    @staticmethod
    def fitness_step(genome:Genome_NN, episode:int, info:Dict[str, Any]) -> None:
        if info["terminated"] == True or info["truncated"] == True: return

        # 1 - update the fitness with the reward
        fitness_obj:Fitness = genome.fitness
        if fitness_obj.extra_info.get(episode) == None:
            fitness_obj.extra_info[episode] = info["reward"]
        else:
            fitness_obj.extra_info[episode] += info["reward"]
        

    def fitness_end(self, genomes:Dict[int, Genome_NN], episodes:List[int]) -> None:
        # 0 - fitness qui check le success

        # bis 1 - In case the observation is not known
        observation_global_max_history:List[np.ndarray] = []
        observation_global_min_history:List[np.ndarray] = []

        episode_score:List[float] = []
        if self.is_action_population_elite == True: # 1.1 - dynamic action population/elites
            fitnesses:dict =  {"best_episode_raw_score": [], "mean_episode_raw_score": [], "action_local_min": [], "action_local_max": [], "action_max_step": [], "action_min_step": []}
        else:
            fitnesses:dict =  {"best_episode_raw_score": [], "mean_episode_raw_score": [], "action_local_min": [], "action_local_max": []}

        for genome in genomes.values():
            fitness_obj:Fitness = genome.fitness
            fitness_obj.score:int = 0

            episode_score:List[float] = []
            for episode in episodes:
                fitness_obj.score += fitness_obj.extra_info[episode]
                episode_score.append(fitness_obj.extra_info[episode])
            # genome.info["best_episode_raw_score"] = max(episode_score)
            # genome.info["mean_episode_raw_score"] = np.mean(episode_score).astype(float)
            fitnesses["best_episode_raw_score"].append(max(episode_score))
            fitnesses["mean_episode_raw_score"].append(np.mean(episode_score).astype(float))
            fitnesses["action_local_min"].append(genome.action_local_min)
            fitnesses["action_local_max"].append(genome.action_local_max)

            # 2 - dynamic action population/elites
            if self.is_action_population_elite == True:                
                fitnesses["action_min_step"].append(genome.info["action_min_step"])
                fitnesses["action_max_step"].append(genome.info["action_max_step"])

                if self.is_action_population_elite_evolution == True:
                    fitnesses["action_max_mean"] = self.action_max_mean
                    fitnesses["action_min_mean"] = self.action_min_mean
                    fitnesses["action_max_std"]  = self.action_max_std
                    fitnesses["action_min_std"]  = self.action_min_std

            # bis 2 - In case the observation is not known            
            observation_global_max_history.append(fitness_obj.extra_info["observation_max_history"])
            observation_global_min_history.append(fitness_obj.extra_info["observation_min_history"])

        # bis 3 - In case the observation is not known
        # print("obs_max use :  ", np.round(self.observation_max_global, 4).tolist(), " obs_min use:   ", np.round(self.observation_min_global, 4).tolist())
        self.observation_max_global = self.observation_max_global.copy()
        self.observation_min_global = self.observation_min_global.copy()
        
        self.obersvation_stats(np.asarray(observation_global_max_history), np.asarray(observation_global_min_history), self.observation_max_global, self.observation_min_global, self.auto_obersvation_ratio)
        # print("obs_max found: ", np.round(self.observation_max_global, 4).tolist(), " obs_min found: ", np.round(self.observation_min_global, 4).tolist())

        return fitnesses, self.observation_max_global, self.observation_min_global


    # @staticmethod
    def obersvation_stats(self, observation_max_history:np.ndarray, observation_min_history:np.ndarray, obs_max:np.ndarray, obs_min:np.ndarray, percent_use:float) -> None:
        observation_use:int = np.ceil(observation_max_history.shape[0] * percent_use).astype(int)
        for i in range(observation_max_history.shape[1]):
            obs_max[i] = np.mean(np.sort(observation_max_history[:, i])[-observation_use:])
            obs_min[i] = np.mean(np.sort(observation_min_history[:, i])[:observation_use])
            if obs_max[i] == obs_min[i]:
                obs_max[i] += 0.1
                obs_min[i] -= 0.1


class Ant(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=27, nb_output:int=8, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=True, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("Ant-v4", render_mode="human" if render == True else None)

class HalfCheetah(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=17, nb_output:int=6, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=True, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("HalfCheetah-v4", render_mode="human" if render == True else None)

class Swimmer(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=8, nb_output:int=2, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=True, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("Swimmer-v4", render_mode="human" if render == True else None)

class Hopper(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=11, nb_output:int=3, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=True, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("Hopper-v4", render_mode="human" if render == True else None)

class Walker2D(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=17, nb_output:int=6, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=True, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("Walker2d-v4", render_mode="human" if render == True else None)

class Humanoid(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=376, nb_output:int=17, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=True, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("Humanoid-v4", render_mode="human" if render == True else None)

class HumanoidStandup(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=376, nb_output:int=17, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=True, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("HumanoidStandup-v4", render_mode="human" if render == True else None)

class InvertedPendulum(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=4, nb_output:int=1, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=True, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("InvertedPendulum-v4", render_mode="human" if render == True else None)
    
class InvertedDoublePendulum(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=6, nb_output:int=1, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=True, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("InvertedDoublePendulum-v4", render_mode="human" if render == True else None)
    
class Reacher(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=11, nb_output:int=2, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=True, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("Reacher-v4", render_mode="human" if render == True else None)
    

# 2.1 - Discrete
class Mountain_Car(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=2, nb_output:int=1, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=True, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("MountainCar-v0", render_mode="human" if render == True else None)
    
    @staticmethod # override the EnvGym_Wrapper decoding function
    def decoding_output_network_to_action(action: np.ndarray) -> int:
        # print(action)
        return np.argmax(action)
    
class Mountain_Car_Continuous(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=2, nb_output:int=1, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=True, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("MountainCarContinuous-v0", render_mode="human" if render == True else None)


class Cart_Pole(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=4, nb_output:int=2, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=False, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("CartPole-v1", render_mode="human" if render == True else None)
    
    @staticmethod # override the EnvGym_Wrapper decoding function
    def decoding_output_network_to_action(action: np.ndarray) -> int:
        return np.argmax(action)

class Acrobot(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=6, nb_output:int=3, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=False, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("Acrobot-v1", render_mode="human" if render == True else None)
    
    @staticmethod # override the EnvGym_Wrapper decoding function
    def decoding_output_network_to_action(action: np.ndarray) -> int:
        return np.argmax(action)

class Pendulum(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=3, nb_output:int=1, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=False, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("Pendulum-v1", render_mode="human" if render == True else None)
    

class Lunar_Lander(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=8, nb_output:int=4, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=False, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make(
                        "LunarLander-v2",
                        continuous = False,
                        gravity = -10.0,
                        enable_wind = False,
                        wind_power = 5.0,
                        turbulence_power = 0.5,
                        render_mode = "human" if render == True else None
                        )
    
    @staticmethod # override the EnvGym_Wrapper decoding function
    def decoding_output_network_to_action(action: np.ndarray) -> int:
        return np.argmax(action)

class Lunar_Lander_Continuous(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=8, nb_output:int=2, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=False, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make(
                        "LunarLander-v2",
                        continuous = True,
                        gravity = -10.0,
                        enable_wind = False,
                        wind_power = 5.0,
                        turbulence_power = 0.5,
                        render_mode = "human" if render == True else None
                        )
    
class Bipedal_Walker(EnvGym_Wrapper):
    def __init__(self, name:str, config_path:str, nb_input:int=24, nb_output:int=4, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=False, auto_obersvation_ratio:float=0.01) -> None:
        EnvGym_Wrapper.__init__(self, name, config_path, nb_input, nb_output, self.__get_env__, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def __get_env__(self, render:bool=False) -> Environment_Gym:
        return gym.make("BipedalWalker-v3", render_mode="human" if render == True else None, hardcore=False)
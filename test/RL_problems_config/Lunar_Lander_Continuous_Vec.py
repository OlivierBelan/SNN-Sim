
from evo_simulator.GENERAL.Genome import Genome_NN
from problem.RL.ENVIRONNEMENT_VEC.ENVIRONNEMENT_VECTORIZE import Environment_Vec
from problem.RL.ENVIRONNEMENT_VEC.GYM_ENV_VECTORIZE import Environment_Gym_Vec
from typing import Dict, Any, Tuple
from RL_problems_config.Config_Problem import Config_Problem

import gymnasium as gym
from gymnasium.vector import VectorEnv

import numpy as np
import numba as nb

# @nb.njit(cache=True, fastmath=False, nogil=True)
def normalize_number(arr:np.ndarray, max_value:np.ndarray, min_value:np.ndarray) -> float:
    normalized = (arr - min_value) / (max_value - min_value)
    return np.clip(normalized, 0.0, 1.0)

class Lunar_Lander_Continuous_Vec(Config_Problem):
    def __init__(self, name:str, config_path:str, nb_input:int=8, nb_output:int=2, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=False, auto_obersvation_ratio:float=0.01) -> None:
        Config_Problem.__init__(self, name, nb_input, nb_output, config_path, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)
        
    def get_env(self, nb_envs:int, vec_mode:str="sync") -> Environment_Gym_Vec:
        self.gym_env:VectorEnv = gym.make_vec(
                                    "LunarLander-v2",
                                    continuous = True,
                                    gravity = -10.0,
                                    enable_wind = False,
                                    wind_power = 5.0,
                                    turbulence_power = 0.5,
                                    num_envs=nb_envs, 
                                    vectorization_mode=vec_mode,
                                    )
        return Environment_Gym_Vec(
            self.name,
            self.gym_env,
            self.encoding_observation_to_input_network,
            self.decoding_output_network_to_action, 
            self.fitness_step, 
            self.fitness_end,
            self.update_observation_min_max,
            self.is_abs_update_observation
        )

    def encoding_observation_to_input_network(self, observation: np.ndarray, observation_max:np.ndarray, observation_min:np.ndarray) -> np.ndarray:
        if self.auto_obersvation == True:
            if self.is_scaling_update_observation == True: observation += (observation * self.observation_scaling)
            observation = normalize_number(observation, self.observation_max_global, self.observation_min_global)
        else:
            observation = normalize_number(observation, observation_max, observation_min)
            self.observation_max_global = observation_max # just to for the print in the end (used vs found)
            self.observation_min_global = observation_min

        if np.any(observation > 1) or np.any(observation < 0):
             raise Exception("observation[i] > 1 or observation[i] < 0")
        
        return observation


    def decoding_output_network_to_action(self, action:np.ndarray, network_type:str) -> np.ndarray:        
        if network_type == "SNN":
            return np.interp(action, [0, 1], [-1, +1])
        else:
            return action # car l'action est déjà entre -1 et 1 (tanh function output)

    @staticmethod
    def fitness_step(reward_pop_cumulative:np.ndarray, info:Dict[str, Any]) -> None:
        # 1 - update the fitness with the reward
        pass
        

    def fitness_end(self, genomes:Dict[int, Genome_NN], rewards:np.ndarray, obs_max:np.ndarray, obs_min:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rewards = rewards.T  # reshape from (episodes, nb_genomes) to (nb_genomes, episodes)

        # 1 - Save fitness information in the genome
        for i, genome in enumerate(genomes.values()):
            genome.fitness.score = np.sum(rewards[i])
            genome.info["best_episode_raw_score"] = np.max(rewards[i])
            genome.info["mean_episode_raw_score"] = np.mean(rewards[i])


        # 2 - Update the observation max and min
        self.observation_max_global = self.observation_max_global.copy()
        self.observation_min_global = self.observation_min_global.copy()
        self.obersvation_stats(obs_max, obs_min, self.observation_max_global, self.observation_min_global, self.auto_obersvation_ratio)

        return self.observation_max_global, self.observation_min_global


    # @staticmethod
    def obersvation_stats(self, observation_max_history:np.ndarray, observation_min_history:np.ndarray, obs_max:np.ndarray, obs_min:np.ndarray, percent_use:float) -> None:
        observation_use:int = np.ceil(observation_max_history.shape[0] * percent_use).astype(int)
        for i in range(observation_max_history.shape[1]):
            obs_max[i] = np.mean(np.sort(observation_max_history[:, i])[-observation_use:])
            obs_min[i] = np.mean(np.sort(observation_min_history[:, i])[:observation_use])
            if obs_max[i] == obs_min[i]:
                obs_max[i] += 0.1
                obs_min[i] -= 0.1

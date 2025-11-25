import gymnasium as gym
from gymnasium.vector import VectorEnv
from typing import List, Callable, Any
from evo_simulator.GENERAL.Fitness import Fitness
from problem.RL.ENVIRONNEMENT_VEC.ENVIRONNEMENT_VECTORIZE import Environment_Vec
# import copy
import numpy as np
import numba as nb


class Environment_Gym_Vec(Environment_Vec):
    def __init__(self, env_name, gym_env:VectorEnv, encoding_observation_to_network_input_function:Callable, decoding_output_network_to_action_function:Callable, fitness_step_function:Callable, fitness_end_function:Callable, update_observation_min_max_function:Callable, is_abs_update_observation:bool = False, seed:int = None, is_Pybullet_Gym:bool = False):
        Environment_Vec.__init__(self, env_name, encoding_observation_to_network_input_function, decoding_output_network_to_action_function, fitness_step_function, fitness_end_function, update_observation_min_max_function)
        
        self.gym_env:VectorEnv = gym_env
        self.seed = seed
        self.is_Pybullet_Gym = is_Pybullet_Gym
        self.nb_envs:int = gym_env.num_envs
        self.reward_cumulative:np.ndarray = np.zeros(self.gym_env.num_envs)
        self.is_abs_update_observation:bool = is_abs_update_observation
        self.terminated_save:np.ndarray = np.zeros(self.gym_env.num_envs, dtype=bool)

    def update(self, action_encoded:np.ndarray, network_type:str="SNN") -> bool:
        # 1 - Decode the action and update the environment with the decoded action
        action_decoded:np.ndarray = self.decoding_output_network_to_action_function(action_encoded, network_type)
        self.observation, self.reward, self.terminated, self.truncated, self.extra_info = self.gym_env.step(action_decoded)

        # 2 - Update the fitness of the current step
        # self.fitness_step_function(
        #     self.reward_cumulative,
        #     {
        #     "observation":self.observation,
        #     "reward": self.reward, 
        #     "action": action_decoded, 
        #     "terminated": self.terminated, 
        #     "truncated": self.truncated,
        #     "extra_info_env": self.extra_info, 
        #     # "observation_max_history": self.observation_max_history,
        #     # "observation_min_history": self.observation_min_history,
        #     "observation_space_high": self.gym_env.observation_space.high,
        #     "observation_space_low": self.gym_env.observation_space.low,
        #     # "observation_history": self.observation_history,
        #     # "reward_history": self.reward_history, 
        #     # "action_history":self.action_history, 
        #     })
        # print("terminated:", self.terminated, "shape:", self.terminated.shape)
        # print("truncated:", self.truncated, "shape:", self.truncated.shape)
        # print("reward:", self.reward, "shape:", self.reward.shape)
        # not_terminated_idx:np.ndarray = np.where((self.terminated == False) & (self.truncated == False))[0]
        self.terminated_save += self.terminated
        self.terminated_save += self.truncated
        not_terminated_idx:np.ndarray = np.where(self.terminated_save == False)[0]
        self.reward_cumulative[not_terminated_idx] += self.reward[not_terminated_idx]

        # 3 - Save the observation, reward and action in the history variables
        # self.observation_history.append(self.observation)
        # self.reward_history.append(self.reward)
        # self.action_history.append(action_decoded)

        if self.observation_max_history is None:
            self.observation_max_history = self.observation[0].copy()
            self.observation_min_history = self.observation[0].copy()

        # print("observation:", self.observation, "shape:", self.observation.shape)
        # print("max_observation", np.max(self.observation, axis=0), "shape:", np.max(self.observation, axis=0).shape)
        # print("min_observation", np.min(self.observation, axis=0), "shape:", np.min(self.observation, axis=0).shape)
        # print("before update obs_max:", self.observation_max_history)
        # print("before update obs_min:", self.observation_min_history)
        if self.is_abs_update_observation == False:
            self.update_observation_history_raw(self.observation_max_history, self.observation_min_history, self.observation)
        elif self.is_abs_update_observation == True:
            self.update_observation_history_abs(self.observation_max_history, self.observation_min_history, self.observation)

        # print("after update obs_max:", self.observation_max_history, "shape:", self.observation_max_history.shape)
        # print("after update obs_min:", self.observation_min_history, "shape:", self.observation_min_history.shape)
        # exit(0)

        return len(not_terminated_idx) > 0

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def update_observation_history_raw(observation_max_history:np.ndarray, observation_min_history:np.ndarray, observation:np.ndarray) -> None:
        for i in nb.prange(observation.shape[0]):
            for j in nb.prange(observation.shape[1]):
                if observation[i, j] > observation_max_history[j]:
                    observation_max_history[j] = observation[i, j]
                if observation[i, j] < observation_min_history[j]:
                    observation_min_history[j] = observation[i, j]

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def update_observation_history_abs(observation_max_history:np.ndarray, observation_min_history:np.ndarray, observation:np.ndarray) -> None:
        for i in nb.prange(observation.shape[0]):
            for j in nb.prange(observation.shape[1]):
                if np.abs(observation[i, j]) > np.abs(observation_max_history[j]):
                    observation_max_history[j] = np.abs(observation[i, j])
                    observation_min_history[j] = -observation_max_history[j]

    def reset(self, seed:int = None):
        self.seed = int(seed) if seed is not None else int(self.seed)
        if self.is_Pybullet_Gym == True:
            self.gym_env.seed(self.seed)
            self.gym_env.action_space.seed(self.seed)
            self.gym_env.observation_space.seed(self.seed)
            self.observation = self.gym_env.reset()
        else:
            if self.seed is not None:
                self.observation, self.extra_info = self.gym_env.reset(seed=self.seed)
            else:
                self.observation, self.extra_info = self.gym_env.reset()
        self.reward:np.ndarray = None
        self.terminated:bool = False
        self.truncated:bool = False
        self.terminated_save:np.ndarray = np.zeros(self.gym_env.num_envs, dtype=bool)
        self.reward_cumulative:np.ndarray = np.zeros(self.gym_env.num_envs)
        # self.observation_history:List[np.ndarray] = []
        # self.reward_history:List[float] = []
        # self.action_history:List[np.ndarray] = []

    def encoding_observation_to_network_input(self) -> np.ndarray:
        if self.encoding_observation_to_network_input_function is not None:
            return self.encoding_observation_to_network_input_function(self.observation.copy(), self.gym_env.observation_space.high, self.gym_env.observation_space.low)
        else:
            raise NotImplementedError("The encoding_observation_to_spike function is not implemented")
            
    def close(self):
        self.gym_env.close()

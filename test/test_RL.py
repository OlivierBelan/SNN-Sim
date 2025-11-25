import os
import sys
import pathlib
import argparse
sys.path.append('../')
sys.path.append('../src/')
sys.path.append('../src/snn_simulator/')
sys.path.append('../src/evo_simulator/')
sys.path.append('../src/snn_simulator/SNN_cython_cuda')
project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

# Add the Firefly path to the sys.path (with its own sample_factory)
base = pathlib.Path(__file__).resolve().parents[1]
firefly = base / "src" / "problem" / "RL" / "Firefly"
if str(firefly) not in sys.path: sys.path.insert(0, str(firefly))

import gymnasium as gym
sys.modules["gym"] = gym
import pybullet_envs_gymnasium

import numpy as np
np.set_printoptions(suppress=True)
from evo_simulator.GENERAL.Index_Manager import device
from evo_simulator.GENERAL.Neuro_Evolution import Neuro_Evolution

from problem.RL.REINFORCEMENT import Reinforcement_Manager
# Algorithms Mono-Objective

# JAX Algorithms
from evo_simulator.ALGORITHMS.EvoSAX.EvoSax_algo import EvoSax_algo



# Problems
# Discrete
from RL_problems_config.Mountain_car import Mountain_Car
from RL_problems_config.Cart_Pole import Cart_Pole
from RL_problems_config.Arcrobot import Acrobot
from RL_problems_config.Lunar_Lander import Lunar_Lander

# Continuous
from RL_problems_config.Bipedal_Walker import Bipedal_Walker
from RL_problems_config.Mountain_car_Continuous import Mountain_Car_Continuous
from RL_problems_config.Pendulum import Pendulum
from RL_problems_config.Lunar_Lander_Continuous import Lunar_Lander_Continuous

# from RL_problems_config.Anymal import Anymal


# Robot
from RL_problems_config.GymanasiumEnvWrapper import HalfCheetah, Hopper, Swimmer, Walker2D, Ant, Humanoid, HumanoidStandup, InvertedPendulum, InvertedDoublePendulum, Reacher

from typing import List, Dict, Tuple, Any, Callable
np.set_printoptions(threshold=sys.maxsize)


# Algo evosax
def evosax_func(name:str, config_path) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_path)
    extra_info:Dict[str, Any] = {}

    return name, EvoSax_algo, config_path, extra_info


def str_to_list(s:str) -> List[float]:
    # e.g. s = "[1,2,3]"
    return [float(x) for x in s[1:-1].split(",")]

def str_to_list_of_list(s:str) -> List[List[int]]:
    # e.g. s = "[[1,2,3], [4,5,6]]"
    return [[int(x) for x in l[1:-1].split(",")] for l in s[2:-2].split("], [")]

def to_bool(s) -> bool:
    if s == "True":
        return True
    else:
        return False

def find_cfg_two_dirs_up(render_path:str):
    """
    Find the config file in the directory two levels up from the
    """
    # Get the absolute path of the render file
    abs_path = os.path.abspath(render_path)
    
    # Get the absolute path of the directory two levels up
    dir_two_up = os.path.abspath(os.path.join(abs_path, "..", ".."))
    
    # Check if the directory exists
    if os.path.exists(dir_two_up) and os.path.isdir(dir_two_up):
        for file in os.listdir(dir_two_up):
            if file.endswith(".cfg") or "config" in file:
                return os.path.join(dir_two_up, file)
    return None    

def parse_arg():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--device', type=str, help='Device on which the code is executed: \'cpu\' or \'gpu\' or \'cpu_gpu\' or \'gpu_cpu\'', default="cpu")
    parser.add_argument('--nb_cpu', type=int, help='Number of cpu', default=1)
    parser.add_argument('--nb_gpu', type=int, help='Number of jobs on gpu', default=1)
    parser.add_argument('--cpu_gpu_ratio', type=str_to_list, help='Ratio of cpu/gpu', default=None)
    parser.add_argument('--nn', type=str, help='Type of neural network', default="SNN")
    parser.add_argument('--algo', type=str, help='Algorithm name', default="NES-evosax")
    parser.add_argument('--problem', type=str, help='Problem name')
    parser.add_argument('--nb_runs', type=int, help='Number of runs', default=3)
    parser.add_argument('--nb_generations', type=int, help='Number of generations', default=50)
    parser.add_argument('--nb_episodes', type=int, help='Number of episodes', default=1)
    parser.add_argument('--record', type=to_bool, help='Record data', default="False")
    parser.add_argument('--config', type=str, help='Config path', default=None)
    parser.add_argument('--seed', type=str_to_list_of_list, help='Seed', default=None)
    parser.add_argument('--debug', type=to_bool, help='Debug', default="False")
    parser.add_argument('--render', type=str, help='path_to_file_genome.pkl', default=False)
    
    args = parser.parse_args()

    if args.render != False and args.config == None:
        args.config = find_cfg_two_dirs_up(args.render)
        if args.config == None:
            raise Exception("Render config file not found, please either specify the config file or put the render config file in the same directory as the config file")
    return args



def get_algorithm(nn:str, algo:str, config:str) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 0 - Config path
    if nn.upper() == "SNN":
        start_config_path = "./config/config_snn/"
    elif nn.upper() == "ANN":
        start_config_path = "./config/config_ann/"
    else:
        raise Exception("Neural network:" + nn + " not found")
    
    # 1. - Algorithms from evoSAX (https://github.com/RobertTLange/evosax)
    if algo == "NES":   return evosax_func("NES", config) if config is not None else evosax_func("NES", start_config_path + "NES_CONFIG_RL.cfg")
    
    else:
        raise Exception("Algorithm" + algo + " not found")

def get_problem(problem:str, config_path:str) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 2 - Environnement
    # 2.1 - Discrete
    if problem == "Mountain_Car":              return Mountain_Car("MountainCar", config_path, nb_input=2, nb_output=3, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=False)
    elif problem == "Cart_Pole":               return Cart_Pole("CartPole",       config_path, nb_input=4, nb_output=2, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Acrobot":                 return Acrobot("Acrobot",          config_path, nb_input=6, nb_output=3, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=False)
    elif problem == "Lunar_Lander":            return Lunar_Lander("LunarLander", config_path, nb_input=8, nb_output=4, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=False)

    # 2.2 - Continuous
    elif problem == "Mountain_Car_Continuous":  return Mountain_Car_Continuous("MountainCarContinous",  config_path, nb_input=2, nb_output=1, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Pendulum":                 return Pendulum("Pendulum",                         config_path, nb_input=3, nb_output=1, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Lunar_Lander_Continuous":  return Lunar_Lander_Continuous("LunarLanderContinuous", config_path, nb_input=8, nb_output=2, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=False)
    elif problem == "Bipedal_Walker":           return Bipedal_Walker("BipedWalker",                    config_path, nb_input=24, nb_output=4, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=False, hardcore=False)

    # 2.3 - Continuous Robot
    elif problem == "Swimmer":                 return Swimmer("Swimmer",         config_path, nb_input=8, nb_output=2, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Hopper":                  return Hopper("Hopper",           config_path, nb_input=11, nb_output=3, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "HalfCheetah":             return HalfCheetah("HalfCheetah", config_path, nb_input=17, nb_output=6, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Walker2D":                return Walker2D("Walker2D",       config_path, nb_input=17, nb_output=6, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Ant":                     return Ant("Ant",                 config_path, nb_input=27, nb_output=8, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Humanoid":                return Humanoid("Humanoid",     config_path, nb_input=376, nb_output=17, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "InvertedPendulum":        return InvertedPendulum("InvertedPendulum", config_path, nb_input=4, nb_output=1, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "InvertedDoublePendulum":  return InvertedDoublePendulum("InvertedDoublePendulum", config_path, nb_input=6, nb_output=1, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Reacher":                 return Reacher("Reacher",         config_path, nb_input=11, nb_output=2, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Anymal":                  return Anymal("Anymal",           config_path, nb_input=115, nb_output=12, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)

    else:
        raise Exception("Problem" + problem + " not found")

def neuro_evo_matrix_func():
    args = parse_arg()

    device = "cpu" if (args.nn == "ANN" and args.device == "cpu") else "cuda"

    # 1 - Algorithm
    name, algorithm, config_path, algo_extra_info = get_algorithm(args.nn, args.algo, args.config)

    # 2 - Problem
    environnement = get_problem(args.problem, config_path)


    # 3 - Seeds    
    max_seeds:int = int(1e6)
    rng = np.random.default_rng()
    seeds = rng.choice(max_seeds, size=(args.nb_runs, args.nb_episodes), replace=False)
    print("seeds: ", seeds)
    
    # 4 - Run
    neuro:Neuro_Evolution = Neuro_Evolution(nb_generations=args.nb_generations, nb_runs=args.nb_runs, is_record=args.record, config_path=config_path, device=args.device, nb_cpu=args.nb_cpu, nb_gpu=args.nb_gpu, cpu_gpu_ratio=args.cpu_gpu_ratio, is_debug=args.debug)
    neuro.init_algorithm(name, algorithm, config_path, algo_extra_info)


    # If you want to run QD Gym uncomment the following line and comment the following line (neuro.run_rastrigin)
    neuro.init_problem_RL(Reinforcement_Manager, config_path, environnement, nb_episode=args.nb_episodes, seeds=seeds, render=args.render)
    neuro.run()

def main():
    neuro_evo_matrix_func()

if __name__ == "__main__":
    main()

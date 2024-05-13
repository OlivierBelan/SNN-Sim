import os
import sys
sys.path.append('../')
sys.path.append('../src/')
sys.path.append('../src/snn_simulator/')
sys.path.append('../src/evo_simulator/')
os.environ["RAY_DEDUP_LOGS"] = "0"
from typing import List, Tuple, Dict
import random


from evo_simulator.GENERAL.Neuro_Evolution import Neuro_Evolution
from problem.SL.SUPERVISED import Supervised_Manager

# Algorithms
from evo_simulator.ALGORITHMS.NEAT.Neat import NEAT
from evo_simulator.ALGORITHMS.CMA_ES.CMA_ES import CMA_ES
from evo_simulator.ALGORITHMS.EvoSAX.EvoSax_algo import EvoSax_algo

import numpy as np
np.set_printoptions(threshold=sys.maxsize)



def min_max_normalize(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals)

def z_score_standardize(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    return (data - mean) / std_dev

def wine_data_set():
    # 1 - Load data
    data = np.genfromtxt("./data_set/wine.csv", delimiter=',', skip_header=False)
    labels = data[:, 0]
    features = data[:, 1:]

    # 2 - Normalize data
    features = min_max_normalize(features)

    # 3 - Shuffle data
    data_list = list(zip(labels, features))
    random.shuffle(data_list)
    labels, features = zip(*data_list)

    # 4 - Convert to numpy array
    labels = np.array(labels, dtype=np.float32) - 1
    features = np.array(features, dtype=np.float32)

    return "WINE", features, labels

def breast_cancer_data_set():
    # 1 - Load data
    data = np.genfromtxt("./data_set/wdbc.data", delimiter=',', dtype=object)

    # 2 - Replace labels by numbers
    labels = data[:, 1]
    labels[labels == b'M'] = 1
    labels[labels == b'B'] = 2
    data[:, 1] = labels
    data = data.astype(np.float32)

    # 3 - Separate labels and features
    labels = data[:, 1]
    features = data[:, 2:]

    # 4 - Normalize data
    features = min_max_normalize(features)

    # 5 - Shuffle data
    data_list = list(zip(labels, features))
    random.shuffle(data_list)
    labels, features = zip(*data_list)

    # 6 - Convert to numpy array
    labels = np.array(labels, dtype=np.float32) - 1
    features = np.array(features, dtype=np.float32)

    return "BREAST_CANCER", features, labels

def xor_data_set():
    features = np.array([[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]], dtype=np.float64)
    label = np.array([[1, 0, 1, 0], [1, 0, 1, 0]])
    # data_list = list(zip(output_expected, data_set))
    # random.shuffle(data_list)
    return "XOR", features, label


# Algo Mono-Objective

def neat_func(config_path) -> Tuple[Neuro_Evolution, str]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, start_config_path + config_path)
    
    return "NEAT", NEAT, neat_config_path

def cma_es_func(config_path) -> Tuple[Neuro_Evolution, str]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, start_config_path + config_path)
    
    return "CMA_ES", CMA_ES, neat_config_path



def evosax_func(name:str, config_path) -> Tuple[Neuro_Evolution, str]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, start_config_path + config_path)
    
    return name, EvoSax_algo, config_path


start_config_path = "./config/config_snn/SL/"
# start_config_path = "./config/config_ann/SL/"

def neuro_evo_func(args:List[str]):
    if len(args) == 0:raise Exception("Error: No arguments")

    # 1.0 - Algorithms
    aglos_dict:Dict[str, Tuple[str, Neuro_Evolution, str]] = {
        "NEAT":           neat_func("NEAT_CONFIG_SL.cfg"),
        "CMA_ES":         cma_es_func("CMA_ES_CONFIG_SL.cfg"),
        # 1.3 - Algorithms from evoSAX (https://github.com/RobertTLange/evosax)
        "NES-evosax":     evosax_func("NES-evosax", "NES-evosax_CONFIG_SL.cfg"),

    }
    name, algorithm, config_path = aglos_dict[args[0]]

    # 2 - Data set
    problem_name, features, labels = wine_data_set() # input size = 13, output size = 3
    # problem_name, features, labels = breast_cancer_data_set() # input size = 30, output size = 2
    # problem_name, features, labels = xor_data_set() # input size = 2, output size = 2
    print("\nLEN DATA SET = ", problem_name, "size", len(features), "LEN input = ", len(features[0]), "labels= ", np.unique(labels), "\n")

    
    nb_runs:int = 3
    nb_generation:int = 50

    # 3 - Train
    neuro_evo:Neuro_Evolution = Neuro_Evolution(nb_generations=nb_generation, nb_runs=nb_runs, is_record=True, config_path=config_path, cpu=20)
    neuro_evo.init_algorithm(name, algorithm, config_path)
    neuro_evo.init_problem_SL(Supervised_Manager, config_path, problem_name, features, labels)
    neuro_evo.run()



def main():
    neuro_evo_func(sys.argv[1:])

if __name__ == "__main__":
    main()

import os
import sys
sys.path.append('../')
sys.path.append('../src/')
sys.path.append('../src/snn_simulator/')
sys.path.append('../src/evo_simulator/')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
os.environ["RAY_DEDUP_LOGS"] = "0"
from typing import List, Tuple, Dict, Any, Callable
import argparse
import random

import evo_simulator.GENERAL.Index_Manager as idx_manager
from evo_simulator.GENERAL.Index_Manager import device
from evo_simulator.GENERAL.Neuro_Evolution import Neuro_Evolution
from problem.SL.SUPERVISED import Supervised_Manager

# Algorithms
from evo_simulator.ALGORITHMS.EvoSAX.EvoSax_algo import EvoSax_algo

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import hyper_substrat_config

import numpy as np
np.set_printoptions(threshold=sys.maxsize)


def str_to_list(s:str) -> List[float]:
    # e.g. s = "[1,2,3]"
    return [float(x) for x in s[1:-1].split(",")]


def get_mnist_data_set(sample_size:int=3000):
    batch_size = sample_size
    data_path='./data_set/mnist'

    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    # mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader_mnist = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_loader_mnist = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
    features, labels = next(iter(train_loader_mnist))
    return "MNIST", features.flatten(1).numpy().astype(np.float32), labels.numpy().astype(np.float32), False


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

    return "WINE", features, labels, False # Name, features, labels, is_continuous_label

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

    return "BREAST_CANCER", features, labels, False # Name, features, labels, is_continuous_label

def xor_data_set():
    features = np.array([[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]], dtype=np.float64)
    label = np.array([[1, 0, 1, 0], [1, 0, 1, 0]])
    # data_list = list(zip(output_expected, data_set))
    # random.shuffle(data_list)
    return "XOR", features, label, False # Name, features, labels, is_continuous_label


def sinus_data_set(nb_points:int=100, amplitude:float=0.5, phase:float=0.0, frequency:float=6.283186, offset:float=0.5):
    # 1 - Generate data
    x = np.linspace(0, 1, nb_points)
    y = amplitude * np.sin(frequency * x + phase) + offset

    # 2 - Normalize data
    # x = min_max_normalize(x.reshape(-1, 1))
    # y = min_max_normalize(y.reshape(-1, 1))

    # 3 - Shuffle data
    # data_list = list(zip(y, x))
    # random.shuffle(data_list)
    # y, x = zip(*data_list)

    # 4 - Convert to numpy array
    y = np.expand_dims(np.array(y, dtype=np.float32), axis=0)
    x = np.expand_dims(np.array(x, dtype=np.float32), axis=0)

    # 5 - plot data
    print("y[0]", y[0,0], "y[-1]", y[0,-1], "shape", y.shape)
    print("x:", x, "shape", x.shape)
    print("y:", y, "shape", y.shape)
    # import matplotlib.pyplot as plt
    # plt.scatter(x, y)
    # plt.xlabel('x')
    # plt.ylabel('sin(x)')
    # plt.title('Sinus Function')
    # plt.show()
    # exit()
    return "SINUS", y, y, True # Name, features, labels, is_continuous_label

# Algo Mono-Objective
def evosax_func(name:str, config_path) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_path)
    extra_info:Dict[str, Any] = {}
    
    return name, EvoSax_algo, config_path, extra_info


def parse_arg():
    def to_bool(s) -> bool:
        if s == "True":
            return True
        else:
            return False
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
    parser.add_argument('--record', type=to_bool, help='Record data', default="False")
    parser.add_argument('--config', type=str, help='Config path', default=None)
    parser.add_argument('--seed', type=List[List[int]], help='Seed', default=None)
    parser.add_argument('--debug', type=to_bool, help='Debug', default="False")

    return parser.parse_args()


def get_algorithm(nn:str, algo:str, cpu:int, config:str) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 0 - Config path
    if nn.upper() == "SNN":
        start_config_path = "./config/config_snn/"
    elif nn.upper() == "ANN":
        start_config_path = "./config/config_ann/"
    else:
        raise Exception("Neural network:" + nn + " not found")
    
    # 1.0 - Algorithms
    if algo == "NES":    return evosax_func("NES",    config) if config is not None else evosax_func("NES",    start_config_path + "NES_CONFIG_SL.cfg")

    else:
        raise Exception("Algorithm:" + algo + " not found")
    

def get_data_set(problem:str) -> Tuple[str, np.ndarray, np.ndarray, bool]:
    if problem == "WINE":            return wine_data_set()           # input size = 13,  output size = 3
    elif problem == "BREAST_CANCER": return breast_cancer_data_set()  # input size = 30,  output size = 2
    elif problem == "XOR":           return xor_data_set()            # input size = 2,   output size = 2
    elif problem == "MNIST":         return get_mnist_data_set()      # input size = 784, output size = 10
    elif problem == "SINUS":         return sinus_data_set()          # input size = 1,   output size = 1
    else:
        raise Exception("Problem:" + problem + " not found")

def neuro_evo_func():
    args = parse_arg()

    idx_manager.device = "cpu" if (args.nn == "ANN" and args.device == "cpu") else "cuda"

    # 1 - Get Algorithm
    name, algorithm, config_path, algo_extra_info = get_algorithm(args.nn, args.algo, args.nb_cpu, args.config)

    # 2 - Get Data Set
    problem_name, features, labels, is_continuous_label = get_data_set(args.problem)
    print("\nLEN DATA SET = ", problem_name, "size", len(features), "LEN input = ", len(features[0]), "labels= ", np.unique(labels), "\n")

    # 3 - Run
    neuro_evo:Neuro_Evolution = Neuro_Evolution(nb_generations=args.nb_generations, nb_runs=args.nb_runs, is_record=args.record, config_path=config_path, device=args.device, nb_cpu=args.nb_cpu, nb_gpu=args.nb_gpu, cpu_gpu_ratio=args.cpu_gpu_ratio, is_debug=args.debug)
    neuro_evo.init_algorithm(name, algorithm, config_path, algo_extra_info)
    neuro_evo.init_problem_SL(Supervised_Manager, config_path, problem_name, features, labels, is_continuous_label)
    neuro_evo.run()



def main():
    parse_arg()
    neuro_evo_func()

if __name__ == "__main__":
    main()

from typing import List, Union, Any, ValuesView, Generator, Dict, Tuple, Union
import numpy as np
import random
from configparser import ConfigParser
import numba as nb
import math
import matplotlib.pyplot as plt

def random_fast_choice(options:List, probs:List) -> Any:
    if (len(options) != len(probs)):
        raise Exception("random_fast_choice: options and probs must have the same length")
    x:float = random.random()
    cum:float = 0.0
    for i, p in enumerate(probs):            
        cum += p            
        if x < cum:
            return options[i]
    return options[-1]

def random_fast_choice_view(options:ValuesView, probs:Generator) -> Any:
    x:float = random.random()
    cum:float = 0.0
    for o, p in zip(options, probs):
        cum += p            
        if x < cum:
            return o
    return o

def normalize_array(array:Union[np.ndarray, float], max:float, min:float) -> Union[np.ndarray, float]:
    if isinstance(array, list):
        array = np.array(array)
    if np.all((array == min)) and (max - min) == 0 or (max - min) == 0:
        return (array - min) + 1
    return (array - min) / (max - min)

def config_function(config_path_file:str, items:List[str]) -> Dict[str, Dict[str, Any]]:
    config_dict:Dict[str, Dict[str, Any]] = {}
    config_parser = ConfigParser()
    with open(config_path_file) as f:
        config_parser.read_file(f)
    
    for section in items:
        if config_parser.has_section(section) == False:
            raise Exception(f"{section} section not found in config file")
        config_dict[section] = dict(config_parser.items(section))
    
    return config_dict

def is_config_section(config_path_file:str, section:str) -> bool:
    config_parser = ConfigParser()
    with open(config_path_file) as f:
        config_parser.read_file(f)
    return config_parser.has_section(section)

def config_function_all(config_path_file:str) -> Dict[str, Dict[str, Any]]:
    config_dict:Dict[str, Dict[str, Any]] = {}
    config_parser = ConfigParser()
    with open(config_path_file) as f:
        config_parser.read_file(f)
    
    for section in config_parser.keys():
        config_dict[section] = dict(config_parser.items(section))
    
    return config_dict

@staticmethod
@nb.njit(cache=True, fastmath=True, nogil=True)
def cosine_distance_array_jit(array_1:np.ndarray, array_2:np.ndarray) -> float:
    return 1 - np.dot(array_1, array_2) / (np.linalg.norm(array_1) * np.linalg.norm(array_2))

@staticmethod
@nb.njit(cache=True, fastmath=True, nogil=True)
def euclidean_distance_array_jit(array_1:np.ndarray, array_2:np.ndarray) -> float:
    return np.linalg.norm(array_1-array_2)


def split(seq:list, num: int, to_numpy:bool = False) -> List[List]:
    if num <= 0:
        raise ValueError("Number of chunks should be greater than 0")

    k, m = divmod(len(seq), num)
    out = []

    i = 0
    for _ in range(num):
        next_i = i + k + (1 if _ < m else 0)  # This distributes the extra m elements to the first m chunks
        if to_numpy:
            out.append(np.array(seq[i:next_i], dtype=np.int32))
        else:
            out.append(seq[i:next_i])
        i = next_i

    return out

def hiddens_from_config(hiddens:str) -> Dict[str, Union[Dict[str, int], int]]:
    hiddens = hiddens.replace(" ", "").split(",")
    hidden_config:Dict[str, Dict[str, int]] = {}
    hidden_config["layer_names"] = []
    nb_hiddens:int = 0
    nb_hiddens_active:int = 0
    for h in hiddens:
        h_params = h.split(":")
        if len(h_params) == 2:
            hidden_config[h_params[0]] = {"nb_neurons": int(h_params[1]), "nb_neurons_active": int(h_params[1])}
            nb_hiddens += int(h_params[1])
            nb_hiddens_active += int(h_params[1])
            hidden_config["layer_names"].append(h_params[0])
        elif len(h_params) == 3:
            if int(h_params[2]) > int(h_params[1]):
                raise Exception("hiddens_nb_from_config: nb_neurons_active must be less or equal to nb_neurons -> (" + h_params[0] + ":" + h_params[1] + ":" + h_params[2] + " - layer:nb_neurons:nb_neurons_active)")
            hidden_config[h_params[0]] = {"nb_neurons": int(h_params[1]), "nb_neurons_active": int(h_params[2])}
            nb_hiddens += int(h_params[1])
            nb_hiddens_active += int(h_params[2])
            hidden_config["layer_names"].append(h_params[0])
        else:
            raise Exception("hiddens_nb_from_config: each hidden must contain at least two or three elements, eg: H1:10 (name:nb_neurons) or H1:10:5 (name:nb_neurons:nb_neurons_active)")
    hidden_config["nb_hiddens"] = nb_hiddens
    hidden_config["nb_hiddens_active"] = nb_hiddens_active
    return hidden_config


def architecture_from_config(architecture:str, nb_layers:int) -> Tuple[List[List[str]], List[str]]:
    string_list = architecture.split(", ")
    archictecture_config = []
    is_inputs = False
    is_outputs = False
    hiddens_layer_names:List[str] = []

    for item in string_list:
        connection = item.split("->")
        if len(connection) != 2:
            raise Exception("architecture_from_config: each connection must be a tuple of two elements")
        if connection[0] == "I":
            is_inputs = True
        if connection[1] == "O":
            is_outputs = True
        if connection[0] not in ["I", "O"] and connection[0] not in hiddens_layer_names:
            hiddens_layer_names.append(connection[0])
        if connection[1] not in ["I", "O"] and connection[1] not in hiddens_layer_names:
            hiddens_layer_names.append(connection[1])
        if connection not in archictecture_config:
            archictecture_config.append(connection)

    if is_inputs == False or is_outputs == False: raise Exception("architecture_from_config: architecture must contain at least one input ( I ) and one output (O)")
    if len(hiddens_layer_names) != nb_layers: raise Exception("archi must contain the same number of hidden layers in hiddens config -> architecture_hidden : architecture ( " +str(len(hiddens_layer_names))+" ) != hidden_config (layers) nb ( "+str(nb_layers)+" )" )
    return archictecture_config, hiddens_layer_names


@staticmethod
@nb.njit(cache=True, fastmath=True, nogil=True)
def epsilon_mu_sigma_jit(parameter:np.ndarray, mu_parameter:np.ndarray, sigma_paramater:np.ndarray, min:np.ndarray, max:np.ndarray, mu_bias:float=0, sigma_coef:float=1.0) -> np.ndarray:
    '''
    Jit function for the epsilon computation
    '''
    # 1- set Epislon with the Gaussian (from randn) distribution (Mu -> center of the distribution, Sigma -> width of the distribution)
    mu:np.ndarray = mu_parameter + mu_bias
    sigma:np.ndarray = sigma_paramater * sigma_coef
    epsilon:np.ndarray = np.random.randn(1, parameter.size) * sigma + mu
    # 2- clip Epsilon and apply it to the neurons parameters
    return np.clip(epsilon.astype(np.float32)[0], min, max)


def oscillation(generation:int, speed:float = 0.005) -> float:
    """
    Compute a simple oscillation function.
    """
    angle = generation * speed * 2 * math.pi
    oscillation = math.sin(angle)
    
    # Avoid the oscillation to reach 1.0 or -1.0
    epsilon = 0.001
    if oscillation >= 1.0:
        oscillation = 1.0 - epsilon
    elif oscillation <= -1.0:
        oscillation = -1.0 + epsilon

    return oscillation



def standardize(arr:np.ndarray) -> np.ndarray:
    return (arr - arr.mean()) / arr.std()

def compare_signals(x, y):
    """
    Compare la dynamique de deux signaux x et y.
    
    Paramètres:
    -----------
    x : array-like
        Premier signal (peut être une liste ou un np.array).
    y : array-like
        Deuxième signal (même dimension que x).

    Retourne:
    ---------
    corr : float
        Coefficient de corrélation entre les dérivées discrètes de x et y.
    """
    # Conversion en np.array si nécessaire
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Vérification basique des dimensions
    if x.shape != y.shape:
        raise ValueError("Les deux signaux doivent avoir la même forme/dimension.")

    # Calcul de la dérivée discrète
    dx = np.diff(x, axis=0)
    dy = np.diff(y, axis=0)

    dx_std = standardize(dx)
    dy_std = standardize(dy)

    # Calcul de la corrélation entre les dérivées
    # (on "aplatit" dx et dy, au cas où il y a plusieurs colonnes)
    corr = np.corrcoef(dx.ravel(), dy.ravel())[0, 1]

    # Plot 1 : les deux signaux
    plt.figure()
    plt.plot(x, label='Encoder Signal x')
    plt.plot(y, label='Sinus Signal y')
    plt.title("Signals")
    plt.legend()
    plt.savefig("./results_SNN/Firefly/encoders/rate/rate_signals.png")
    plt.show()

    # # Plot 2 : leurs dérivées
    plt.figure()
    plt.plot(dx, label='d_encoder')
    plt.plot(dy, label='d_sinus')
    plt.title("Derivatives of the signals")
    plt.legend()
    plt.savefig("./results_SNN/Firefly/encoders/rate/rate_derivatives.png")
    plt.show()

    # Plot 3 : les dérivées standardisées
    plt.figure()
    plt.plot(dx_std, label='d_encoder standardise')
    plt.plot(dy_std, label='d_sinus standardise')
    plt.title("Derivative standardized of the signals")
    plt.legend()
    plt.savefig("./results_SNN/Firefly/encoders/rate/rate_derivatives_standardized.png")
    plt.show()

    return corr
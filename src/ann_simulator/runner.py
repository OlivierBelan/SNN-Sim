
import torch
import torch.nn as nn
import numpy as np
from evo_simulator import TOOLS
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.NN import NN
from typing import Dict, List, Union, Tuple, Any, Callable
import time
import re
from evo_simulator.GENERAL.Index_Manager import device
import evo_simulator.GENERAL.Index_Manager as Index_Manager
# import warnings
# warnings.filterwarnings("ignore")

class NN_Custom_torch(nn.Module):
    def __init__(self, genome:Genome_NN, forward_config:Dict[str, Dict[str, Any]], forward_order:List[str]):
        super(NN_Custom_torch, self).__init__()
        self.device = Index_Manager.device
        self.genome_architecture_layers:List[List[str]] = genome.nn.architecture_layers
        self.genome_architecture_neurons:Dict[str, Dict[str, np.ndarray]] = genome.nn.architecture_neurons
        self.forward_config:Dict[str, Dict[str, Any]] = forward_config
        self.forward_order:List[str] = forward_order

        self.connection:Dict[str, nn.Linear] = {}
        self.layers:Dict[str, Dict[str, Union[torch.Tensor, Dict[str, nn.Linear]]]] = {}
        self.hidden_layers:Dict[str, Dict[str, Union[torch.Tensor, Dict[str, nn.Linear]]]] = {}


        self.layers_name_forward:List[str] = ["I"] + genome.hiddens_layer_names + ["O"]
        self.hidden_layer_names:List[str] = genome.hiddens_layer_names


        for layer_name in self.layers_name_forward:
            self.layers[layer_name] = {}
            self.layers[layer_name]["output"]:torch.Tensor = torch.zeros(self.genome_architecture_neurons[layer_name]["size"]).to(self.device)
            self.layers[layer_name]["output_prev"]:torch.Tensor = torch.zeros(self.genome_architecture_neurons[layer_name]["size"]).to(self.device)
            self.layers[layer_name]["bias"]:torch.Tensor = torch.zeros(self.genome_architecture_neurons[layer_name]["size"]).to(self.device)
            self.layers[layer_name]["norm"]:nn.LayerNorm = nn.LayerNorm(self.genome_architecture_neurons[layer_name]["size"]).to(self.device)
            self.layers[layer_name]["size"]:int = self.genome_architecture_neurons[layer_name]["size"]

            if layer_name not in ["I", "O"]:
                self.hidden_layers[layer_name] = self.layers[layer_name]

        for source_name, target_name in self.genome_architecture_layers:
            connection_name:str = source_name + "->" + target_name
            self.connection[connection_name] = nn.Linear(self.genome_architecture_neurons[source_name]["size"], self.genome_architecture_neurons[target_name]["size"], bias=False, device=device)
            for param in self.connection[connection_name].parameters():
                param.requires_grad = False

        
        # print("\n ALL connection:\n", self.connection)
        # for layer_name, info in self.layers.items():
        #     print("\n", layer_name, "->", info)

        # print("\nhidden_layers:\n")
        # for layer_name, info in self.hidden_layers.items():
        #     print(layer_name, "->", info, "\n")

        # print("\nforward_config:", self.forward_config)
        # print("\nforward_order:", self.forward_order)
        # exit()


    def forward_debug(self, input_raw:torch.Tensor):
        
        # 0.0 - Reset output tensor layers        
        for layer in self.layers.values(): # I -> H_0...H_n-1 -> O
            output_layer:torch.Tensor = torch.zeros(layer["size"]).to(device)
            layer["output"] = output_layer

        # 0.1 - Add input raw to input layer
        self.layers["I"]["output"] = self.layers["I"]["output"] + input_raw

        # 1 - Forward
        for order in self.forward_order:

            input_activation:Callable = self.forward_config[order]["layer_input"]["activation"]
            input_norm:bool = self.forward_config[order]["layer_input"]["norm"]
            for input_layer_config in self.forward_config[order]["layer_input"]["layer"]:
                input_layer_name, input_tensor_name = input_layer_config
                input_tensor:torch.Tensor = self.layers[input_layer_name][input_tensor_name]
                print("order:", order)
                print("\ninput_layer_name:", input_layer_name)
                print("input_tensor_name:", input_tensor_name)
                print("input_tensor:", input_tensor)
                print("input_activation:", input_activation)
                print("input_norm:", input_norm)
                for output_layer_name, output_layer_config in self.forward_config[order]["layer_output"].items():
                    output_activation:Callable = output_layer_config["activation"]
                    output_norm:bool = output_layer_config["norm"]
                    output_tensor:torch.Tensor = self.layers[output_layer_name]["output"]
                    output_bias:torch.Tensor = self.layers[output_layer_name]["bias"]
                    connection_weight:nn.Linear = self.connection[input_layer_name + "->" + output_layer_name]
                    print("\noutput_layer_name:", output_layer_name)
                    print("output_activation:", output_activation)
                    print("output_norm:", output_norm)
                    print("output_bias:", output_bias)
                    print("output_tensor before activation:\n", output_tensor)
                    if input_norm == True or output_norm == True:
                        norm_function:nn.LayerNorm = self.layers[output_layer_name]["norm"]
                        self.layers[output_layer_name]["output"] = output_tensor + output_activation(input_activation(norm_function(connection_weight(input_tensor) + output_bias)))
                    else:
                        self.layers[output_layer_name]["output"] = output_tensor + output_activation(input_activation(connection_weight(input_tensor) + output_bias))
                    print("\noutput_tensor after activation:\n", self.layers[output_layer_name]["output"]," \n")

        # 2 - Output Layer
        output_activation:Callable = self.forward_config["O"]["activation"]
        output_norm:bool = self.forward_config["O"]["norm"]
        output_tensor:torch.Tensor = self.layers["O"]["output"]
        print("\nO before activation\n:", self.layers["O"]["output"])
        if output_norm == True:
            norm_function:nn.LayerNorm = self.layers["O"]["norm"]
            self.layers["O"]["output"] = output_activation(norm_function(output_tensor))
        else:
            self.layers["O"]["output"] = output_activation(output_tensor)
        print("\nO after activation\n:", self.layers["O"]["output"])

        # print("\n ALL connection:\n", self.connection)
        # for layer_name, info in self.layers.items():
        #     print("\n", layer_name, "->", info)

        exit()
        # 3 - Update output_prev
        for layer in self.layers.values():
            layer["output_prev"] = layer["output"]

        return self.layers["O"]["output"].cpu()
                    
    def rescale(self, x, src_min, src_max, dst_min, dst_max, round_int=False):
        """
        Remappe linéairement x de [src_min, src_max] vers [dst_min, dst_max].
        Si round_int=True, renvoie des entiers (round to nearest, banker’s rounding).
        """
        # S’assure que tout est un tensor sur le même device / dtype que x
        x = torch.as_tensor(x)
        device, dtype = x.device, x.dtype

        src_min = torch.as_tensor(src_min, dtype=dtype, device=device)
        src_max = torch.as_tensor(src_max, dtype=dtype, device=device)
        dst_min = torch.as_tensor(dst_min, dtype=dtype, device=device)
        dst_max = torch.as_tensor(dst_max, dtype=dtype, device=device)

        y = (x - src_min) / (src_max - src_min) * (dst_max - dst_min) + dst_min

        if round_int:
            y = torch.round(y)          # décimales=0 par défaut (PyTorch ≥2.1)
        return y

    def quant_256(self, x:torch.Tensor, a_max:float=10) -> torch.Tensor:
        x = torch.relu(x)
        x = torch.round(((x/a_max) * 2**8), decimals=0)
        x = torch.clamp(x, 0, 255)
        return x

    def quant_128(self, x:torch.Tensor, a_max:float=10) -> torch.Tensor:
        x = torch.round(((x/a_max) * 2**7), decimals=0)
        x = torch.clamp(x, -127, 127)
        return x

    def forward(self, input_raw:torch.Tensor):
        
        # Quantize input
        # input_raw = self.rescale(input_raw, 0.0, 1.0, -127.0, 127.0) # rescale input to [-127;127]
        # input_raw = self.quant_128(input_raw, a_max=1.0) # quantize input to [-127;127]
        
        # 0.0 - Reset output tensor layers
        for layer in self.layers.values(): # I -> H_0...H_n-1 -> O
            output_layer:torch.Tensor = torch.zeros(layer["size"]).to(self.device)
            layer["output"] = output_layer

        # 0.1 - Add input raw to input layer
        self.layers["I"]["output"] = self.layers["I"]["output"] + input_raw

        # 1 - Forward
        for order in self.forward_order:

            input_activation:Callable = self.forward_config[order]["layer_input"]["activation"]
            input_norm:bool = self.forward_config[order]["layer_input"]["norm"]
            for input_layer_config in self.forward_config[order]["layer_input"]["layer"]:
                input_layer_name, input_tensor_name = input_layer_config
                input_tensor:torch.Tensor = self.layers[input_layer_name][input_tensor_name]

                for output_layer_name, output_layer_config in self.forward_config[order]["layer_output"].items():
                    output_activation:Callable = output_layer_config["activation"]
                    output_norm:bool = output_layer_config["norm"]
                    output_tensor:torch.Tensor = self.layers[output_layer_name]["output"]
                    output_bias:torch.Tensor = self.layers[output_layer_name]["bias"]                    
                    connection_weight:nn.Linear = self.connection[input_layer_name + "->" + output_layer_name]

                    if input_norm == True or output_norm == True:
                        norm_function:nn.LayerNorm = self.layers[output_layer_name]["norm"]
                        self.layers[output_layer_name]["output"] = output_tensor + output_activation(input_activation(norm_function(connection_weight(input_tensor) + output_bias)))
                    else:
                        self.layers[output_layer_name]["output"] = output_tensor + output_activation(input_activation(connection_weight(input_tensor) + output_bias))

        # 2 - Output Layer
        output_activation:Callable = self.forward_config["O"]["activation"]
        output_norm:bool = self.forward_config["O"]["norm"]
        output_tensor:torch.Tensor = self.layers["O"]["output"]

        if output_norm == True:
            norm_function:nn.LayerNorm = self.layers["O"]["norm"]
            self.layers["O"]["output"] = output_activation(norm_function(output_tensor))
        else:
            self.layers["O"]["output"] = output_activation(output_tensor)

        # 3 - Update output_prev
        for layer in self.layers.values():
            layer["output_prev"] = layer["output"]
        
        # print("\n ALL connection:\n", self.connection)
        # for layer_name, info in self.layers.items():
        #     print("\n", layer_name, "->", info)

        return self.layers["O"]["output"].cpu()

    def get_weight_layers(self, weight:np.ndarray, synapses_indexes:Tuple[np.ndarray, np.ndarray], source_indexes:np.ndarray, target_indexes:np.ndarray) -> np.ndarray:
        input_synpases_indexes = np.where(np.isin(synapses_indexes[0], source_indexes))[0]
        target_synapse_indexes = np.where(np.isin(synapses_indexes[1][input_synpases_indexes], target_indexes))[0]
        weight_sub = weight[synapses_indexes[0][input_synpases_indexes[target_synapse_indexes]], synapses_indexes[1][input_synpases_indexes[target_synapse_indexes]]]
        return weight_sub


    def set_parameters_2(self, genome:Genome_NN, weights_layers_indexes:Dict[str, Tuple[np.ndarray, np.ndarray]], is_bias:bool = False, is_dynamic_topology:bool = False) -> None:
        g_nn:NN = genome.nn
        if is_dynamic_topology == True:
            weights:np.ndarray = g_nn.parameters["weight"].copy()
            synapses_unactive_indexes:Tuple[np.ndarray, np.ndarray] = g_nn.synapses_unactives_weight_indexes
            weights[synapses_unactive_indexes[0], synapses_unactive_indexes[1]] = 0.0 # set unactive synapses to 0.0 (for the NEAT algorithm)
        else:
            weights:np.ndarray = g_nn.parameters["weight"]

        for source_name, target_name in self.genome_architecture_layers:
            connection_key:str = source_name + "->" + target_name
            self.connection[connection_key].weight.data = torch.tensor(weights[weights_layers_indexes[connection_key]]).reshape(self.connection[connection_key].weight.data.shape).to(self.device)

            # Quantize weights
            # self.connection[connection_key].weight.data = self.quant_128(self.connection[connection_key].weight.data, a_max=1.0) # quantize weights to [-127;127]

        if is_bias == True:
            biases:np.ndarray = g_nn.parameters["bias"]
            biases_index:int = 0
            for layer in self.layers.values():
                layer["bias"] = torch.tensor(biases[biases_index:biases_index+layer["size"]]).to(self.device)
                biases_index += layer["size"]


class ANN_Runner():
    def __init__(self, config_path:str):
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path, ["Genome_NN"])
        self.weights_layers_indexes:Dict[str, Tuple[np.ndarray, np.ndarray]] = None
        self.forward_config, self.forward_order = self.set_forward_config(self.config["Genome_NN"]["forward"])
        self.net_torch:NN_Custom_torch = None
        self.is_net_set:bool = False
        self.device = Index_Manager.device

    def get_weight_layers_indexes(self, synapses_indexes:Tuple[np.ndarray, np.ndarray], source_indexes:np.ndarray, target_indexes:np.ndarray) -> np.ndarray:
        input_synpases_indexes = np.where(np.isin(synapses_indexes[0], source_indexes))[0]
        target_synapse_indexes = np.where(np.isin(synapses_indexes[1][input_synpases_indexes], target_indexes))[0]
        return (synapses_indexes[0][input_synpases_indexes[target_synapse_indexes]], synapses_indexes[1][input_synpases_indexes[target_synapse_indexes]])


    def set_net_torch_population(self, population:Dict[int, Genome_NN], is_bias:bool) -> None:
        for genome in population.values():

            genome.net_torch:NN_Custom_torch = NN_Custom_torch(genome, self.forward_config, self.forward_order).to(self.device)
            if self.weights_layers_indexes == None:
                self.weights_layers_indexes:Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                for source_name, target_name in genome.nn.architecture_layers:
                    self.weights_layers_indexes[source_name + "->" + target_name] = self.get_weight_layers_indexes(genome.nn.synapses_indexes, genome.nn.architecture_neurons[source_name]["neurons_indexes"], genome.nn.architecture_neurons[target_name]["neurons_indexes"])
            genome.net_torch.set_parameters_2(genome, self.weights_layers_indexes, is_bias)


    def unset_net_torch_population(self, population:Dict[int, Genome_NN]) -> None:
        for genome in population.values():
            genome.net_torch = None
    

    def run_SL(self, population:Dict[int, Genome_NN], inputs:np.ndarray) -> Dict[int, np.ndarray]:
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        results:Dict[int, np.ndarray] = {}
        for genome in population.values():
            results[genome.id] = genome.net_torch(inputs).detach().numpy()

        return results
    
    def run_SL_v2(self, population:Dict[int, Genome_NN], indexes:List[int], inputs:np.ndarray, is_bias:bool, is_dynamic_topology:bool) -> List[float]:
        inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        results:List[float] = []
        for index in indexes:
            genome = population[index]

            # 1 - Set net_torch and parameters
            if is_dynamic_topology == True or self.is_net_set == False: # Like NEAT etc...
                self.net_torch:NN_Custom_torch = NN_Custom_torch(genome, self.forward_config, self.forward_order).to(self.device)
                if self.weights_layers_indexes == None:
                    self.weights_layers_indexes:Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                    for source_name, target_name in genome.nn.architecture_layers:
                        self.weights_layers_indexes[source_name + "->" + target_name] = self.get_weight_layers_indexes(genome.nn.synapses_indexes, genome.nn.architecture_neurons[source_name]["neurons_indexes"], genome.nn.architecture_neurons[target_name]["neurons_indexes"])
                self.is_net_set = True
            
            self.net_torch.set_parameters_2(genome, self.weights_layers_indexes, is_bias, is_dynamic_topology)
            
            # 2 - Run net_torch
            results.append(self.net_torch(inputs).detach().numpy())

        # 3 - Return results
        return np.array(results)

    def run_RL(self, population:Dict[int, Genome_NN], inputs:Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        results:Dict[int, np.ndarray] = {}
        for genome_id, genome in population.items():
            results[genome_id] = genome.net_torch(torch.tensor(inputs[genome_id], dtype=torch.float32).to(self.device)).detach().numpy()

        return results

    def run_RL_v2(self, population:Dict[int, Genome_NN], indexes:List[int], inputs:Dict[int, np.ndarray], is_bias:bool, is_dynamic_topology:bool) -> List[float]:
        results:Dict[int, np.ndarray] = {}
        for index in indexes:
            genome = population[index]

            # 1 - Set net_torch and parameters
            if is_dynamic_topology == True or self.is_net_set == False: # Like NEAT etc...
                self.net_torch:NN_Custom_torch = NN_Custom_torch(genome, self.forward_config, self.forward_order).to(self.device)
                if self.weights_layers_indexes == None:
                    self.weights_layers_indexes:Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                    for source_name, target_name in genome.nn.architecture_layers:
                        self.weights_layers_indexes[source_name + "->" + target_name] = self.get_weight_layers_indexes(genome.nn.synapses_indexes, genome.nn.architecture_neurons[source_name]["neurons_indexes"], genome.nn.architecture_neurons[target_name]["neurons_indexes"])
                self.is_net_set = True
            
            self.net_torch.set_parameters_2(genome, self.weights_layers_indexes, is_bias, is_dynamic_topology)
            
            # 2 - Run net_torch
            results[index] = self.net_torch(torch.tensor(inputs[index], dtype=torch.float32).to(self.device)).detach().numpy()

        # 3 - Return results
        return results


    def set_forward_config(self, forward_from_config:str) -> Dict[str, Dict[str, Any]]:
        forward_order = []
        forward_config:Dict[str, Dict[str, Any]] = {}
        for layer in re.split(r',(?![^()]*\))', forward_from_config): # thanks gpt for the regex...
            forward_order.append(layer.replace(" ", ""))
        index_O = None
        for index, forward in enumerate(forward_order):
            forward_config.update(self.get_layers(forward))
            if "->" not in forward and "O_" in forward:
                index_O = index
        if index_O != None:
            forward_order.pop(index_O)
        if "O" not in forward_config:
            forward_config["O"] = {"activation": self.get_activation(["raw"]), "norm": False}
        return forward_config, forward_order

    def get_layers(self, forward:str):
        layers:List[str] = []
        param:List[str] = []
        forward_config:Dict[str, Dict[str, Any]] = {}

        # 0. Set O layer config
        if ("O" == forward or "O_" in forward) and "->" not in forward:
            forward_config["O"] = {"activation": self.get_activation(forward.split("_")[1:]), "norm": "norm" in forward}
            return forward_config

        forward_config[forward] = {"layer_input": {}, "layer_output": {}}
        layers_and_param:List[str] = forward.split("->")
        
        # 1. Set Layers INTPUTs config
        layers_and_param_in:str = layers_and_param[0]
        if "(" not in layers_and_param_in:
            layers.append(layers_and_param_in) # get layers
            param.append(None) # in this case, there is no param
        else:
            layers = layers_and_param_in.split("(")[1].split(")")[0].split(",") # get layers
            param = layers_and_param_in.split("(")[1].split(")")[1].split("_") # get param (e.g "relu_norm" -> ["relu", "norm"])
            param = [x for x in param if x] # remove empty string

        layer_conf = []
        for layer in layers:
            if "_prev" in layer or "_prev_" in layer: # get if use previous output (e.g "H1_prev" -> ["H1", "output_prev"])
                layer_conf.append([layer.replace("_prev_", '').replace("_prev", ''), "output_prev"])
            else: # get if use current output (e.g "H1" -> ["H1", "output"])
                layer_conf.append([layer, "output"])
        layers = layer_conf


        forward_config[forward]["layer_input"]["layer"] = layers
        forward_config[forward]["layer_input"]["activation"] = self.get_activation(param)
        forward_config[forward]["layer_input"]["norm"] = "norm" in param


        # 2. Set Layers OUTPUTs config
        layers_and_param_out:str = layers_and_param[1]
        if "(" not in layers_and_param_out:
            layer_name = layers_and_param_out.split("_")[0]
            param = layers_and_param_out.split("_")[1:]
            activation = self.get_activation(param)
            norm = "norm" in param
            forward_config[forward]["layer_output"][layer_name] = {"activation": activation, "norm": norm}
        else:
            elements = layers_and_param_out.replace("(", "").replace(")", "").split(",")
            for element in elements:
                layer_name = element.split("_")[0]
                param = element.split("_")[1:]
                activation = self.get_activation(param)
                norm = "norm" in param
                forward_config[forward]["layer_output"][layer_name] = {"activation": activation, "norm": norm}
        return forward_config

    def get_activation(self, activation_list:List[str]) -> Callable:
        for activation in activation_list:
            if activation == "relu":
                return torch.relu
            elif activation == "tanh":
                return torch.tanh
            elif activation == "sigmoid":
                return torch.sigmoid
            elif activation == "raw":
                return lambda x: x
        return lambda x: x


    def rescale(self, x, src_min, src_max, dst_min, dst_max, round_int=False):
        """
        Remappe linéairement x de [src_min, src_max] vers [dst_min, dst_max].
        Si round_int=True, renvoie des entiers (round to nearest, banker’s rounding).
        """
        # S’assure que tout est un tensor sur le même device / dtype que x
        x = torch.as_tensor(x)
        device, dtype = x.device, x.dtype

        src_min = torch.as_tensor(src_min, dtype=dtype, device=device)
        src_max = torch.as_tensor(src_max, dtype=dtype, device=device)
        dst_min = torch.as_tensor(dst_min, dtype=dtype, device=device)
        dst_max = torch.as_tensor(dst_max, dtype=dtype, device=device)

        y = (x - src_min) / (src_max - src_min) * (dst_max - dst_min) + dst_min

        if round_int:
            y = torch.round(y)          # décimales=0 par défaut (PyTorch ≥2.1)
        return y

# def check_values(values):
#     if np.any(np.isnan(values)) or np.any(np.isinf(values)) or np.any(np.abs(values) > 100_000_000):
#         print("values:", values)
#         raise ValueError("Invalid value detected for actuator")

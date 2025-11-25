import sys

sys.setrecursionlimit(100000)
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.Fitness import Fitness_Manager
from evo_simulator.GENERAL.Population import Population_NN as Population
from evo_simulator import TOOLS

from snn_simulator.runner_api_cython import Runner_Info
from snn_simulator.runner_api_cython import Runner as SNN_Runner
from ann_simulator.runner import ANN_Runner

from problem.Problem import Problem

import random
from typing import List, Dict, Tuple, Any
import numpy as np

class Supervised_Manager(Runner_Info, Problem):
    def __init__(self, config_path:str, features:np.ndarray, labels:np.ndarray, nb_generations:int, is_continuous_label:bool=False, is_gpu:bool=False):
        Runner_Info.__init__(self, config_path)
        Problem.__init__(self)
        # Public variables
        self.config_path:str = config_path
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path, ["Genome_NN", "NEURO_EVOLUTION"])
        self.network_type:str = self.config["Genome_NN"]["network_type"]
        self.nb_generations:int = nb_generations
        self.is_gpu:bool = is_gpu
        self.generation = 0
        self.is_continuous_label:bool = is_continuous_label

        if self.network_type == "SNN":
            self.runner = SNN_Runner(self.config_path, is_gpu=self.is_gpu) # Initialize the Runner
        elif self.network_type == "ANN":
            self.runner = ANN_Runner(self.config_path)
        self.is_bias:bool = TOOLS.is_config_section(config_path, "bias_neuron_parameter")

        # Private variables
        if isinstance(features, np.ndarray) == False and features == None: return
        if isinstance(labels, np.ndarray) == False and labels == None: return

        if self.is_continuous_label == True:
            features = features.reshape(features.shape[1], features.shape[0])
            labels   = labels.reshape(labels.shape[1], labels.shape[0])
            features = labels # for test
        self.__batch_index:int = 0
        self.__build_batches(features, labels) # create batches of data set
    

    def run(self, population:Population, generation:int, seed:int=None, indexes:np.ndarray | List[int] =None) -> Population:  
        self.generation = generation
                        
        # 1 - Select Batch for the run
        features_batch, labels_batch = self.__select_random_batch(generation)
        
        # 2 - RUN NNs
        if self.network_type == "SNN":
            record:np.ndarray = self.run_snns(population, indexes, features_batch)

        elif self.network_type == "ANN":
            record: Dict[int, np.ndarray] = self.run_anns(population, indexes, features_batch)
        
        # 3 - CHECK SNNs ACCURACY AND UPDATE GENOMES FITNESS
        nn_accuracy = self.__evaluate_accuracy_and_update_genomes_fitnesses_array(record, labels_batch)
        return nn_accuracy

    def run_anns(self, population:Population, indexes:np.ndarray, features:np.ndarray) -> Dict[int, np.ndarray]:
        # print("----------------------------RUN ANNs----------------------------")
        # 0 - Check if the number of inputs of the genome is equal to the number of features of the data set
        if indexes is None: indexes = population.population_genome_ids
        if population.nb_inputs != len(features[0]):
            raise Exception(str("The number of inputs of the genome (" + str(population.nb_inputs) + ") is different from the number of features of the data set (" + str(len(features[0]))+ ")"))

        # self.runner.set_net_torch_population(genomes, indexes, self.is_bias)
        records:np.ndarray = self.runner.run_SL_v2(population.population, indexes, features, self.is_bias, population.is_dynamic_topology)
        # self.runner.unset_net_torch_population(genomes, indexes) # I don't think this is necessary
        return records


    def run_snns(self, population:Population, indexes:np.ndarray, features:np.ndarray) -> Dict[str, Dict[int, np.ndarray]]:
        # print("----------------------------RUN SNNs----------------------------")
        # 0 - Set indexes
        if indexes is None: indexes = population.population_genome_ids

        if self.generation == 0:
            if "input" not in self.record_layer:
                self.output_indexes:np.ndarray = (population.genome_core.nn.outputs["neurons_indexes_formated"] - population.genome_core.inputs).flatten()
            else:
                self.output_indexes:np.ndarray = population.genome_core.nn.outputs["neurons_indexes_formated"].flatten()

        # 1 - Check if the number of inputs of the genome is equal to the number of features of the data set
        if population.nb_inputs != len(features[0]):
            raise Exception(str("The number of inputs of the genome (" + str(population.nb_inputs) + ") is different from the number of features of the data set (" + str(len(features[0]))+ ")"))

        # 2 - Initialize NNs in the Runner
        self.runner.init_networks(population, indexes, self.generation == self.nb_generations -1) # Initialize the networks

        # # 3 - Initialize Inputs in the Runner
        # self.runner.init_inputs_networks_SL(features) # Set the inputs spikes

        # 4 - Run NNs in the Runner
        records = self.runner.run(features, is_raw_data=True)

        return records


    def __evaluate_accuracy_and_update_genomes_fitnesses_array(self, record:np.ndarray, labels:np.ndarray) -> np.ndarray:

        if self.network_type == "SNN" and self.is_continuous_label == False:
            nets_accuracy:np.ndarray = Fitness_Manager.deterministic_neuron_output_accuracy_numerical_pop(record[:, :, self.output_indexes], labels)
        elif self.network_type == "SNN" and self.is_continuous_label == True:
            # print("record", record[0], record[0].shape)
            # print("labels", labels, labels.shape)
            TOOLS.compare_signals(record[0], labels)
            exit()
            nets_accuracy:np.ndarray = Fitness_Manager.Mean_squared_Error(record[:, :, self.output_indexes], labels)

        elif self.network_type == "ANN" and self.is_continuous_label == False:
            nets_accuracy:np.ndarray = Fitness_Manager.deterministic_neuron_output_accuracy_numerical_pop(record, labels)
        elif self.network_type == "ANN" and self.is_continuous_label == True:
            nets_accuracy:np.ndarray = Fitness_Manager.Mean_squared_Error(record, labels)
        return nets_accuracy

    def __build_batches(self, features:List[List[float]], labels:List[float]):
        self.batch_features:int = min(self.batch_features, len(labels))
        self.batch_running:int = min(self.batch_running, self.batch_features)

        if self.is_continuous_label == False:
            data_set:Tuple[(List[List[float]], List[float])] = list(zip(features, labels))
            random.shuffle(data_set)
            data_set = list(zip(*data_set))
            labels:List[float] = data_set[1]
            features:List[List[float]] = data_set[0]
            self.labels_batches = np.split(labels, np.arange(self.batch_features, len(labels), self.batch_features))
            self.features_batches = np.split(features, np.arange(self.batch_features, len(features), self.batch_features))
            self.__batch_index:np.ndarray = np.random.randint(len(self.labels_batches), size=self.nb_generations)
        else:
            self.labels_batches = np.split(labels, np.arange(self.batch_features, len(labels), self.batch_features))
            self.features_batches = np.split(features, np.arange(self.batch_features, len(features), self.batch_features))
            self.__batch_index:np.ndarray = np.arange(len(self.labels_batches))

        # print("self.__batch_index", self.__batch_index, self.__batch_index.shape)
        # exit()

    def __select_random_batch(self, generation) -> Tuple[np.ndarray, np.ndarray]:
        features = self.features_batches[self.__batch_index[generation]]
        labels = self.labels_batches[self.__batch_index[generation]]
        return features, labels

    def record_energy(self, genomes:Dict[int, Genome_NN], records:Dict[str,Dict[str, np.ndarray]]):
            has_spike:bool = "spike" in self.record_layer
            has_augmented:bool = "augmented" in self.record_layer
            if has_spike == False and has_augmented == False: return

            for id, genome in genomes.items():
                if has_spike:
                    genome.info["energy_spikes_sum"] = records["spike"][id].sum()

                if has_augmented:
                    genome.info["energy_augmented_spikes_sum"] = records["augmented"][id].sum()
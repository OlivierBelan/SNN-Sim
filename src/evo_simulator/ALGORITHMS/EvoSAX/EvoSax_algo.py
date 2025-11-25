from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.Attribute import Attribute_Paramaters
import evo_simulator.TOOLS as TOOLS
from evo_simulator.GENERAL.Index_Manager import get_new_genome_id
from evo_simulator.GENERAL.Population import Population_NN as Population
from typing import Dict, Any, List, Union
import numpy as np
import time

from ALGORITHMS.EvoSAX.SNES_numpy import SNES_numpy
from ALGORITHMS.EvoSAX.FitnessShaper_numpy import FitnessShaper


class EvoSax_algo(Algorithm):
    def __init__(self, config_path_file:str, algo_name:str, attribute_mananger:Attribute_Paramaters, extra_info:Dict[Any, Any] = None) -> None:
        Algorithm.__init__(self, config_path_file, algo_name, attribute_mananger, extra_info)

        # Initialize configs
        self.config_es:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, [algo_name,"Genome_NN","NEURO_EVOLUTION", "Runner_Info"])

        self.pop_size:int = int(self.config_es[algo_name]["pop_size"])
        self.is_first_generation:bool = True
        self.verbose:bool = True if self.config_es[algo_name]["verbose"] == "True" else False
        self.optimization_type:str = self.config_es["NEURO_EVOLUTION"]["optimization_type"]
        self.algo_name:str = algo_name
        self.config_path_file:str = config_path_file
        self.generation:int = 0
        return


    def __init_es(self, config_path_file:str=None) -> None:

        self.optimizer:Union[SNES_numpy] = self.__init_snes(config_path_file, self.parameters_size)


        self.es_hyperparameters = self.optimizer.default_params
        self.params_state = self.optimizer.initialize(None, self.es_hyperparameters)

    def __init_snes(self, config_path_file:str, parameters_size:int):
        config_snes:Dict[str, Any] = TOOLS.config_function(config_path_file, ["NES"])["NES"]
        
        self.optimizer:SNES_numpy = SNES_numpy(
                                    popsize=self.pop_size, 
                                    num_dims=parameters_size, 

                                    sigma_init=float(config_snes["sigma_init"]), 
                                    mean_decay=float(config_snes["mean_decay"]),

                                    seed=None,
                                    fitness_shaper=None
                                    )

        return self.optimizer

    def run(self, global_population:Population) -> Population:

        self.population_manager = global_population
        if self.is_first_generation == True: 
            start_time = time.time()
            self.__first_generation(self.population_manager)
            self.population_manager.sync_genomes_to_population(is_sync_population_to_genome=True)
            self.__update_population_parameter_population(self.population_manager)
            self.population_manager.sync_population_to_genomes()
            global_population = self.population_manager
            print(self.name+": First generation time:", time.time() - start_time, "s")
            return global_population
        
        # 1 - Update
        self.__update_es_by_fitness(self.population_manager)

        self.generation += 1

        # 2 - Update population parameters
        self.__update_population_parameter_population(self.population_manager)

        # 3 - Update population
        global_population.population = self.population_manager.population

        return global_population
            
    def __first_generation(self, population_manager:Population) -> None:
        self.is_first_generation = False
        population_manager.first_generation()

        self.genome_core:Genome_NN = population_manager.genome_core

        self.network_type = population_manager.network_type
        self.neuron_params_names = population_manager.neuron_params_names
        self.synapse_params_names = population_manager.synapse_params_names

        self.neuron_parameters_size:int = population_manager.neuron_parameters_size
        self.synapse_parameters_size:int = population_manager.synapse_parameters_size
        self.layer_parameters_size:int = population_manager.layer_parameters_size
        self.synapses_actives_indexes:np.ndarray = population_manager.synapses_actives_indexes

        self.is_combinatorial_modulo = population_manager.is_combinatorial_modulo
        self.is_action_dynamic_RL = population_manager.is_action_dynamic_RL_evolution


        self.is_energy = population_manager.is_energy
        if self.is_energy == True:
            self.energy_length = population_manager.energy_length
            self.energy_reset_value = population_manager.energy_reset_value


        self.params_to_update = population_manager.params_to_update
        self.parameters_size = population_manager.parameters_size
        self.__init_es(config_path_file=self.config_path_file)
        return

    def __update_population_parameter_population(self, population_manager:Population) -> None:
        # 1 - Get parameters algorithms
        self.population_parameters_from_optimizer, self.state = self.optimizer.ask(None, self.params_state, self.es_hyperparameters)
        
        param_index:int = 0
        sigma_index:int = 0
        population_params:Dict[str, np.ndarray] = population_manager.parameters

        # 2 - Update parameters in the population
        if self.is_combinatorial_modulo == True or self.is_action_dynamic_RL == True:
            param_index, sigma_index = population_manager.update_modules(param_index, sigma_index, self.population_parameters_from_optimizer, self.state.sigma, self.generation)

        for param in self.params_to_update:

            # 2.3 Update Neuron parameters
            if param in self.neuron_params_names:

                if param == "energy":
                    population_params[param][:,:,:] = self.population_parameters_from_optimizer[:, param_index:param_index+population_manager.energy_size].reshape(population_params[param].shape)
                    param_index += population_manager.energy_size

                else: # neuron parameters
                    population_params[param][:,:] = self.population_parameters_from_optimizer[:, param_index:param_index+population_manager.nb_neurons_population]
                    param_index += self.neuron_parameters_size

            # 2.4 Update Synapse parameters
            elif param in self.synapse_params_names:
                population_params[param][:, self.synapses_actives_indexes[0], self.synapses_actives_indexes[1]] = self.population_parameters_from_optimizer[:, param_index:param_index+self.synapse_parameters_size]
                param_index += self.synapse_parameters_size


        param_index *= self.pop_size
        population_manager.sync_population_to_genomes()

    def __update_es_by_fitness(self, population_manager:Population) -> None:
        self.population_manager.update_info()
        genomes_dict:Dict[int, Genome_NN] = population_manager.population
        fitnesses:List[int] = []
        for genome in genomes_dict.values():
            fitnesses.append(genome.fitness.score)
        fitnesses:np.ndarray = np.array(fitnesses)

        fit_shaper = FitnessShaper(
                        # centered_rank=True,
                        # z_score=True,
                        # w_decay=0.1,
                        # maximize=True if self.optimization_type == "maximize" and isinstance(self.optimizer, PGPE_numpy) == False else False,
                        maximize=True,
                        )

        fitnesses = fit_shaper.apply(self.population_parameters_from_optimizer, fitnesses)

        self.params_state = self.optimizer.tell(self.population_parameters_from_optimizer, fitnesses, self.params_state, self.es_hyperparameters)

import sys
sys.setrecursionlimit(100000)
from snn_simulator.runner_api_cython import Runner as SNN_Runner
from snn_simulator.runner_api_cython import Runner_Info
from ann_simulator.runner import ANN_Runner

from evo_simulator.GENERAL.Population import Population_NN as Population
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator import TOOLS


from problem.Problem import Problem
from problem.RL.ENVIRONNEMENT_VEC.ENVIRONNEMENT_VECTORIZE import Environment_Manager_Vec, Environment_Vec
from problem.RL.ENVIRONNEMENT import Environment_Manager, Environment
from snn_simulator.snn_decoder import Decoder


import time
from typing import List, Dict, Any, Callable, Tuple
import numpy as np
import sys
import psutil

class Reinforcement_Manager(Runner_Info, Problem):
    def __init__(self, environment_builder:Callable, config_path:str, seeds:np.ndarray=None, nb_episode:int=None, nb_generations:int=None, cpu_id:int=None, cpu_affinity:bool=False, is_gpu:bool= False):

        # if cpu_id is not None and sys.platform == "linux" and cpu_affinity == True:
        #     nb_max_cpu = psutil.cpu_count(logical=True)
        #     id_cpu = nb_max_cpu - cpu_id - 1
        #     print("CPU ID:", id_cpu)
        #     psutil.Process().cpu_affinity([id_cpu])        
        self.is_gpu:bool = is_gpu
        self.reset(environment_builder, config_path, seeds, nb_episode, nb_generations)

    def reset(self, environment_builder:Callable, config_path:str, seeds:np.ndarray=None, nb_episode:int=None, nb_generations:int=None) -> None:
        Runner_Info.__init__(self, config_path, nb_episode=nb_episode)
        Problem.__init__(self)

        self.config_path:str = config_path
        self.seeds_from_param:np.ndarray= seeds

        self.is_vec:bool = False
        self.environment_builder:Callable = environment_builder
        if "Vec" in self.environment_builder.name:
            self.is_vec:bool = True
            self.environment_manager:Environment_Manager_Vec = Environment_Manager_Vec(self.environment_builder) # Will be Initialized in the run_generation function
        else:
            self.environment_manager:Environment_Manager = Environment_Manager(self.environment_builder) # Will be Initialized in the run_generation function

        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path, ["Genome_NN", "NEURO_EVOLUTION"])
        self.network_type:str = self.config["Genome_NN"]["network_type"]
        self.nb_original_outputs:int = int(self.config["Genome_NN"]["outputs"])
        self.input_multiplicator:int = int(self.config["Genome_NN"]["inputs_multiplicator"])
        self.is_dynamic_topology:bool = True if self.config["Genome_NN"]["is_dynamic_topology"] == "True" else False
        self.is_bias:bool = TOOLS.is_config_section(config_path, "bias_neuron_parameter")
        self.nb_generations:int = nb_generations
        self.__init_runner()
    

    def __init_runner(self) -> None:
        if self.network_type == "SNN":
            self.runner = SNN_Runner(self.config_path, self.nb_episode, self.is_gpu) # Initialize the Runner
            self.snn_decoder = Decoder(self.config_path)
        elif self.network_type == "ANN":
            self.runner = ANN_Runner(self.config_path)
        else:
            raise Exception("The network type is not recognized", self.network_type, "The network type must be SNN or ANN")

    def run(self, population:Population, generation:int, seeds:np.ndarray = None, indexes:np.ndarray = None, obs_max:np.ndarray = None, obs_min:np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  
        self.generation = generation
        
        # 0 - Set population received and update the observation_max and observation_min (if needed)
        if indexes is not None:
            self.pop_indexes:np.ndarray = indexes
            self.genome_indexes:List[int] = self.pop_indexes.tolist()
            genomes_dict:Dict[int, Genome_NN] = population.population
            self.genomes:Dict[int, Genome_NN] = {genome_id:genomes_dict[genome_id] for genome_id in self.genome_indexes}
        else:
            self.pop_indexes:np.ndarray = population.population_genome_ids
            self.genome_indexes:List[int] = self.pop_indexes.tolist()
            self.genomes:Dict[int, Genome_NN] = population.population
        
        if obs_max is not None: self.environment_manager.update_observation_min_max(obs_max, obs_min)
        if seeds is not None: self.seeds_from_param = seeds
        self.environment_manager.update_action_min_max(self.genomes, population.extra_info)

 
        # 1 - Check if the number of inputs of the genomes is equal to the number of inputs of the environment
        if population.nb_inputs != self.environment_manager.input_size:
            raise Exception(str("The number of inputs of the genome (" + str(population.nb_inputs) + ") is different from the number of inputs of the environment (" + str(self.environment_manager.input_size)+ ")"))

        # 2 - RUN NNs
        if self.network_type == "SNN" and self.is_vec == False: fitnesses = self.__run_snns(population)
        if self.network_type == "SNN" and self.is_vec == True:  fitnesses = self.__run_snns_vec(population)
        elif self.network_type == "ANN": fitnesses = self.__run_anns(self.genomes)
        return fitnesses, self.obs_max, self.obs_min


    def __run_snns(self, population:Population) -> np.ndarray:
        seeds_batch:List[int] = None
        episodes:List[int] = None
        self.seeds_list:List[int] = self.__get_seeds()
        self.__reset_genomes_fitness(self.genomes)
        nns_cython_dict = self.runner.init_networks(population, self.pop_indexes, self.generation == self.nb_generations)
        envs_dict:Dict[int, List[Environment]] = self.environment_manager.create_environments(genomes_ids=self.genome_indexes, seeds=self.seeds_list[:self.nb_episode].tolist())
        # print("Neuron_actives_total:", self.runner.neurons_total_used, "; Synapses_actives_total:", self.runner.synapses_total_used)

        if "input" not in self.record_layer: output_indexes_record:np.ndarray = population.genome_core.nn.outputs["neurons_indexes"] - population.genome_core.nn.nb_inputs
        else: output_indexes_record:np.ndarray = population.genome_core.nn.outputs["neurons_indexes"]
        output_indexes:np.ndarray = population.genome_core.nn.outputs["neurons_indexes"]
        # print("output_indexes", output_indexes)
        # print("output_indexes_record", output_indexes_record)

        is_actives:bool = True
        for i in range(0, self.nb_episode, self.nb_episode):
            seeds_batch = self.seeds_list[i:i+self.nb_episode].tolist()
            episodes:List[int] = list(range(i, i+self.nb_episode))

            # 0 - Init RL episode (create/reset environments and inits nns)
            self.environment_manager.reset(seeds_batch)

            while is_actives == True:

                # 1 - (RL Observation) Init inputs networks &  Check if genome_inputs == observation_inputs
                observation_spikes:np.ndarray = self.environment_manager.encoding_observation_to_snn_input()

                # 1.1 - Shape the observation_spikes to match the input_multiplicator
                if self.input_multiplicator > 1: observation_spikes = np.repeat(observation_spikes, self.input_multiplicator, axis=2)

                # 2 - (RL Action) Run the networks (cython) & get the actions
                actions_dict = self.runner.run(observation_spikes, is_raw_data=False)[self.record_type]

                # 3 - Decoding Spikes
                actions_dict = self.decoding_spikes(actions_dict, output_indexes, output_indexes_record, self.genomes)

                # 3 - (RL Update/Step) Update the environments with the actions
                is_actives_pre = self.environment_manager.update_environments(self.genomes, actions_dict, episodes, generation=self.generation)

                is_actives = is_actives_pre


        # 5 - Set fitnesses of genomes after the end of evaluation
        fitnesses, self.obs_max, self.obs_min = self.environment_manager.fitness_end(self.genomes, list(range(0, self.nb_episode)))

        return fitnesses


    def __run_snns_vec(self, population:Population) -> Dict[int, Genome_NN]:
        seeds_batch:List[int] = None
        self.seeds_list:List[int] = self.__get_seeds()
        self.__reset_genomes_fitness(self.genomes)
        nns_cython_dict = self.runner.init_networks(population, self.pop_indexes, self.generation == self.nb_generations)
        # envs_dict:Dict[int, List[Environment]] = self.environment_manager.create_environments(genomes_ids=self.genome_indexes, seeds=self.seeds_list[:self.nb_episode].tolist())
        envs:List[Environment_Vec] = self.environment_manager.create_environments(nb_envs=len(self.genomes), seeds=self.seeds_list[:self.nb_episode].tolist())

        is_actives:bool = True
        if "input" not in self.record_layer: output_indexes_record:np.ndarray = population.genome_core.nn.outputs["neurons_indexes"] - population.genome_core.nn.nb_inputs
        else: output_indexes_record:np.ndarray = population.genome_core.nn.outputs["neurons_indexes"]
        output_indexes:np.ndarray = population.genome_core.nn.outputs["neurons_indexes"]

        # print("output_indexes", output_indexes)
        # print("output_indexes_record", output_indexes_record)

        for i in range(0, self.nb_episode, self.nb_episode):
            seeds_batch = self.seeds_list[i:i+self.nb_episode].tolist()

            # 0 - Init RL episode (create/reset environments and inits nns)
            self.environment_manager.reset(seeds_batch)

            while is_actives == True:
                    
                # 1 - (RL Observation) Init inputs networks &  Check if genome_inputs == observation_inputs
                observation_spikes:np.ndarray = self.environment_manager.encoding_observation_to_snn_input()

                # 1.1 - Shape the observation_spikes to match the input_multiplicator
                if self.input_multiplicator > 1: observation_spikes = np.repeat(observation_spikes, self.input_multiplicator, axis=2)

                # 2 - (RL Action) Run the networks (cython) & get the actions
                actions_vec:np.ndarray = self.runner.run(observation_spikes, is_raw_data=True)

                # 3 - Decoding Spikes
                actions_vec = self.decoding_spikes(actions_vec, output_indexes, output_indexes_record, self.genomes)

                # 3 - (RL Update/Step) Update the environments with the actions
                is_actives:bool = self.environment_manager.update_environments(actions_vec, output_indexes_record)

        fitnesses, self.obs_max, self.obs_min = self.environment_manager.fitness_end(self.genomes)
        return fitnesses

    def __run_anns(self, genomes:Dict[int, Genome_NN]) -> Dict[int, Genome_NN]:
        seeds_batch:List[int] = None
        self.seeds_list:List[int] = self.__get_seeds()
        self.__reset_genomes_fitness(genomes)

        for i in range(self.nb_episode):

            is_actives:bool = True
            envs_dict:Dict[int, List[Environment]] = self.environment_manager.create_environments(genomes_ids=self.genome_indexes, seeds=[self.seeds_list[i]])
            seeds_batch = [self.seeds_list[i]]

            # 0 - Init RL episode (create/reset environments and inits nns)
            self.environment_manager.reset(seeds_batch)

            while is_actives == True:
                    
                # 1 - (RL Observation) Init inputs networks &  Check if genome_inputs == observation_inputs
                observation_dict:Dict[int, np.ndarray] = self.environment_manager.encoding_observation_to_ann_input()

                # 2 - (RL Action) Run the networks (cython) & get the actions
                actions_dict:Dict[int, np.ndarray] = self.runner.run_RL_v2(genomes, self.genome_indexes, observation_dict, is_bias=self.is_bias, is_dynamic_topology=self.is_dynamic_topology)

                # 3 - (RL Update/Step) Update the environments with the actions
                is_actives = self.environment_manager.update_environments(genomes, actions_dict, seeds_batch, generation=self.generation)

        # 5 - Set fitnesses of genomes after the end of evaluation
        fitnesses, self.obs_max, self.obs_min = self.environment_manager.fitness_end(genomes, self.seeds_list.tolist())
        return fitnesses



    def decoding_spikes(self, actions_dict:Dict[int, np.ndarray], output_indexes_nn:np.ndarray, output_indexes_record:np.ndarray, genomes:Dict[int, Genome_NN]) -> Dict[int, np.ndarray]:
        if self.decoder == "augmented": return actions_dict

        for id, actions in actions_dict.items():

            if self.decoder == "max_spikes": 
                actions_dict[id] = np.array([self.snn_decoder.max_spikes(action[output_indexes_record], self.nb_original_outputs) for action in actions], dtype=np.float32)
            
            elif self.decoder == "augmented":
                pass

            elif self.decoder == "rate":
                actions_dict[id] = np.array([self.snn_decoder.rate(action[output_indexes_record], self.nb_original_outputs, self.ratio_max_output_spike) for action in actions], dtype=np.float32)
            
            elif self.decoder == "voltage":
                voltage_min:np.ndarray = genomes[id].nn.parameters["voltage"][output_indexes_nn] if self.is_voltages_min_decoder == True else self.voltage_min
                voltage_max:np.ndarray = genomes[id].nn.parameters["threshold"][output_indexes_nn] if self.is_threshold_max_decoder == True else self.voltage_max
                actions_dict[id] = self.snn_decoder.voltage(actions[:, output_indexes_record], voltage_min, voltage_max)
            
            elif self.decoder == "coeff":
                actions_dict[id] = np.array([self.snn_decoder.coefficient(action[output_indexes_record], genomes[id].nn.parameters["coeff"][output_indexes_nn], self.nb_original_outputs) for action in actions], dtype=np.float32)

        return actions_dict

    def __reset_genomes_fitness(self, genomes:Dict[int, Genome_NN]) -> None:
        for genome in genomes.values():
            genome.fitness.reset()

    def __get_seeds(self) -> np.ndarray:
        if self.seeds_from_param is not None:
            # if len(self.seeds_from_param) != self.nb_episode: raise Exception("The number of seeds given in the parameter is different from the number of episodes ->", self.nb_episode, "vs", len(self.seeds_from_param))
            return self.seeds_from_param
        else:
            return np.random.choice(np.arange(1e6), size=self.nb_episode, replace=False)
    
    def run_render(self, population:Population) -> None:
        if self.network_type == "SNN": self.run_render_SNN(population)
        elif self.network_type == "ANN": self.run_render_ANN(population)
        else: raise Exception("The network type is not recognized", self.network_type, "The network type must be SNN or ANN")

    def run_render_ANN(self, population:Population) -> None:
        genome:Genome_NN = population.population[0]
        obs_max:np.ndarray = genome.info["obs_max"]
        obs_min:np.ndarray = genome.info["obs_min"]
        seeds:np.ndarray = genome.info["seed"]

        if obs_max is None:
            obs_max:np.ndarray = np.full(genome.inputs, 5.0, dtype=np.float64) # 5 is arbitrary
            obs_min:np.ndarray = np.full(genome.inputs, -5.0, dtype=np.float64) # -5 is arbitrary

        genome.fitness.reset()

        # 0 - Set net_torch
        self.runner.set_net_torch_population({genome.id:genome}, is_bias=self.is_bias)


        for seed in seeds.tolist():
            env_render = self.environment_manager.create_new_env(-1, is_render=True)
            env_render.seed = seed
            env_render.reset(seed=seed)
            env_render.update_observation_min_max_function(obs_max, obs_min)
            env_render.update_action_min_max_render_function(genome)
            
            while env_render.terminated == False and env_render.truncated == False:
                # 0 - Render
                env_render.render()

                # 1 - Observation
                observation:np.ndarray = env_render.encoding_observation_to_network_input()

                # 2 - Action
                actions:Dict[int, np.ndarray] = self.runner.run_RL_v2({genome.id:genome}, [genome.id], {genome.id:np.array([observation])}, is_bias=self.is_bias, is_dynamic_topology=self.is_dynamic_topology)
                print("actions", actions[genome.id][0], "shape", actions[genome.id][0].shape)

                # 3 - Update
                env_render.update(actions[genome.id][0], genome, seed, generation=None)

                # 4 - Render
                # if self.env_render.is_QDgym == False:
                #     self.env_render.gym_env.render()
                # time.sleep(0.01)

        env_render.close()        
        # 5 - Unset net_torch
        self.runner.unset_net_torch_population({genome.id:genome})


        fitness:dict = env_render.fitness_end_function(population.population, [seed])[0]
        print("genome", genome, "id", genome.id)
        print("after genome.score", genome.fitness.score)
        print("genome.best_fitness", genome.info["best_episode_raw_score"])
        print("After render: best_episode_raw_score", fitness["best_episode_raw_score"])
        print("After render: mean_episode_raw_score", fitness["mean_episode_raw_score"])
        exit()

    def run_render_SNN(self, population:Population) -> None:
        genome:Genome_NN = population.population[0]
        obs_max:np.ndarray = genome.info["obs_max"]
        obs_min:np.ndarray = genome.info["obs_min"]
        seeds:np.ndarray = genome.info["seed"]


        if obs_max is None:
            obs_max:np.ndarray = np.full(genome.inputs, 5.0, dtype=np.float64) # 5 is arbitrary
            obs_min:np.ndarray = np.full(genome.inputs, -5.0, dtype=np.float64) # -5 is arbitrary

        genome.fitness.reset()
        nns_cython_dict = self.runner.init_networks(population, np.array([0], dtype=np.int32), False)


        if "input" not in self.record_layer: output_indexes_record:np.ndarray = genome.nn.outputs["neurons_indexes"] - genome.nn.nb_inputs
        else: output_indexes_record:np.ndarray = genome.nn.outputs["neurons_indexes"]
        output_indexes:np.ndarray = genome.nn.outputs["neurons_indexes"]

        for seed in seeds.tolist():
            env_render = self.environment_manager.create_new_env(-1, is_render=True)
            env_render.seed = seed
            env_render.reset(seed=seed)
            env_render.update_observation_min_max_function(obs_max, obs_min)
            env_render.update_action_min_max_render_function(genome)
            
            while env_render.terminated == False and env_render.truncated == False:
                # 0 - Render
                env_render.render()

                # 1 - Observation
                observation:np.ndarray = np.expand_dims((np.expand_dims(env_render.encoding_observation_to_network_input(), axis=0)), axis=0)
                # observation = np.array([[observation]], dtype=np.float64)

                # 1.1 - Shape the observation_spikes to match the input_multiplicator
                # if self.input_multiplicator > 1: observation = np.repeat(observation, self.input_multiplicator, axis=2)

                # 2 - Action
                actions_dict:Dict[int, np.ndarray] = self.runner.run(observation, is_raw_data=False)[self.record_type]
        
                # 2.bis - Decoding Spikes
                actions_dict = self.decoding_spikes(actions_dict, output_indexes, output_indexes_record, {0, genome})
        
                # 3 - Update
                env_render.update(actions_dict[0][0], genome, seed, generation=None)

                # 4 - Render
                # if self.env_render.is_QDgym == False:
                #  self.env_render.gym_env.render()
                # time.sleep(0.01)

            env_render.close()

            fitness:dict = env_render.fitness_end_function(population.population, [seed])[0]
            print("After render: best_episode_raw_score", fitness["best_episode_raw_score"])
            print("After render: mean_episode_raw_score", fitness["mean_episode_raw_score"])
        exit()
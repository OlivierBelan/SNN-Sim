from evo_simulator.GENERAL.Genome import Genome
from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Population import Population_NN as Population
from evo_simulator.GENERAL.Index_Manager import get_new_population_id
from evo_simulator.GENERAL.Attribute import Attribute_Paramaters

from problem.Problem import Problem

from typing import List, Dict, Tuple, Any, Union
import evo_simulator.TOOLS as TOOLS
from evo_simulator.Record.Record import Record
import numpy as np
import time
import ray
import os
import shutil
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

class Neuro_Evolution:

    def __init__(self, nb_generations:int, nb_runs:int, is_record:bool, config_path:str, device:str, nb_cpu:int, nb_gpu:int, cpu_gpu_ratio:List[int]=None, is_debug:bool=False):
        
        self.config_path:str = self.__build_config_cache(config_path, os.getcwd() + "/config_cache/")

        # 1- Initialize configs
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(self.config_path, ["NEURO_EVOLUTION", "Genome_NN", "Runner_Info"])
        self.debug:bool = is_debug
        
        self.nb_generations = nb_generations
        self.nb_runs = nb_runs if nb_runs != None else int(self.config["NEURO_EVOLUTION"]["nb_runs"])
        if self.config["NEURO_EVOLUTION"]["verbose"] == "True":
            self.verbose:bool = True
            self.verbose_minimal:bool = False
        elif self.config["NEURO_EVOLUTION"]["verbose"] == "False":
            self.verbose:bool = False
            self.verbose_minimal:bool = False
        elif self.config["NEURO_EVOLUTION"]["verbose"] == "minimal":
            self.verbose:bool = False
            self.verbose_minimal:bool = True
        
        self.is_render:bool = False
        self.is_record:bool = is_record
        self.record:Record = None
        self.network_type:str = self.config["Genome_NN"]["network_type"]
        self.nb_hidden_neurons:int = TOOLS.hiddens_from_config(self.config["Genome_NN"]["hiddens"])["nb_hiddens"]
        self.optimization_type:str = self.config["NEURO_EVOLUTION"]["optimization_type"] # maximize, minimize, closest_to_zero
        self.cpu_affinity:bool = False if self.config["Runner_Info"].get("cpu_affinity", None) == None else True if self.config["Runner_Info"]["cpu_affinity"] == "True" else False
        self.attributes_manager:Attribute_Paramaters = Attribute_Paramaters(self.config_path)

        if self.network_type == "SNN":
            import snn_simulator.build as build
            build.build_simulator()
            self.check_snn_refractory_and_delay()
            self.run_length:int = int(self.config["Runner_Info"]["run_time"])
            self.run_with_margin:int = int(self.config["Runner_Info"]["run_time_margin"]) + self.run_length
            self.dt:float = float(self.config["Runner_Info"]["dt"])
            self.decay_method:str = self.config["Runner_Info"]["decay_method"]
            self.encoder:str = self.config["Runner_Info"]["encoder"]
            self.decoder:str = self.config["Runner_Info"]["decoder"]
            self.record_run:str = self.config["Runner_Info"]["record"]
            self.encoder_record_text:str = self.encoder
            self.decoder_record_text:str = self.decoder

            # Voltage Decoding
            self.is_voltage_encoding:bool = True if self.decoder == "voltage" else False
            if self.decoder == "voltage":
                self.config.update(TOOLS.config_function(self.config_path, ["Voltage_Decoder"]))
                self.is_output_threshold:bool = False if self.config["Voltage_Decoder"]["disable_output_threshold"] == "True" else True
                self.voltage_min:str = self.config["Voltage_Decoder"]["voltage_min"]
                self.voltage_max:str = self.config["Voltage_Decoder"]["voltage_max"]
                self.decoder_record_text:str = self.decoder + "_voltage_min-" + self.voltage_min + "_voltage_max-" + self.voltage_max
                self.max_possible_spikes:int = 0

            # Rate Decoding / Augmented Decoding
            elif self.decoder == "rate":
                self.config.update(TOOLS.config_function(self.config_path, ["Rate_Decoder"]))
                self.decoding_ratio:float = float(self.config["Rate_Decoder"]["ratio_max_output_spike"])
                self.output_multiplier:int = int(self.config["Genome_NN"]["outputs_multiplicator"])
                self.max_possible_spikes:int = self.run_length * self.output_multiplier * self.decoding_ratio
                self.decoder_record_text:str = self.decoder + "_max_spikes-" + str(self.max_possible_spikes)

            elif self.decoder == "augmented":
                self.config.update(TOOLS.config_function(self.config_path, ["Augmented_Decoder"]))
                self.max_possible_spikes:int = self.config["Augmented_Decoder"]["spike_max"]
                self.decoder_record_text:str = self.decoder + "_spike_max-" + str(self.max_possible_spikes)

            # Combinatorial Encoding
            self.is_combinatorial_encoding:bool = True if self.encoder == "combinatorial" else False
            if self.is_combinatorial_encoding == True:
                self.config.update(TOOLS.config_function(self.config_path, ["Combinatorial_Encoder"]))
                self.combinatorial_factor:int  = int(self.config["Combinatorial_Encoder"]["combinatorial_factor"])
                if self.run_length % self.combinatorial_factor != 0: raise Exception("run_time must be a multiple of combinatorial_factor -> run_time % combinatorial_factor == 0: run_time =",self.run_length, "combinatorial_factor =", self.combinatorial_factor)
                self.combinatorial_combinaison_size:float = float(self.config["Combinatorial_Encoder"]["combinatorial_combinaison_size"])
                self.combinatorial_combinaison_size_max:int = int(float(self.config["Combinatorial_Encoder"]["combinatorial_combinaison_size_max"]))
                self.combinatorial_combinaison_noise:float = float(self.config["Combinatorial_Encoder"]["combinatorial_combinaison_noise"])
                self.combinatorial_combinaison_noise_decay:float = float(self.config["Combinatorial_Encoder"]["combinatorial_combinaison_noise_decay"])
                self.combinatorial_ocillator_noise:float = float(self.config["Combinatorial_Encoder"]["combinatorial_ocillator_noise"])
                self.combinatorial_ocillator_noise_decay:float = float(self.config["Combinatorial_Encoder"]["combinatorial_ocillator_noise_decay"])
                self.combinatorial_filter:str = self.config["Combinatorial_Encoder"]["combinatorial_filter"]
                if self.run_length > 50:
                    self.combinatorial_combinaisons:int = min(self.combinatorial_combinaison_size, self.combinatorial_combinaison_size_max)
                else:
                    self.combinatorial_combinaisons = min((2**(self.run_length//self.combinatorial_factor)), self.combinatorial_combinaison_size_max)
                self.combinatorial_combinaisons:int = min(self.combinatorial_combinaisons, self.combinatorial_combinaison_size)
                self.encoder_record_text:str = "combinatorial_factor-" + str(self.combinatorial_factor) + "_combinaison-" + str(self.combinatorial_combinaisons) + "_noise-" + str(self.combinatorial_combinaison_noise) + "_noise_decay-" + str(self.combinatorial_combinaison_noise_decay)
        
        if self.network_type == "ANN":
            self.is_bias = TOOLS.is_config_section(self.config_path, "bias_neuron_parameter")
            # self.is_layer_normalization = True if self.config["Genome_NN"]["is_layer_normalization"] == "True" else False

        # 2 - Times parameters
        self.run_nb:int = 0
        self.generation:int = 0
        self.solved_at_generation:int = -1
        self.generations_times_list:List[float] = []

        # 3 - Device parameters
        self.device:str = device
        self.is_gpu:bool = False
        self.is_cpu_gpu:bool = False
        if "cpu" in self.device and "gpu" in self.device: # cpu_gpu or gpu_cpu
            self.device:str = "cpu_gpu"
            self.cpu:int = nb_cpu
            self.gpu:int = nb_gpu
            self.is_gpu:bool = True
            self.is_cpu_gpu:bool = True
            self.gpu_resource:float = (1/nb_gpu)
            # self.gpu_resource:float = (1/nb_gpu) - 0.01
            if cpu_gpu_ratio == None: raise ValueError("cpu_gpu_ratio must be set -> ex: --cpu_gpu_ratio \"[0.5, 0.5]\"")
            cpu_gpu_ratio_sum:int = np.sum(cpu_gpu_ratio)
            if cpu_gpu_ratio_sum != 1.0:
                cpu_norm = cpu_gpu_ratio[0] / cpu_gpu_ratio_sum
                gpu_norm = cpu_gpu_ratio[1] / cpu_gpu_ratio_sum
                cpu_gpu_ratio:List[float] = [cpu_norm, gpu_norm]
                print("Warning: sum of cpu_gpu_ratio != 1.0 (has to normalize it) -> cpu_norm:", cpu_norm, "gpu_norm:", gpu_norm)
            self.cpu_gpu_ratio:List[int, int] = cpu_gpu_ratio
            if nb_cpu + nb_gpu <= os.cpu_count():
                self.cpu:int = nb_cpu
                self.gpu:int = nb_gpu                
            else: 
                cpu_norm = nb_cpu / (nb_cpu + nb_gpu)
                gpu_norm = nb_gpu / (nb_cpu + nb_gpu)
                self.cpu:int = int(os.cpu_count() * cpu_norm)
                self.gpu:int = int(os.cpu_count() * gpu_norm)
                print("Warning: cpu + gpu > os.cpu_count() (has to normalize it) -> cpu:", self.cpu, "gpu:", self.gpu)

        elif "gpu" in self.device:
            self.device:str = "gpu"
            self.cpu:int = nb_gpu
            self.gpu:int = nb_gpu
            self.is_gpu:bool = True
            self.gpu_resource:float = (1/nb_gpu)
        
        elif "cpu" in self.device:
            self.device:str = "cpu"
            self.cpu:int = nb_cpu
            self.gpu:int = 0
            self.is_gpu:bool = False
            self.gpu_resource:float = 0.0
        
        else : raise ValueError("device must be 'cpu' or 'gpu'")

        print("Device:", self.device, "CPU:", self.cpu, "GPU:", self.gpu, "is_gpu:", self.is_gpu, "gpu_resource:", self.gpu_resource)

        self.is_parallele:bool = True if self.cpu > 1 or self.gpu > 1 else False
        if self.is_parallele == True:
            # 0 - Initialize Ray
            if ray.is_initialized() == True: ray.shutdown() # (if ray is already initialized, shutdown and reinit)
        
            self.cpu:int = self.cpu if self.cpu <= os.cpu_count() else os.cpu_count() # give the number of logical cpu cores

            runtime_env = {"working_dir": project_root, "env_vars": {"PYTHONPATH": f"{project_root}"},"excludes": [".git", ".venv", "build", "__pycache__", "*results*", "*gifs*"]}
            if self.device == "cpu_gpu": ray.init(num_cpus=self.cpu+self.gpu, num_gpus=1, include_dashboard=False, runtime_env=runtime_env) # number of cpu cores to use for the simulation
            elif self.device == "cpu":   ray.init(num_cpus=self.cpu, include_dashboard=False, runtime_env=runtime_env) # number of cpu cores to use for the simulation
            elif self.device == "gpu":   ray.init(num_cpus=self.gpu, num_gpus=1, include_dashboard=False, runtime_env=runtime_env) # number of cpu cores to use for the simulation
            else: raise ValueError("device must be 'cpu' or 'gpu'")
            # print("Ray cluster resources:", ray.cluster_resources())
            # print("Ray available resources:", ray.available_resources())
        else:
            self.cpu:int = 1
            print("Sequential mode CPU: 1")

    def __build_config_cache(self, config_path:str, config_cache_path:str) -> None:
        config_name = config_path.split("/")[-1]
        if not os.path.exists(config_cache_path):
            os.makedirs(config_cache_path)
        name = self.__init_file(config_cache_path, config_name)
        shutil.copyfile(config_path, config_cache_path + name)
        return config_cache_path + name
 
    def __init_file(self, config_cache_path, file_name:str) -> str:
        if os.path.exists(config_cache_path + file_name):
            i = 1
            while os.path.exists(config_cache_path + file_name + "_" + str(i)):
                i += 1
            return file_name + "_" + str(i) + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
        else:
            return file_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")


    def init_algorithm(self, name:str, algorithm_builder:Algorithm, config_path:str, extra_info:Dict[Any, Any]=None) -> None:
        self.config_path_algorithm:str = self.__build_config_cache(config_path, os.getcwd() + "/config_cache/")
        self.algorithm_builder:Algorithm = algorithm_builder
        self.algorithm_name_config:str = name
        self.algorithm_extra_info:Dict[Any, Any] = extra_info
        # self.algorithm:Algorithm = self.algorithm_builder(self.config_path_algorithm, self.algorithm_name_config, self.attributes_manager, self.algorithm_extra_info)
        self.algorithm_name:str = self.config["NEURO_EVOLUTION"]["algo_name"]
    
    def re_init_algorithm(self, run_nb:int) -> Algorithm:
        self.algorithm:Algorithm = self.algorithm_builder(self.config_path_algorithm, self.algorithm_name_config, self.attributes_manager, self.algorithm_extra_info)

        if self.problem_type == "RL":
            self.obs_max:np.ndarray = None
            self.obs_min:np.ndarray = None
            self.seeds_RL:np.ndarray = self.seeds_from_param[run_nb]
            if self.is_parallele == True:
                # Cause I already init/rest before hand
                if run_nb > 0: [self.problems_ray[i].reset.remote(self.environment_builder, self.config_path_problem, self.seeds_RL, self.nb_episode, self.nb_generations) for i in range(self.cpu)]
                # self.problem.reset(self.environment_builder, self.config_path_problem, self.seeds_RL, self.nb_episode)
            else:
                self.problem.reset(self.environment_builder, self.config_path_problem, self.seeds_RL, self.nb_episode, self.nb_generations)
        if self.record != None:
            self.record.best_genome = None

        return self.algorithm

    def init_problem_SL(self, problem_builder:Problem, config_path:str, name:str, features:np.ndarray, labels:np.ndarray, is_continuous_label:bool=False) -> None:
        self.config_path_problem:str = self.__build_config_cache(config_path, os.getcwd() + "/config_cache/")
        self.problem_type:str = "SL"
        self.is_RL_problem:bool = False
        self.environment_name:str = name
        self.seeds_RL:np.ndarray = None
        self.is_continuous_label:bool = is_continuous_label
        genome_input_size:int = int(self.config["Genome_NN"]["inputs"])
        genome_output_size:int = int(self.config["Genome_NN"]["outputs"])
        if is_continuous_label == False and genome_input_size  != len(features[0]):        raise Exception("Error: input size is not the same as the data set, genome input size = " + str(genome_input_size), "data set size = " + str(len(features[0])))
        if is_continuous_label == False and genome_output_size != len(np.unique(labels)):  raise Exception("Error: output size is not the same as the data set, genome output size = " + str(genome_output_size), "data set size = " + str(len(np.unique(labels))))
        if is_continuous_label == True  and genome_input_size  != features.shape[0]:       raise Exception("Error: input size is not the same as the data set, genome input size = " + str(genome_input_size), "data set size = " + str(features.shape[0]))
        if is_continuous_label == True  and genome_output_size != labels.shape[0]:         raise Exception("Error: output size is not the same as the data set, genome output size = " + str(genome_output_size), "data set size = " + str(labels.shape[0]))
        if self.is_parallele == True:
            features_id = ray.put(features)
            labels_id = ray.put(labels)
            if self.device == "cpu_gpu": 
                problem_builder_ray_cpu:Problem = ray.remote(problem_builder)
                problem_builder_ray_gpu:Problem = ray.remote(num_gpus=self.gpu_resource)(problem_builder)
                self.problems_ray = []
                for i in range(self.cpu + self.gpu):
                    if i < self.cpu: self.problems_ray.append(problem_builder_ray_cpu.remote(self.config_path_problem, features_id, labels_id, self.nb_generations, is_continuous_label=is_continuous_label, is_gpu=False))
                    else:            self.problems_ray.append(problem_builder_ray_gpu.remote(self.config_path_problem, features_id, labels_id, self.nb_generations, is_continuous_label=is_continuous_label, is_gpu=True))
                print("Problems Ray CPU_GPU:", len(self.problems_ray))

            elif self.device == "gpu":
                problem_builder_ray:Problem = ray.remote(num_gpus=self.gpu_resource)(problem_builder)
                self.problems_ray = [problem_builder_ray.remote(self.config_path_problem, features_id, labels_id, self.nb_generations, is_continuous_label=is_continuous_label, is_gpu=self.is_gpu) for i in range(self.cpu)]
                print("Problems Ray GPU:", len(self.problems_ray))

            elif self.device == "cpu":
                problem_builder_ray:Problem = ray.remote(problem_builder)
                self.problems_ray = [problem_builder_ray.remote(self.config_path_problem, features_id, labels_id, self.nb_generations, is_continuous_label=is_continuous_label, is_gpu=False) for _ in range(self.cpu)]
                print("Problem Ray CPU:", self.problems_ray)
        else:
            self.problem:Problem = problem_builder(self.config_path_problem, features, labels, self.nb_generations, is_continuous_label=is_continuous_label, is_gpu=self.is_gpu)
            print("Problem:", self.problem)

    def init_problem_RL(self, problem_builder:Problem, config_path:str, environment, nb_episode:int, seeds:np.ndarray, render:Union[str, bool]=False) -> None:
        self.config_path_problem:str = self.__build_config_cache(config_path, os.getcwd() + "/config_cache/")
        self.environment_builder = environment
        self.environment_name:str = environment.name
        self.seeds_from_param:np.ndarray = seeds
        self.seeds_RL:np.ndarray = None
        self.nb_episode:int = nb_episode
        self.problem_type:str = "RL"
        self.is_RL_problem:bool = True
        if self.network_type == "SNN":
            self.is_online:bool = True if self.config["Runner_Info"]["online"] == "True" else False
        self.obs_max:np.ndarray = None
        self.obs_min:np.ndarray = None

        self.is_render:bool = True if render != False else False
        self.render_path_file:str = render if render != False else None


        if self.is_parallele == True and self.is_render==False: # Parallele

            if self.device == "cpu_gpu": 
                problem_builder_ray_cpu:Problem = ray.remote(problem_builder)
                problem_builder_ray_gpu:Problem = ray.remote(num_gpus=self.gpu_resource)(problem_builder)
                self.problems_ray = []
                for i in range(self.cpu + self.gpu):
                    if i < self.cpu: self.problems_ray.append(problem_builder_ray_cpu.remote(self.environment_builder, self.config_path_problem, seeds[0], nb_episode, self.nb_generations, cpu_id=i, cpu_affinity=self.cpu_affinity, is_gpu=False))
                    else:            self.problems_ray.append(problem_builder_ray_gpu.remote(self.environment_builder, self.config_path_problem, seeds[0], nb_episode, self.nb_generations, cpu_id=i, cpu_affinity=self.cpu_affinity, is_gpu=True))
                print("Problems Ray CPU_GPU:", len(self.problems_ray))

            elif self.device == "gpu": 
                problem_builder_ray:Problem = ray.remote(num_gpus=self.gpu_resource)(problem_builder)
                self.problems_ray = [problem_builder_ray.remote(self.environment_builder, self.config_path_problem, seeds[0], nb_episode, self.nb_generations, cpu_id=i, cpu_affinity=self.cpu_affinity, is_gpu=self.is_gpu) for i in range(self.cpu)]
                print("Problems Ray GPU:", len(self.problems_ray))

            elif self.device == "cpu":                   
                problem_builder_ray:Problem = ray.remote(problem_builder)
                self.problems_ray = [problem_builder_ray.remote(self.environment_builder, self.config_path_problem, seeds[0], nb_episode, self.nb_generations, cpu_id=i, cpu_affinity=self.cpu_affinity, is_gpu=self.is_gpu) for i in range(self.cpu)]
                print("Problems Ray CPU:", len(self.problems_ray))

            # self.problem:Problem = problem_builder(self.environment_builder, self.config_path_problem, seeds, nb_episode, is_gpu=self.is_gpu)

        else: # Sequential
            self.problem:Problem = problem_builder(self.environment_builder, self.config_path_problem, seeds, nb_episode, self.nb_generations, is_gpu=self.is_gpu)
            print("Problem Sequential: 1")


    def run(self):
        if self.debug == True:
            if self.is_render == True:
                self.run_runder_RL(self.render_path_file)

            if self.is_parallele == True:
                if self.is_cpu_gpu == True: self.run_parallele_cpu_gpu()
                else: self.run_parallele()
            else:
                self.run_sequential()

            # rename record folder to done
            if self.is_record == True:
                self.record.end_run()
        else:
            try:
                if self.is_render == True:
                    self.run_runder_RL(self.render_path_file)

                if self.is_parallele == True:
                    # remove ray logs files
                    if os.path.exists("/tmp/ray"): 
                        shutil.rmtree("/tmp/ray")
                    self.run_parallele()
                else:
                    self.run_sequential()

                # rename record folder to done
                if self.is_record == True:
                    self.record.end_run()

            except Exception as e:
                # rename record folder to fail
                if self.is_record == True:
                    self.record.fail_run()
                print("Error:", e)

    def run_sequential(self):
        self.run_nb:int = 0

        for self.run_nb in range(self.nb_runs):
            self.algorithm = self.re_init_algorithm(run_nb=self.run_nb)
            population:Population = Population(get_new_population_id(), self.algorithm_name_config, self.config_path, self.attributes_manager)
            population.population = {}
            self.prev_obs_max:np.ndarray = None
            self.prev_obs_min:np.ndarray = None
            for self.generation in range(self.nb_generations):
                self.generation_time:float = time.time()
                    
                # Update Parameters
                population:Population = self.algorithm.run(population)

                # Run on Problem
                self.print_info_run(population)
                if self.is_RL_problem == True:
                    self.prev_obs_max:np.ndarray = self.obs_max
                    self.prev_obs_min:np.ndarray = self.obs_min
                    res, self.obs_max, self.obs_min = self.problem.run(population, self.generation, seeds=self.seeds_RL, obs_max=self.obs_max, obs_min=self.obs_min)
                    population.update_array(res, is_RL=True, obs_max=self.prev_obs_max, obs_min=self.prev_obs_min, seed=self.seeds_RL)
                    # self.__set_population_obs_max_min(population, self.prev_obs_max, self.prev_obs_min)
                else:
                    res:np.ndarray= self.problem.run(population, self.generation, seed=None)
                    population.update_array(res)

                # Record
                self.__update_records(population)

                # Update times info
                self.__update_info_run_and_print_stats(population)


    def run_parallele(self):
        self.run_nb:int = 0

        population_ids_cpu:List[np.ndarray] = None
        for self.run_nb in range(self.nb_runs):
            self.algorithm = self.re_init_algorithm(run_nb=self.run_nb)
            population:Population = Population(get_new_population_id(), self.algorithm_name_config, self.config_path, self.attributes_manager)
            self.prev_obs_max:np.ndarray = None
            self.prev_obs_min:np.ndarray = None
            for self.generation in range(self.nb_generations):
                self.generation_time:float = time.time()
                    
                # 1 - Update population parameters
                population = self.algorithm.run(population)
                
                # 2 - Check and Split population for each cpu
                pop_size:int = len(population.population)
                if pop_size < self.cpu: raise ValueError("pop_size ("+ str(pop_size)+") must be equal or greater than cpu number (" + str(self.cpu) + ")")

                # 3 - Regroup cpu populations in a list of populations
                if population_ids_cpu is None:
                    population_ids_cpu = TOOLS.split(population.population_genome_ids_list, self.cpu, to_numpy=True)

                # 4 - Run on Problem
                print("Device:", self.device, "CPU:", self.cpu, "GPU:", self.gpu, "gpu_resource:", self.gpu_resource)
                self.print_info_run(population)

                if self.generation != 0 and self.problem_type == "RL": # REINFROCEMENT LEARNING
                    self.prev_obs_max:np.ndarray = self.obs_max
                    self.prev_obs_min:np.ndarray = self.obs_min
                    population_ray_id = ray.put(population)
                    res_ids = [self.problems_ray[i].run.remote(population_ray_id, self.generation, seeds=self.seeds_RL, indexes=population_ids_cpu[i], obs_max=self.obs_max, obs_min=self.obs_min) for i in range(self.cpu)]

                else:                                                  # SUPERVISED LEARNING
                    population_ray_id = ray.put(population)
                    res_ids = [self.problems_ray[i].run.remote(population_ray_id, self.generation, indexes=population_ids_cpu[i]) for i in range(self.cpu)]
                    # ray.internal.free([population_ray_id])
                
                results = ray.get(res_ids)
                
                
                # 5 - Regroup population - Normal way
                if self.is_RL_problem == True:
                    fitnesses, observations_max, observations_min = map(list, zip(*results))
                    population.update_array(fitnesses, is_RL=True, obs_max=self.prev_obs_max, obs_min=self.prev_obs_min, seed=self.seeds_RL)
                    # self.__set_population_obs_max_min(population, self.prev_obs_max, self.prev_obs_min)
                    self.obs_max, self.obs_min = self.__update_obersevation_stats(np.array(observations_max), np.array(observations_min))
                else:
                    population.update_array(results)


                # 6 - Record
                self.__update_records(population)

                # 7 - Update times info + print stats
                self.__update_info_run_and_print_stats(population)
                                
        ray.shutdown()


    def run_parallele_cpu_gpu(self):
        self.run_nb:int = 0
        nb_cpu_gpu:int = self.cpu + self.gpu
        populations_cpu_gpu_idx:List[np.ndarray] = None
        for self.run_nb in range(self.nb_runs):
            self.algorithm = self.re_init_algorithm(run_nb=self.run_nb)
            population:Population = Population(get_new_population_id(), self.algorithm_name_config, self.config_path, self.attributes_manager)
            self.prev_obs_max:np.ndarray = None
            self.prev_obs_min:np.ndarray = None
            for self.generation in range(self.nb_generations):
                self.generation_time:float = time.time()
                    
                # 1 - Update population parameters
                population = self.algorithm.run(population)
                
                # 2 - Check and Split population for each cpu
                pop_size:int = len(population.population)
                if pop_size < nb_cpu_gpu: raise ValueError("pop_size ("+ str(pop_size)+") must be equal or greater than nb_cpu and nb_gpu number (" + str(nb_cpu_gpu) + ")")

                if populations_cpu_gpu_idx is None:
                    pop_key_list:List[int] = population.population_genome_ids_list
                    ratio_size:int = int(self.cpu_gpu_ratio[0]*pop_size)
                    populations_cpu_gpu_idx:List[List[int]] = TOOLS.split(pop_key_list[:ratio_size], self.cpu, to_numpy=True) # indexes for cpu
                    populations_cpu_gpu_idx.extend(TOOLS.split(pop_key_list[ratio_size:], self.gpu, to_numpy=True)) # add indexes for gpu

                # 4 - Run on Problem
                print("Device:", self.device, "CPU:", self.cpu, "GPU:", self.gpu, "cpu_gpu_ratio", self.cpu_gpu_ratio, "gpu_resource:", self.gpu_resource)
                self.print_info_run(population)

                if self.generation != 0 and self.is_RL_problem == True: # REINFROCEMENT LEARNING
                    self.prev_obs_max:np.ndarray = self.obs_max
                    self.prev_obs_min:np.ndarray = self.obs_min
                    population_ray_id = ray.put(population)
                    population_ids = [self.problems_ray[i].run.remote(population_ray_id, self.generation, seeds=self.seeds_RL, indexes=populations_cpu_gpu_idx[i], obs_max=self.obs_max, obs_min=self.obs_min) for i in range(nb_cpu_gpu)]

                else:                                                   # SUPERVISED LEARNING
                    population_ray_id = ray.put(population)
                    population_ids = [self.problems_ray[i].run.remote(population_ray_id, self.generation, indexes=populations_cpu_gpu_idx[i]) for i in range(nb_cpu_gpu)]
                
                results = ray.get(population_ids)
                
                
                # 5 - Regroup population - Normal way
                if self.is_RL_problem == True:
                    fitnesses, observations_max, observations_min = map(list, zip(*results))
                    population.update_array(fitnesses, is_RL=True, obs_max=self.prev_obs_max, obs_min=self.prev_obs_min, seed=self.seeds_RL)
                    # self.__set_population_obs_max_min(population, self.prev_obs_max, self.prev_obs_min)
                    self.obs_max, self.obs_min = self.__update_obersevation_stats(np.array(observations_max), np.array(observations_min))
                else:
                    population.update_array(results)


                # 6 - Record
                self.__update_records(population)


                # 7 - Update times info + print stats
                self.__update_info_run_and_print_stats(population)
                                
        ray.shutdown()


    def run_runder_RL(self, render_path_file:str) -> None:
        best_genome:Genome = Record.load_best(render_path_file)
        population = Population(get_new_population_id(), self.algorithm_name_config, self.config_path, self.attributes_manager)
        pop_dict:Dict[int, Genome] = {best_genome.id:best_genome}
        population.population = pop_dict

        population.sync_genomes_to_population(is_sync_population_to_genome=True)
        population.sync_population_to_run()
        self.problem.run_render(population)

    def __update_obersevation_stats(self, observation_max:np.ndarray, observation_min:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.max(observation_max, axis=0), np.min(observation_min, axis=0)

    def __update_records(self, population:Population) -> None:
        if self.is_record == True and self.generation > 0:
            if self.record == None:
                if self.problem_type == "RL":
                    if self.network_type == "SNN":
                        self.environment_name:str =  self.algorithm_name + "_" + self.encoder_record_text + "_" + self.decoder_record_text + "_" + self.problem_type + "_" + self.network_type + "_" + self.environment_name + "-pop-" + str(len(population.population)) + "-neuron-" + str(self.nb_hidden_neurons)
                    else:
                        self.environment_name:str =  self.algorithm_name + "_" +  self.problem_type + "_" + self.network_type + "_" + self.environment_name + "-pop-" + str(len(population.population)) + "-neuron-" + str(self.nb_hidden_neurons)
                else:
                    if self.network_type == "SNN":
                        self.environment_name:str =  self.algorithm_name + "_" + self.encoder_record_text + "_" + self.decoder_record_text + "_" + self.problem_type + "_" + self.network_type + "-pop-" + str(len(population.population)) + "-neuron-" + str(self.nb_hidden_neurons)
                    else:
                        self.environment_name:str =  self.algorithm_name + "_" + self.problem_type + "_" + self.network_type + "-pop-" + str(len(population.population)) + "-neuron-" + str(self.nb_hidden_neurons)
                self.record:Record = Record(self.config_path, self.environment_name)
            if self.record.record_from_algo == True: # Need find a better way to do this....
                self.record.save_info(self.algorithm.population_best, self.run_nb)
            else:
                self.record.save_info(population, self.run_nb)

    def __update_info_run_and_print_stats(self, population:Population) -> None:
        self.generation += 1
        self.generation_time = time.time() - self.generation_time
        if len(self.generations_times_list) <= 7: self.generations_times_list.append(self.generation_time)
        else: self.generations_times_list.pop(0); self.generations_times_list.append(self.generation_time)

        if self.verbose == True:
            if self.problem_type == "RL":
                self.__print_stats_RL(population)
            else:
                self.__print_stats_SL(population)

        if self.verbose_minimal == True:
            self.__print_stats_minimal(population)

        if self.network_type == "SNN" and self.encoder == "combinatorial":
            self.combinatorial_combinaison_noise = self.combinatorial_combinaison_noise * self.combinatorial_combinaison_noise_decay

    def print_info_run(self, population:Population) -> None:
        if self.verbose == False: return
        if self.network_type == "SNN":
            # GENERAL INFO
            print(
            "network_type:", self.network_type,
            "optimization_type:", self.optimization_type,
            "env_name:", self.environment_name,
            "record:", self.is_record,
            "seeds:", self.seeds_RL,
            "cpu:", self.cpu
            )
            combinatorial_print:str = "combinatorial_noise: " + str(round(self.combinatorial_combinaison_noise, 3)) + " combinatorial_combinaisons: " + str(self.get_combinatorial_combinaisons(population)) if self.is_combinatorial_encoding == True else ""
            voltage_print:str = "is_output_threshold: " + str(self.is_output_threshold) + " voltage_min: " + self.voltage_min + " voltage_max: " + self.voltage_max if self.is_voltage_encoding == True else ""
            if self.problem_type == "RL":
                print("encoder:", self.encoder, "decoder:", self.decoder, "record:", self.record_run, "decay:", self.decay_method, "is_output_threshold:", voltage_print, combinatorial_print)
                print("run_lenght:", self.run_length, "run_with_margin:", self.run_with_margin, "dt:", self.dt, "is_refractory:", self.is_refractory, "is_delay:", self.is_delay, "is_online:", self.is_online, "max_possible_spikes:", self.max_possible_spikes)
            else:
                print("run_lenght:", self.run_length, "run_with_margin:", self.run_with_margin, "dt:", self.dt, "is_refractory:", self.is_refractory, "is_delay:", self.is_delay, "encoder:", self.encoder, "decoder:", self.decoder, "record", self.record_run, "decay:", self.decay_method, voltage_print, combinatorial_print)

        if self.network_type == "ANN":
            is_bias:str = "bias: True" if self.is_bias == True else "bias: False"
            # is_layer_normalization:str = "layer_norm: True" if self.is_layer_normalization == True else "layer_norm: False"
            # GENERAL INFO
            print(
            "network_type:", self.network_type,
            is_bias,
            # is_layer_normalization,
            "optimization_type:", self.optimization_type,
            "env_name:", self.environment_name,
            "record:", self.is_record,
            "seeds:", self.seeds_RL,
            "cpu:", self.cpu
            )

    def get_combinatorial_combinaisons(self, population:Population) -> int:
        if "dynamic" in self.combinatorial_filter:
            if self.generation == 0: return self.combinatorial_combinaisons
            size:float = self.combinatorial_combinaisons
            mu:float = population.best_genome.combinatorial_modulo
            sigma:float = population.best_genome.combinatorial_modulo_sigma
            mod:float = size - ((1-mu%1)**(1 - np.exp(-10/size)) * size)
            ocillator:float = population.best_genome.combinatorial_modulo_ocillator_noise

            # combinaisons:List[int] = []
            # for i in range(self.combinatorial_combinaisons):
            #     combinaisons.append(i-i%mod)
            # print("combinaisons:", np.unique(np.rint(combinaisons)), "shape:", np.unique(np.rint(combinaisons)).shape)
            # return len(np.unique(np.rint(combinaisons)))

            indices = np.arange(size, dtype=float)
            shifted = indices - np.mod(indices, mod)  
            shifted = np.rint(shifted)
            nb_combinaisons = len(np.unique(shifted))
            print("combinatorial_init:", size, "mu:", round(mu, 5), "sigma:", round(sigma,5), "mod:", round(mod,4), "ocillator:", round(ocillator,4), "ocillator_speed", self.combinatorial_ocillator_noise, "ocillator_decay:", self.combinatorial_ocillator_noise_decay, "nb_combinaisons:", nb_combinaisons)
            self.combinatorial_ocillator_noise *= self.combinatorial_ocillator_noise_decay
            return nb_combinaisons
        else:
            return self.combinatorial_combinaisons

    def print_all_fitness(self, population:Population) -> None:
        fitnesses = []
        for genome in population.population.values():
            fitnesses.append(genome.fitness.score)
        fitnesses.sort(reverse=True)
        print("Fitnesses:", fitnesses)

    def __get_info_stats_by_alogrithm_SL(self, population:Population):
        population.update_info()
        best:Genome = population.best_genome
        stats:List[List[str, int, Tuple[int, float, int, int], float, int]] = []
        algorithm = self.algorithm # new
        algorithm.population_manager.update_info()
        name:str = algorithm.name
        pop_size:int = len(population.population)
        best:Genome = algorithm.population_manager.best_genome
        best_info:Tuple[int, float, int, int] = (best.id, round(best.fitness.score, 3), len(best.nn.hiddens["neurons_indexes_active"]), len(best.nn.synapses_actives_indexes[0]))
        avg_fitness:float = round(algorithm.population_manager.fitness.mean, 3)
        stats.append([name, pop_size, best_info, avg_fitness])
        return stats
    
    def __print_stats_SL(self, population:Population):
        if self.verbose == False: return
        minutes:float = (np.mean(self.generations_times_list)*(self.nb_generations - self.generation)) // 60
        seconds:float = (self.generation_time*(self.nb_generations - self.generation)) % 60
        time_left:str = str(int(minutes)) + "." + str(int(seconds))
        print("NEURO_EVO Run:",self.run_nb+1,"/", self.nb_runs, 
        "Generation:", self.generation, "/", self.nb_generations, 
        "pop_size:", len(population.population), 
        "time:", round(self.generation_time, 3), "s",
        "est_time: " + time_left +  " min",
        )
        t = time.localtime()
        currrent_time = time.strftime("%H:%M:%S", t)
        print("----------------------------------------------------------------------------------------------------------------------> local time: [" + str(currrent_time) + "]")
        titles = [["Algorithm", "Pop Size" ,"Best(id, fit, neur, syn)", "Avg fitness (pop)"]]
        titles.extend(self.__get_info_stats_by_alogrithm_SL(population))
        col_width = max(len(str(word)) for row in titles for word in row) + 2  # padding
        for row in titles:
            print("".join(str(word).ljust(col_width) for word in row))
        print("\n")

    def __get_info_stats_by_alogrithm_RL(self, population:Population):
        population.update_info()
        best:Genome = population.best_genome
        stats:List[List[str, int, Tuple[int, float, int, int], float, int]] = []
        algorithm = self.algorithm # new
        name:str = algorithm.name
        pop_size:int = len(population.population)
        best_episode:float = round(float(best.info["best_episode_raw_score"]), 3)
        mean_episodes:float = round(float(best.info["mean_episode_raw_score"]), 3)
        cum_episodes:float = round(float(best.fitness.score), 3)
        best_info:Tuple[int, float, float, float, int, int] = (best.id, best_episode, mean_episodes, cum_episodes, len(best.nn.hiddens["neurons_indexes_active"]), len(best.nn.synapses_actives_indexes[0]))
        avg_fitness:float = round(population.fitness.mean, 3)
        stats.append([name, pop_size, best_info, avg_fitness])
        return stats

    def __print_stats_RL(self, population:Population):
        if self.prev_obs_max is not None:
            print("observation_max:",np.round(self.prev_obs_max, 3), "action_local_max:", population.best_genome.action_local_max)
            print("observation_min:", np.round(self.prev_obs_min, 3), "action_local_min:", population.best_genome.action_local_min)
        if self.verbose == False: return
        minutes:float = (np.mean(self.generations_times_list)*(self.nb_generations - self.generation)) // 60
        seconds:float = (self.generation_time*(self.nb_generations - self.generation)) % 60
        time_left:str = str(int(minutes)) + "." + str(int(seconds))
        print("Run:",self.run_nb+1,"/", self.nb_runs, 
        "Generation:", self.generation, "/", self.nb_generations, 
        "pop_size:", len(population.population), 
        "time:", round(self.generation_time, 3), "s",
        "est_time: " + time_left +  " min"
        )
        t = time.localtime()
        currrent_time = time.strftime("%H:%M:%S", t)
        print("----------------------------------------------------------------------------------------------------------------------> local time: [" + str(currrent_time) + "]")
        titles = [["Algorithm", "Pop Size", "Best(id, fit, mean, cum, neur, syn)", "    Avg fitness (pop)"]]
        titles.extend(self.__get_info_stats_by_alogrithm_RL(population))
        # col_width = max(len(str(word)) for row in titles for word in row) + 2  # padding
        for j, row in enumerate(titles):
            # print("".join(str(word).ljust(col_width) for word in row))
            row_str:str = ""
            for i, word in enumerate(row):
                if i != 2:
                    row_str += str(word).ljust(15)
                if i == 2:
                    if j == 0:
                        row_str += str(word).ljust(50)
                    else:
                        row_str += "    " + str(word).ljust(50)
            print(row_str)
        print("\n")

    def __print_stats_minimal(self, population:Population) -> None:
            population.update_info()
            best:Genome = population.best_genome
            minutes:float = (np.mean(self.generations_times_list)*(self.nb_generations - self.generation)) // 60
            seconds:float = (self.generation_time*(self.nb_generations - self.generation)) % 60
            time_left:str = str(int(minutes)) + "." + str(int(seconds))
            print(self.environment_name + "-" + self.network_type, "Best:", round(best.fitness.score, 3), "Mean_score:", round(population.fitness.mean, 3),"Run:", self.run_nb+1,"/", self.nb_runs, "Generation:", self.generation, "/", self.nb_generations, "time:", round(self.generation_time, 3), "s", "est_time: " + time_left +  " min", "device:", self.device, "nb_devices:", self.cpu)
            return

    def check_snn_refractory_and_delay(self) -> None:
        if TOOLS.is_config_section(self.config_path, "refractory_neuron_parameter") == False:
            self.is_refractory:bool = False
        else:
            config_snn:Dict[str, Any] = TOOLS.config_function(self.config_path, ["refractory_neuron_parameter"])
            self.is_refractory:bool = np.any(np.array([
                float(config_snn["refractory_neuron_parameter"]["max"]),
                float(config_snn["refractory_neuron_parameter"]["min"])]) != 0.0)
        if TOOLS.is_config_section(self.config_path, "delay_synapse_parameter") == False:
            self.is_delay:bool = False
        else:
            config_snn:Dict[str, Any] = TOOLS.config_function(self.config_path, ["delay_synapse_parameter"])
            self.is_delay:bool = np.any(np.array([
                float(config_snn["delay_synapse_parameter"]["max"]),
                float(config_snn["delay_synapse_parameter"]["min"])]) != 0.0)
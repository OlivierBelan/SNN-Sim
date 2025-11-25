from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.Population import Population_NN as Population
from evo_simulator.GENERAL.NN import NN
import evo_simulator.TOOLS as TOOLS
from .snn_api_cython import SNN
from typing import List, Dict, Any, Set
# from . import spikegen
import numpy as np
import torch


class Runner_Info:
    def __init__(self, config_path:str, nb_episode:int=0) -> None:
        self.config_dict:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path, ["Runner_Info", "Genome_NN"])
        self.config_path:str = config_path
        self.network_type:str = self.config_dict["Genome_NN"]["network_type"]
        self.runner_type:str = self.config_dict["Runner_Info"]["runner_type"]

        if self.network_type == "SNN":
            self.run_time:int = int(self.config_dict["Runner_Info"]["run_time"])
            self.run_time_margin:int = int(self.config_dict["Runner_Info"]["run_time_margin"])
            self.dt:float = float(self.config_dict["Runner_Info"]["dt"])
            self.neuron_reset:str = self.config_dict["Runner_Info"]["neuron_reset"]
            self.is_voltage_negative:str = self.config_dict["Runner_Info"]["is_voltage_negative"]
            self.disable_output_threshold:bool = False
            self.decay_method:str = self.config_dict["Runner_Info"]["decay_method"]
            self.spike_count_negative:Set[str] = set(self.config_dict["Runner_Info"]["spike_count_negative"].replace(",", " ").replace("  ", " ").split(" "))
            self.encoder:str = self.config_dict["Runner_Info"]["encoder"]
            self.decoder:str = self.config_dict["Runner_Info"]["decoder"]
            self.record:set[str] = set(self.config_dict["Runner_Info"]["record"].replace(",", " ").replace("  ", " ").split(" "))
            self.record_layer:List[str] = self.config_dict["Runner_Info"]["record_layer"].replace(",", " ").replace("  ", " ").split(" ")
            self.input_spike_amplitude:float = float(self.config_dict["Runner_Info"]["input_spike_amplitude"])

            # Neural Network (LIF, LIF_energy, LIF_energy_sinusoidal)
            self.neuron_model:str = self.config_dict["Runner_Info"]["neuron_model"]
            if self.neuron_model not in ["LIF", "LIF_energy", "LIF_sinusoidal", "LIF_fourier", "LIF_energy_sinusoidal", "LIF_energy_fourier", "LIF_R_STDP"]: raise Exception("neuron_model", self.neuron_model, "not supported, please choose between 'LIF', 'LIF_energy', 'LIF_sinusoidal', 'LIF_fourier, 'LIF_energy_sinusoidal', 'LIF_energy_fourier'")


            if "none" in self.spike_count_negative: self.spike_count_negative = {"none"}
            if any([x not in {"none", "input", "hidden", "output"} for x in self.spike_count_negative]) == True: raise Exception("spike_count_negative", self.spike_count_negative, "not supported, please choose between 'none', 'input', 'hidden' and 'output'")
            if self.decay_method not in ["lif", "beta"]: raise Exception("decay_method", self.decay_method, "not supported, please choose between 'lif' and 'beta'")
            if self.encoder not in ["poisson", "binomial", "exact", "rate", "combinatorial", "latency", "direct", "burst", "derivative"]: raise Exception("encoding_method", self.encoder, "not supported, please choose between 'poisson', 'binomial', 'exact', 'rate', 'combinatorial', 'latency','direct' and 'derivative'")
            if self.decoder not in ["spike", "voltage", "augmented", "rate","max_spike", "coeff"]: raise Exception("decoding_method", self.decoder, "not supported, please choose between 'spike', 'voltage', 'augmented', 'rate' and 'max_spike'")
            for record_type in self.record:
                if record_type not in ["spike", "voltage", "augmented"]: raise Exception("record_decoding_method", record_type, "not supported, please choose between 'spike', 'voltage' or 'augmented'")            
            self.record_type:str = list(self.record)[0]
            if len(set(self.record_layer) - set(["input", "output", "hidden"])) != 0: raise Exception("record_layer", self.record_layer, "not supported, please choose between/or all 'input', 'output' and 'hidden'")

            if TOOLS.is_config_section(config_path, "delay_synapse_parameter") == False:
                self.is_delay = False
                self.delay_max:int = 0
            else:
                self.config_dict.update(TOOLS.config_function(config_path, ["delay_synapse_parameter"]))
                self.is_delay = True
                self.delay_max:int = float(self.config_dict["delay_synapse_parameter"]["max"])
                self.delay_max:int = np.ceil(self.delay_max).astype(int)

            self.is_refractory = TOOLS.is_config_section(config_path, "refractory_neuron_parameter")

            self.is_energy:bool = False
            self.is_energy_battery:bool = False
            if self.neuron_model == "LIF_energy" or self.neuron_model == "LIF_energy_sinusoidal" or self.neuron_model == "LIF_energy_fourier":
                self.init_energy_network(config_path)
                

            self.init_encoder(config_path)
            self.init_decoder(config_path)
            self.init_R_STDP(config_path)

        
        # used in Supervised
        if self.runner_type == "Supervised":
            self.batch_population:int = int(self.config_dict["Runner_Info"]["batch_population"])
            self.batch_features:int = int(self.config_dict["Runner_Info"]["batch_features"])
            self.batch_running:int = int(self.config_dict["Runner_Info"]["batch_running"])
            self.batch_running = self.batch_running if self.batch_running < self.batch_features else self.batch_features
        
        # used in Reinfocement
        if self.runner_type == "Reinforcement":
            if nb_episode == 0: raise Exception("nb_episode must be > 0")
            self.nb_episode:int = nb_episode
            if self.network_type == "SNN":
                self.online:bool = True if self.config_dict["Runner_Info"]["online"] == "True" else False

    def init_R_STDP(self, config_path:str) -> None:
        self.is_r_stdp:bool = True if "R_STDP" in self.neuron_model else False
        if self.is_r_stdp == False: return
        self.config_dict.update(TOOLS.config_function(config_path, ["R_STDP"]))

        self.learning_rate:float = float(self.config_dict["R_STDP"]["learning_rate"])
        self.tau_syn:float = float(self.config_dict["R_STDP"]["tau_syn"])
        self.weight_level:float = float(self.config_dict["R_STDP"]["weight_level"])
        self.t_window:int = int(self.config_dict["R_STDP"]["t_window"])

        self.stdp_A_plus:float = float(self.config_dict["R_STDP"]["a_plus"])
        self.stdp_A_minus:float = float(self.config_dict["R_STDP"]["a_minus"])
        self.stdp_tau_plus:float = float(self.config_dict["R_STDP"]["tau_plus"])
        self.stdp_tau_minus:float = float(self.config_dict["R_STDP"]["tau_minus"])
        self.stdp_w_max:float = float(self.config_dict["R_STDP"]["weight_max"])
        self.stdp_w_min:float = float(self.config_dict["R_STDP"]["weight_min"])

        self.stdp_p_noise:float = float(self.config_dict["R_STDP"]["p_noise"])
        self.stdp_w_noise:float = float(self.config_dict["R_STDP"]["w_noise"])
        self.stdp_sign_noise:float = float(self.config_dict["R_STDP"]["sign_noise"])

        self.R_plus_plus:float = float(self.config_dict["R_STDP"]["r_plus_plus"])
        self.R_plus_minus:float = float(self.config_dict["R_STDP"]["r_plus_minus"])
        self.R_minus_plus:float = float(self.config_dict["R_STDP"]["r_minus_plus"])
        self.R_minus_minus:float = float(self.config_dict["R_STDP"]["r_minus_minus"])

        if self.stdp_w_max < self.stdp_w_min: raise Exception("stdp_w_max must be > stdp_w_min")


    def init_energy_network(self, config_path:str) -> None:
        self.is_energy:bool = True
        self.config_dict.update(TOOLS.config_function(config_path, ["Energy_Network"]))
        self.energy_update:str = self.config_dict["Energy_Network"]["energy_update"]
        self.energy_norm:str = self.config_dict["Energy_Network"]["energy_norm"]
        self.is_energy_battery:bool = True if self.config_dict["Energy_Network"]["energy_battery"] == "True" else False
        self.energy_length:int = int(self.config_dict["Energy_Network"]["energy_length"])
        self.energy_interp_min:float = float(self.config_dict["Energy_Network"]["energy_interp_min"])
        self.energy_interp_max:float = float(self.config_dict["Energy_Network"]["energy_interp_max"])
        self.energy_is_interp:bool = True if self.config_dict["Energy_Network"]["energy_is_interp"] == "True" else False
        self.energy_keep_sign:bool = True if self.config_dict["Energy_Network"]["energy_keep_sign"] == "True" else False
        self.energy_decimal:int = int(self.config_dict["Energy_Network"]["energy_decimal"])        
        if self.energy_interp_min >= self.energy_interp_max: raise Exception("energy_interp_min must be < energy_interp_max")
        if self.energy_update not in ["constant", "ascending", "descending", "rate", "weight_acceleration"]: raise Exception("energy_update", self.energy_update, "not supported, please choose between 'constant', 'ascending', 'descending', 'rate' and 'weight_acceleration'")
        if self.energy_norm not in ["min_max_all", "min_max_row", "min_max_column", "L1_all", "L1_row", "L1_column", "L1_sum_all", "L1_sum_row", "L1_sum_column", "L2_all", "L2_row", "L2_column", "L2_sum_all", "L2_sum_row", "L2_sum_column", "none"]: raise Exception("energy_norm", self.energy_norm, "not supported, please choose between 'min_max_all', 'min_max_row', 'min_max_column', 'L1_all', 'L1_row', 'L1_column', 'L1_sum_all', 'L1_sum_row', 'L1_sum_column', 'L2_all', 'L2_row', 'L2_column', 'L2_sum_all', 'L2_sum_row', 'L2_sum_column'")
        if self.energy_length < 1: raise Exception("energy_length must be >= 1")


    def init_encoder(self, config_path:str) -> None:
        if self.encoder == "poisson":
            self.config_dict.update(TOOLS.config_function(config_path, ["Poisson_Encoder"]))
            self.spike_rate:int = int(self.config_dict["Poisson_Encoder"]["spike_rate"])
            self.max_nb_spikes:int = int(self.config_dict["Poisson_Encoder"]["max_nb_spikes"])

        elif self.encoder == "binomial":
            self.config_dict.update(TOOLS.config_function(config_path, ["Binomial_Encoder"]))
            self.reduce_noise:int = int(self.config_dict["Binomial_Encoder"]["reduce_noise"])
            self.max_nb_spikes:int = int(self.config_dict["Binomial_Encoder"]["max_nb_spikes"])
        
        elif self.encoder == "exact":
            self.config_dict.update(TOOLS.config_function(config_path, ["Exact_Encoder"]))
            self.max_nb_spikes:int = int(self.config_dict["Exact_Encoder"]["max_nb_spikes"])
                
        elif self.encoder == "combinatorial":
            self.config_dict.update(TOOLS.config_function(config_path, ["Combinatorial_Encoder"]))
            self.combinatorial_factor:int = int(self.config_dict["Combinatorial_Encoder"]["combinatorial_factor"])
            self.combinatorial_combinaison_size:float = float(self.config_dict["Combinatorial_Encoder"]["combinatorial_combinaison_size"])
            self.combinatorial_combinaison_size_max:int = int(float((self.config_dict["Combinatorial_Encoder"]["combinatorial_combinaison_size_max"])))
            self.combinatorial_combinaison_noise:float = float(self.config_dict["Combinatorial_Encoder"]["combinatorial_combinaison_noise"])
            self.combinatorial_combinaison_noise_decay:float = float(self.config_dict["Combinatorial_Encoder"]["combinatorial_combinaison_noise_decay"])
            self.combinatorial_filter:str = self.config_dict["Combinatorial_Encoder"]["combinatorial_filter"] # has to be 'energy', 'modulo_static', 'modulo_dynamic', 'modulo_energy_static' or 'modulo_energy_dynamic'
            if self.combinatorial_filter not in ["random", "binary", "energy", "modulo_static", "modulo_dynamic", "modulo_energy_static", "modulo_energy_dynamic", "modulo_static_dynamic", "modulo_energy_static_dynamic", "modulo_dynamic_static", "modulo_energy_dynamic_static", "number_ones", "modulo_number_ones_static", "modulo_number_ones_dynamic", "modulo_number_ones_static_dynamic"]: raise Exception("combinatorial_filter", self.combinatorial_filter, "not supported, please choose between 'binary', 'energy', 'modulo_static', 'modulo_dynamic', 'modulo_energy_static', 'modulo_energy_dynamic', 'modulo_static_dynamic', 'modulo_energy_static_dynamic', 'modulo_dynamic_static', 'modulo_energy_dynamic_static', 'number_ones', 'modulo_number_ones_static', 'modulo_number_ones_dynamic', 'modulo_number_ones_static_dynamic'")
            self.is_comibatorial_modulo_static = True if all(word in self.combinatorial_filter for word in ["modulo", "static"]) else False
            self.is_comibatorial_modulo_dynamic = True if all(word in self.combinatorial_filter for word in ["modulo", "dynamic"]) else False
            self.combinatorial_modulo_init:float = float(self.config_dict["Combinatorial_Encoder"]["combinatorial_modulo"]) # has to be >= 1.0
            self.combinatorial_modulo:np.ndarray = np.zeros(1, dtype=np.float32) # has to be >= 1.0
            self.combinatorial_print_table_debug:bool = True if self.config_dict["Combinatorial_Encoder"]["combinatorial_print_table_debug"] == "True" else False

            self.combinatorial_is_first_decay:bool = True
            self.combinatorial_roll:bool = True if self.config_dict["Combinatorial_Encoder"]["combinatorial_roll"] == "True" else False
            if self.run_time % self.combinatorial_factor != 0: raise Exception("run_time must be a multiple of combinatorial_factor -> run_time % combinatorial_factor == 0: run_time =",self.run_time, "combinatorial_factor =", self.combinatorial_factor)

        elif self.encoder == "rate":
            self.config_dict.update(TOOLS.config_function(config_path, ["Rate_Encoder"]))
        
        elif self.encoder == "latency":
            self.config_dict.update(TOOLS.config_function(config_path, ["Latency_Encoder"]))

        elif self.encoder == "direct":
            self.config_dict.update(TOOLS.config_function(config_path, ["Direct_Encoder"]))
            self.direct_max = float(self.config_dict["Direct_Encoder"]["direct_max"])
            self.direct_min = float(self.config_dict["Direct_Encoder"]["direct_min"])

        elif self.encoder == "derivative":
            self.config_dict.update(TOOLS.config_function(config_path, ["Derivative_Encoder"]))
            self.derivative_threshold = float(self.config_dict["Derivative_Encoder"]["derivative_threshold"])
            self.derivative_max_delta_latency = float(self.config_dict["Derivative_Encoder"]["derivative_max_delta_latency"])
            self.derivative_is_latency = True if "latency" in self.config_dict["Derivative_Encoder"]["derivative_type"] else False
            self.derivative_is_latency_positional:bool = True if "positional" in self.config_dict["Derivative_Encoder"]["derivative_type"] else False
            self.derivative_use_prev_input:bool = True if "True" in self.config_dict["Derivative_Encoder"]["derivative_use_prev_input"] else False

        else:
            raise Exception("Encoder", self.encoder, "not supported, please choose between 'poisson', 'binomial', 'exact', 'rate', 'combinatorial', 'latency' and 'direct'")

    def init_decoder(self, config_path:str) -> None:
        self.is_all_neurons_to_decode:bool = True # for augmented decoder
        self.spike_max:int = 0
        self.spike_distribution_run:int = 0
        self.spike_distribution_importance:int = 0
        self.importance_type:str = ""
        self.linear_spike_importance_type:str = ""
        self.spike_type:str = ""
        self.is_normalize:bool = False
        self.is_interpolate:bool = False
        self.interpolate_max:float = 0.0
        self.interpolate_min:float = 0.0
        self.is_voltage_reset:bool = False
        self.is_neurons_update_with_augmented:bool = False
        self.is_augmented:bool = False


        # DECODERS: rate, voltage, augmented, max_spike, coeff
        if self.decoder == "rate":
            self.config_dict.update(TOOLS.config_function(config_path, ["Rate_Decoder"]))
            self.ratio_max_output_spike = np.clip(float(self.config_dict["Rate_Decoder"]["ratio_max_output_spike"]), 0.0, 1.0)
        
        elif self.decoder == "voltage":
            self.config_dict.update(TOOLS.config_function(config_path, ["Voltage_Decoder"]))
            self.disable_output_threshold:bool = True if self.config_dict["Voltage_Decoder"]["disable_output_threshold"] == "True" else False
            nb_outputs:int = int(self.config_dict["Genome_NN"]["outputs"])
            self.voltage_max:str = self.config_dict["Voltage_Decoder"]["voltage_max"]
            self.voltage_min:str = self.config_dict["Voltage_Decoder"]["voltage_min"]
            self.is_voltages_min_decoder:bool = True if self.voltage_min == "voltage" else False
            self.is_threshold_max_decoder:bool = True if self.voltage_max == "threshold" else False
            if self.is_threshold_max_decoder == False:
                try:
                    self.voltage_max:np.ndarray = np.full(nb_outputs, float(self.voltage_max))
                except:
                    raise Exception("voltage_max:", self.voltage_max, "not supported, please choose a float value or 'threshold'")
            if self.is_voltages_min_decoder == False:
                try:
                    self.voltage_min:np.ndarray = np.full(nb_outputs, float(self.voltage_min))
                except:
                    raise Exception("voltage_min:", self.voltage_min, "not supported, please choose a float value or 'voltage'")
     
        elif self.decoder == "augmented":
            self.is_augmented = True
            self.config_dict.update(TOOLS.config_function(config_path, ["Augmented_Decoder"]))
            self.spike_max:int = int(self.config_dict["Augmented_Decoder"]["spike_max"])
            self.spike_distribution_run:int = int(self.config_dict["Augmented_Decoder"]["spike_distribution_run"])
            self.spike_distribution_importance:int = int(self.config_dict["Augmented_Decoder"]["spike_distribution_importance"])
            self.is_normalize:bool = True if self.config_dict["Augmented_Decoder"]["is_normalize"] == "True" else False
            self.is_interpolate:bool = True if self.config_dict["Augmented_Decoder"]["is_interpolate"] == "True" else False
            self.interpolate_max:float = float(self.config_dict["Augmented_Decoder"]["interpolate_max"])
            self.interpolate_min:float = float(self.config_dict["Augmented_Decoder"]["interpolate_min"])
            # positive, absolute, raw (positive and negative)
            self.spike_type:str = self.config_dict["Augmented_Decoder"]["spike_type"]
            # first_index, by_index, all, nothing 
            self.importance_type:str = self.config_dict["Augmented_Decoder"]["importance_type"]
            # ascending, descending
            self.linear_spike_importance_type:str = self.config_dict["Augmented_Decoder"]["linear_spike_importance_type"]
            self.output_multiplier:float = float(self.config_dict["Genome_NN"]["outputs_multiplicator"])
            self.is_voltage_reset:bool = True if self.config_dict["Augmented_Decoder"]["is_voltage_reset"] == "True" else False
            self.is_neurons_update_with_augmented:bool = True if self.config_dict["Augmented_Decoder"]["is_neurons_update_with_augmented"] == "True" else False
            self.is_all_neurons_to_decode:bool = True if self.config_dict["Augmented_Decoder"]["is_all_neurons_to_decode"] == "True" else False

            if self.spike_type not in ["positive", "absolute", "raw"]: raise Exception("spike_type", self.spike_type, "not supported, please choose between 'positive', 'absolute' and 'raw'")
            if self.importance_type not in ["first_index", "by_index", "nothing"]: raise Exception("importance_type", self.importance_type, "not supported, please choose between 'first_index', 'by_index' and 'nothing'")
            if self.linear_spike_importance_type not in ["ascending", "descending"]: raise Exception("linear_spike_importance_type", self.linear_spike_importance_type, "not supported, please choose between 'ascending' and 'descending'")
        
        elif self.decoder == "max_spike":
            pass
        
        elif self.decoder == "coeff":
            pass
        else:
            raise Exception("Decoder", self.decoder, "not supported, please choose between 'rate', 'voltage', 'augmented', 'max_spike' and 'coeff'")

class Runner(Runner_Info):
    def __init__(self, config_path:str, nb_episode:int=None, is_gpu:bool = False) -> None:
        Runner_Info.__init__(self, config_path, nb_episode) # Get SNN Runner Info from config file

        # 1 - INIT runner type (Supervised Learning or Reinforcement Learning)
        if self.runner_type not in ["Supervised", "Reinforcement"]:
            raise Exception("Runner type", self.runner_type, "not supported, please choose between 'Supervised' and 'Reinforcement'")
        

        # 3 - INIT Variables
        self.snn_list_len:int = 0
        self.snns_cython_dict:Dict[int, SNN] = {}
        self.neurons_total_used:int = 0
        self.synapses_total_used:int = 0
        self.cython_init:bool = False
        self.is_gpu:bool = is_gpu
        self.is_encoder_init:bool = False

    def init_cython_runner(self) -> None:
        if self.cython_init == True: return

        # from snn_cython import SNN_cython
        from SNN_cython_cuda.SNN_cython.snn_cython import SNN_cython
        from SNN_cython_cuda.SNN_cython.snn_cython import SNN_cython_population

        if self.runner_type == "Supervised": # Supervised Learning Runner
            from SNN_cython_cuda.SNN_cython.runner_SL_cython import Runner_SL_cython
            self.__runner_cython = Runner_SL_cython()

        elif self.runner_type == "Reinforcement": # Reinforcement Learning Runner
            from SNN_cython_cuda.SNN_cython.runner_RL_cython import Runner_RL_cython
            self.__runner_cython = Runner_RL_cython()


        self.SNN_cython_class = SNN_cython
        self.SNN_cython_population_class = SNN_cython_population
        self.cython_init:bool = True

        if self.runner_type == "Supervised":
            self.__runner_cython.init(run_time=self.run_time, run_time_margin=self.run_time_margin, dt=self.dt, input_spike_amplitude=self.input_spike_amplitude, is_augmented=self.is_augmented, is_energy=self.is_energy, neuron_reset=self.neuron_reset, record_layer=self.record_layer, disable_output_threshold=self.disable_output_threshold, decay_method=self.decay_method, is_delay=self.is_delay, is_refractory=self.is_refractory, record_decoding_method=self.record)

        elif self.runner_type == "Reinforcement":
            self.__runner_cython.init(run_time=self.run_time, run_time_margin=self.run_time_margin, dt=self.dt, nb_episode=self.nb_episode, online=self.online, is_augmented=self.is_augmented, is_energy=self.is_energy, is_r_stdp=self.is_r_stdp, neuron_reset=self.neuron_reset, record_layer=self.record_layer, disable_output_threshold=self.disable_output_threshold, decay_method=self.decay_method, is_delay=self.is_delay, is_refractory=self.is_refractory, delay_max=self.delay_max, record_decoding_method=self.record, spike_count_negative=self.spike_count_negative, is_voltage_negative=self.is_voltage_negative)

        if self.is_energy == True:
            self.__runner_cython.energy.init_param(self.energy_update, self.energy_norm, self.energy_length, self.is_energy_battery, self.energy_is_interp, self.energy_interp_min, self.energy_interp_max, self.energy_keep_sign, self.energy_decimal)

        if self.is_augmented == True:
            self.__runner_cython.augmented.init_param(self.is_all_neurons_to_decode, self.spike_max, self.spike_distribution_run, self.spike_distribution_importance, self.importance_type, self.linear_spike_importance_type, self.spike_type, self.is_normalize, self.is_interpolate, self.interpolate_max, self.interpolate_min, self.is_voltage_reset, self.is_neurons_update_with_augmented)

        # if self.is_r_stdp == True:
        #     self.__runner_cython.r_stdp.init_R_STDP(self.learning_rate, self.weight_level, self.t_window, self.stdp_A_plus, self.stdp_A_minus, self.stdp_tau_plus, self.stdp_tau_minus, self.tau_syn, self.stdp_w_min, self.stdp_w_max, self.stdp_p_noise, self.stdp_w_noise, self.stdp_sign_noise, self.R_plus_plus, self.R_plus_minus, self.R_minus_plus, self.R_minus_minus)

    def init_networks(self, population:Population, population_genome_ids:np.ndarray, is_last_run:bool = True) -> Dict[int, NN]:
        self.init_cython_runner()
        self.snn_list_len = population_genome_ids.size
        
        # 2 - Create snns_cython population
        population_cython:Dict[int, NN] = self.create_snns_population(population) # create snns_cython population

        # 3 -  Init networks
        if self.runner_type == "Supervised":
            self.__runner_cython.init_run(population_cython, population_genome_ids, batch_features=self.batch_features, batch_running=self.batch_running, batch_population=self.batch_population, is_gpu=self.is_gpu, is_last_run=is_last_run)
        elif self.runner_type == "Reinforcement":
            self.__runner_cython.init_run(population_cython, population_genome_ids=population_genome_ids, is_gpu=self.is_gpu, is_last_run=is_last_run)

        self.init_encoder_cython(population) # Init encoder

        return self.snns_cython_dict


    def init_encoder_cython(self, population:Population) -> None:                

        if self.encoder == "combinatorial": # check every time combinatorial as it can change/evolve during the run
            if self.combinatorial_is_first_decay == False:
                self.__runner_cython.encoder.combinaison_noise = self.combinatorial_combinaison_noise * self.combinatorial_combinaison_noise_decay
            if self.is_comibatorial_modulo_static == True:
                self.__runner_cython.encoder.combinatorial_modulo = np.full(self.snn_list_len, self.combinatorial_modulo_init, dtype=np.float32)
            if self.is_comibatorial_modulo_dynamic == True:
                self.__runner_cython.encoder.combinatorial_modulo = population.combinatorial_modulo
            self.combinatorial_is_first_decay = False

        if self.is_encoder_init == True: return

        if self.encoder == "combinatorial":
            self.__runner_cython.encoder.combinatorial_encoder_init(self.combinatorial_factor, self.combinatorial_combinaison_size, self.combinatorial_combinaison_size_max, self.combinatorial_roll, self.combinatorial_filter, self.combinatorial_print_table_debug)

        elif self.encoder == "poisson":
            self.__runner_cython.encoder.poisson_encoder_init(self.spike_rate, self.max_nb_spikes)

        elif self.encoder == "binomial":
            self.__runner_cython.encoder.binomial_encoder_init(self.reduce_noise, self.max_nb_spikes)
        
        elif self.encoder == "exact":
            self.__runner_cython.encoder.exact_encoder_init(self.max_nb_spikes)
                
        elif self.encoder == "rate":
            self.__runner_cython.encoder.rate_encoder_init()            
        
        elif self.encoder == "latency":
            self.__runner_cython.encoder.latency_encoder_init()

        elif self.encoder == "direct":
            self.__runner_cython.encoder.direct_encoder_init(self.direct_min, self.direct_max)

        elif self.encoder == "derivative":
            self.__runner_cython.encoder.derivative_encoder_init(self.derivative_threshold, self.derivative_max_delta_latency, self.derivative_is_latency, self.derivative_is_latency_positional, self.derivative_use_prev_input)

        elif self.encoder == "raw":
            self.__runner_cython.encoder.raw_encoder_init()

        else:
            raise Exception("Encoder", self.encoder, "not supported, please choose between 'poisson', 'binomial', 'exact', 'rate', 'combinatorial', 'latency' and 'direct'")
        self.is_encoder_init = True


    def run(self, features:np.ndarray, is_raw_data:bool=False) -> Dict[str, Dict[int, np.ndarray]] | np.ndarray:
        records:Dict[str, Dict[int, np.ndarray]] = {}
        
        # 1 - Run networks Supervised
        if (self.runner_type == "Supervised"):
            self.__runner_cython.run(features)
        
        # 1 - Run networks Reinforcement
        elif (self.runner_type == "Reinforcement" and self.snn_list_len == len(features[0])): # Check if nb of snns == nb of features
            self.__runner_cython.run(features)

        else:
            print("runner type", self.runner_type)
            print("snn_list_len", self.snn_list_len, "features", len(features[0]))
            print("features", features, "shape", features[0].shape)
            raise ValueError("Runner type", self.runner_type, "not supported, please choose between 'Supervised' and 'Reinforcement'")

        if "spike"     in self.record and is_raw_data == False: records["spike"]     = self.__runner_cython.get_record_spikes()
        if "voltage"   in self.record and is_raw_data == False: records["voltage"]   = self.__runner_cython.get_record_voltages()
        if "augmented" in self.record and is_raw_data == False: records["augmented"] = self.__runner_cython.get_record_augmented_spikes()
        
        if "spike"     in self.record and is_raw_data == True: return self.__runner_cython.get_record_spikes_raw()
        if "voltage"   in self.record and is_raw_data == True: return self.__runner_cython.get_record_voltages_raw()
        if "augmented" in self.record and is_raw_data == True: return self.__runner_cython.get_record_augmented_raw()

        return records

    def free_GPU(self) -> None:
        self.__runner_cython.free_GPU()

    def create_snns_population(self, population:Population) -> Population:
        # Just in order to get useful info
        # self.neurons_total_used = neuron_actives_indexes_pop.shape[0]
        # self.synapses_total_used = synapses_actives_indexes_pop.shape[1]

        # 2 - Create cython SNNs
        # for genome in genomes_dict.values():
        #     snn_python:NN = genome.nn
            # print("1 - unactives_neurons_indexes", np.where(~snn_python.neurons_status)[0].astype(np.int32))
            # print("2 - unactives_neurons_indexes", snn_python.neuron_unactives_indexes)
            # print("is_equal", np.array_equal(np.where(~snn_python.neurons_status)[0].astype(np.int32), snn_python.neuron_unactives_indexes))
            # exit()

        population.sync_topology_status() # Sync neuron and synapse topology status with the population
        population_cython = self.SNN_cython_population_class() # Create SNN_cython object
        population_cython.init_network(
                                # 0 - NN General Parameters
                                population.parameters, 
                                
                                population.population_genome_ids,

                                # 1 - Neurons indexes
                                population.input_idx,
                                population.output_idx,
                                population.parameters["hidden_neurons_actives_indexes"],

                                # 2 - Neurons and Synapses Population indexes active
                                population.parameters["neurons_actives_indexes"],
                                population.parameters["synapses_actives_indexes"],
                                
                                # 3 - Other Parameters that can optionnaly be used
                                self.is_delay,
                                self.is_refractory,
                                self.is_energy,
                                self.is_energy_battery,
                                population.is_dynamic_topology,
                                self.disable_output_threshold
                                )

        if population.is_dynamic_topology == True:
            for i, genome in enumerate(population.population.values()):
                population_cython.init_network_unactive_indexes(i, genome.nn.neuron_unactives_indexes, np.asarray(genome.nn.synapses_unactives_weight_indexes))

        return population_cython


    def get_encoded_data(self) -> np.ndarray:
        return self.__runner_cython.get_encoded_data()

    def set_reward_runner(self, rewards:np.ndarray) -> None:
        self.__runner_cython.set_reward(rewards)

    def get_weight_runner(self) -> np.ndarray:
        if self.runner_type == "Supervised":
            raise Exception("get_weight_cython:", self.runner_type, "not supported yet, available only for Reinforcement Learning Mode")
        else:
            return self.__runner_cython.get_weight()

    def get_weights_concatenated_runner(self) -> np.ndarray:
        if self.runner_type == "Supervised":
            raise Exception("get_weight_cython:", self.runner_type, "not supported yet, available only for Reinforcement Learning Mode")
        else:
            return self.__runner_cython.get_weights_concatenated()

    def r_STDP_apply_reward_runner(self, reward:np.ndarray) -> None:
        if self.runner_type == "Supervised":
            raise Exception("R_STDP_apply_reward:", self.runner_type, "not supported yet, available only for Reinforcement Learning Mode")
        else:
            self.__runner_cython.R_STDP_Apply_Reward(reward)
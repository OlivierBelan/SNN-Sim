cimport cython
import numpy as np
cimport numpy as np
np.import_array()
cimport libc.math as math
from libc.stdlib cimport rand, RAND_MAX

cdef class R_STDP:
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef void init_param(self, float learning_rate=0.01, float weight_level=0.1, int t_window = 1, float A_plus=0.01, float A_minus=0.012, float tau_plus=20.0, tau_minus=20.0, float tau_syn=200.0, float w_min=-1.0, float w_max=1.0, float p_noise_stdp=0.02, float w_noise_stdp=0.2, float sign_noise_stdp=0.5, float R_plus_plus=0.0, float R_plus_minus=0.0, float R_minus_plus=0.0, float R_minus_minus=0.0):
        self.stdp_learning_rate = learning_rate
        self.tau_syn = tau_syn
        self.weight_level= weight_level
        self.t_window = t_window

        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max

        self.is_r_stdp = True
        self.decay_neuron_trace = math.expf(-1.0 / self.tau_plus) # 1.0 / tau_plus
        self.decay_synapse_trace = math.expf(-1.0 / self.tau_syn) # 1.0 / tau_syn
        self.p_noise_stdp = p_noise_stdp
        self.w_noise_stdp = w_noise_stdp # need to multiply by the threshold if added to the weight
        self.sign_noise_stdp = sign_noise_stdp # 0.5 for positive and negative noise

        self.R_plus_plus = R_plus_plus
        self.R_plus_minus = R_plus_minus
        self.R_minus_plus = R_minus_plus
        self.R_minus_minus = R_minus_minus

        # print("R-STDP parameters: learning_rate", learning_rate, "weight_level", weight_level, "t_window", t_window, "A_plus", A_plus, "A_minus", A_minus, "tau_plus", tau_plus, "tau_minus", tau_minus, "tau_syn", tau_syn, "w_min", w_min, "w_max", w_max, "p_noise_stdp", p_noise_stdp, "w_noise_stdp", w_noise_stdp, "decay_neuron_trace", self.decay_neuron_trace, "decay_synapse_trace", self.decay_synapse_trace, "R_plus_plus", R_plus_plus, "R_plus_minus", R_plus_minus, "R_minus_plus", R_minus_plus, "R_minus_minus", R_minus_minus)
        # exit()


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void init_run(self, int nb_episode, int nb_networks, int nb_neurons, int[:,:] synapses_actives_indexes_view):
        self.nb_episode = nb_episode
        self.nb_networks = nb_networks
        self.nb_neurons = nb_neurons
        self.synapses_actives_indexes_view = synapses_actives_indexes_view

        # Initialize the traces
        self.neuron_trace  = np.zeros((nb_episode, nb_networks, nb_neurons), dtype=np.float32) # pre and post neuron trace
        self.synapse_trace = np.zeros((nb_networks, nb_neurons, nb_neurons), dtype=np.float32) # synapse trace

        if self.stdp_time_step == None: self.stdp_time_step = 0

        # print("R-STDP init_run: nb_episode", nb_episode, "nb_networks", nb_networks, "nb_neurons", nb_neurons)
        # exit()
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void init_step(self, int first_time_step, bint is_online):

        if first_time_step == 0: # code here will be call only at the first time step on the run
            self.background_spikes = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons), dtype=np.float32) # background_spikes is used for R-STDP
            if ((is_online == True and self.stdp_time_step == 0) or is_online == False):
                self.synapse_trace = np.zeros((self.nb_networks, self.nb_neurons, self.nb_neurons), dtype=np.float32)
                self.neuron_trace = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons), dtype=np.float32) # neuron_trace is used for MSTDPTs
                # self.weights_concatenated = np.empty((0,) + np.shape(self.weight[0, self.synapses_actives_indexes_view[0], self.synapses_actives_indexes_view[1]]), dtype=np.float32)
            return

        # STDP offline: reset traces otherwise we keep the previous values (online manner)
        if is_online == False: 
            self.background_spikes = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons), dtype=np.float32) # background_spikes is used for R-STDP
            self.synapse_trace = np.zeros((self.nb_networks, self.nb_neurons, self.nb_neurons), dtype=np.float32)
            self.neuron_trace = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons), dtype=np.float32) # neuron_trace is used for MSTDPTs

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef float add_noise_spikes_and_voltages(self, int episode, int network, int neuron, float voltage, float threshold):
        if rand() < (self.p_noise_stdp * RAND_MAX): # add noise to the voltage for testing purpose in R-STDP
            self.background_spikes[episode, network, neuron] = 1.0
            voltage += (self.w_noise_stdp*threshold) if rand() < (self.sign_noise_stdp * RAND_MAX) else -(self.w_noise_stdp*threshold) # add noise to the voltage
        else:
            self.background_spikes[episode, network, neuron] = 0.0
        return voltage


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void update_neuron_trace(self,  int episode,
                                                int network_idx,
                                                int neuron_idx,
                                                bint is_spike
                                                ):

        # Update Neuron Trace
        self.neuron_trace[episode, network_idx, neuron_idx] *= self.decay_neuron_trace # math.expf(-dt / self.tau_plus)
        if is_spike == True: self.neuron_trace[episode, network_idx, neuron_idx] += 1


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void update_synpase_trace(self, int episode,  
                                                int network,
                                                int neuron_pre,
                                                int neuron_post,
                                                bint   is_spike_pre, # pre  (from current step spike)
                                                bint   is_spike_post # post (from current step spike)
                                                ):

        """
        Update the weight with R-STDP rule
        """
        # 1 - Computes STDP Delta connction
        cdef float synapse_trace_delta = 0.0
        cdef float trace_pre  = self.neuron_trace[episode, network, neuron_pre]
        cdef float trace_post = self.neuron_trace[episode, network, neuron_post]

        synapse_trace_delta =  ((is_spike_post  *   self.A_plus    * trace_pre) + # (pre -> post) LTP
                                (is_spike_pre   * (-self.A_minus)  * trace_post)) # (post -> pre) LTD
        
        self.synapse_trace[network, neuron_pre, neuron_post] *= self.decay_synapse_trace # math.expf(-dt/τ_syn)
        self.synapse_trace[network, neuron_pre, neuron_post] += synapse_trace_delta

        # cdef float clip = 6.0          # trace maximum
        # if   self.synapse_trace[network, neuron_pre, neuron_post] >  clip: self.synapse_trace[network, neuron_pre, neuron_post] =  clip
        # elif self.synapse_trace[network, neuron_pre, neuron_post] < -clip: self.synapse_trace[network, neuron_pre, neuron_post] = -clip

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef void R_STDP_Apply_Reward(self, float[:,:] rewards, float[:,:,:] weight_view):
        """
        Apply the reward to the weight
        """
        cdef int i, j, k, neuron_in, neuron_out
        cdef int connections = self.synapses_actives_indexes_view.shape[1]
        cdef float reward_sign, weight_delta, reward, syn_trace
        cdef np.ndarray[np.float32_t, ndim=3] prev_weight = np.copy(weight_view)
        print("BEFORE: weights\n", prev_weight[0, self.synapses_actives_indexes_view[0], self.synapses_actives_indexes_view[1]], "shape", np.shape(prev_weight[0, self.synapses_actives_indexes_view[0], self.synapses_actives_indexes_view[1]]))

        print("APPLY REWARD: reward", np.array(rewards), "\nsynpase_trace", np.array(self.synapse_trace)[0, self.synapses_actives_indexes_view[0], self.synapses_actives_indexes_view[1]], "mean_synpase_trace", np.mean(np.array(self.synapse_trace)[0, self.synapses_actives_indexes_view[0], self.synapses_actives_indexes_view[1]]))
    
        for i in range(self.nb_episode):
            for j in range(self.nb_networks):
                for k in range(connections):
                    neuron_in  = self.synapses_actives_indexes_view[0, k]
                    neuron_out = self.synapses_actives_indexes_view[1, k]

                    weight_delta = 0.0
                    reward = rewards[i,j]
                    reward_sign = 1.0 if reward > 0 else -1.0
                    syn_trace = self.synapse_trace[j, neuron_in, neuron_out]

                    # 2 - Update the weight with the learning_rate + reward + synapse trace
                    # weight_delta = self.stdp_learning_rate * reward_sign * syn_trace # R-STDP trace

                    # weight_delta = self.stdp_learning_rate * reward[i,j] * syn_trace # R-STDP trace
                    
                    if   reward > 0.0 and syn_trace > 0.0:
                        print("R_plus_plus", "reward", reward, "syn_trace", syn_trace)
                        weight_delta = self.stdp_learning_rate * self.R_plus_plus * reward * syn_trace

                    elif reward > 0.0 and syn_trace < 0.0:
                        print("R_plus_minus", "reward", reward, "syn_trace", syn_trace)
                        weight_delta = self.stdp_learning_rate * self.R_plus_minus * reward * syn_trace
                        
                    elif reward < 0.0 and syn_trace > 0.0:
                        print("R_minus_plus", "reward", reward, "syn_trace", syn_trace)
                        weight_delta = self.stdp_learning_rate * self.R_minus_plus * reward * syn_trace

                    elif reward < 0.0 and syn_trace < 0.0:
                        print("R_minus_minus", "reward", reward, "syn_trace", syn_trace)
                        weight_delta = self.stdp_learning_rate * self.R_minus_minus * reward * syn_trace

                    print("Δw", weight_delta, "trace", syn_trace, "neuron_in", neuron_in, "neuron_out", neuron_out, "reward", reward,  "reward_sign", reward_sign)

                    weight_view[j, neuron_in, neuron_out] += weight_delta # R-STDP trace

                    # 3 - Clip the weight between w_min and w_max
                    if   weight_view[j, neuron_in, neuron_out] < self.w_min: weight_view[j, neuron_in, neuron_out] = self.w_min
                    elif weight_view[j, neuron_in, neuron_out] > self.w_max: weight_view[j, neuron_in, neuron_out] = self.w_max

                    if neuron_out == 2: # keep the weight positive for output neuron 2
                        weight_view[j, neuron_in, neuron_out] = 10


                    # 4 - Reset the synapse pre/post trace
                    self.synapse_trace[j, neuron_in, neuron_out] = 0.0

                    if math.isnan(weight_view[j, neuron_in, neuron_out]):
                        print("weight[network_idx, neuron_in, neuron_out]", np.array(weight_view[j, neuron_in, neuron_out]), "reward", reward, "learning_rate", self.stdp_learning_rate)
                        exit()
        print("AFTER REWARD weight", np.array(weight_view)[0, self.synapses_actives_indexes_view[0], self.synapses_actives_indexes_view[1]])
        print("DIFFERENCE weight", np.array(weight_view)[0, self.synapses_actives_indexes_view[0], self.synapses_actives_indexes_view[1]] - prev_weight[0, self.synapses_actives_indexes_view[0], self.synapses_actives_indexes_view[1]], "shape", np.shape(np.array(weight_view)[0, self.synapses_actives_indexes_view[0], self.synapses_actives_indexes_view[1]] - prev_weight[0, self.synapses_actives_indexes_view[0], self.synapses_actives_indexes_view[1]]))
        # print("OUTPUT_WEIGHT", np.array(self.output_weight), "shape", np.shape(np.array(self.output_weight)))


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void set_hidden_weight_positive(self, np.ndarray[np.float32_t, ndim=3] weight, int nb_synapses, int output_indexes_start):
        cdef float [:,:,:] weight_view = weight
        cdef int i, j, c, neuron_in, neuron_out

        for i in range(self.nb_networks):
            for j in range(self.nb_neurons):
                for c in range(nb_synapses):
                    neuron_in = self.synapses_actives_indexes_view[0, c]
                    neuron_out = self.synapses_actives_indexes_view[1, c]
                    if neuron_out >= output_indexes_start and neuron_out <= output_indexes_start:
                        if weight_view[i, neuron_in, neuron_out] < 0.0: weight_view[i, neuron_in, neuron_out] = 0.0 # set to 0.0 in case of negative weight

        output_synapses_indexes = np.where(np.array(self.synapses_actives_indexes_view[1]) == output_indexes_start)
        output_synapses_indexes = np.array(self.synapses_actives_indexes_view)[:, output_synapses_indexes[0]]
        self.output_weight = weight[0, output_synapses_indexes[0], output_synapses_indexes[1]]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef float _SAFE_EXP(self, float x):
        if   x > 80.0:  x = 80.0      # exp(80) ≈ 5.54e34  (float32 sûr)
        elif x < -80.0: x = -80.0
        return math.expf(x)


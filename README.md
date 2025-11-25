# SNN-sim

This repository contains a simple but flexible implementation of **clock-based Spiking Neural Networks (SNNs)** using **Cython**, **CUDA**, and **Ray**. (A more detailed description of the simulator will be provided in a future.) -- Also a more complete version of this runner will be available in Evo-Sim (https://github.com/OlivierBelan/Evo-Sim), with more features and algorithms NeuroEvolution related.

The core neuron model is **Leaky Integrate-and-Fire (LIF)**, and most neuron and synapse parameters are configurable and optimisable, including:

* Synaptic weights
* Neuron voltages and thresholds
* Time constants (τ)
* Delays and refractory period
* Constant currents and “energy” variables
* Network topology (feedforward, skip connections, recurrent patterns, etc.)

The simulator is designed for:

* **Reinforcement Learning (RL)** with NeuroEvolution (e.g. NES)
* **Supervised Learning (SL)** tasks such as MNIST

---

## Features

* Clock-based SNN simulator with LIF neurons
* CPU and GPU backends (Cython / CUDA)
* Integration with **Natural Evolution Strategies** (NES) for optimising SNN parameters
* Support for multiple **encoders/decoders** (rate, latency, combinatorial, derivative, voltage-based, augmented, etc.)
* Flexible network architectures
* Separate RL and SL pipelines (e.g. HalfCheetah for RL, MNIST for SL)

---

## Installation

This repository uses [`uv`](https://github.com/astral-sh/uv) to manage dependencies.

Install `uv` (for example):

```bash
pip install uv
```

(or any other method you prefer to install `uv`).

Then, from the repository root, install the dependencies with:

```bash
bash setup.sh
```

If you want to use the **GPU backend**, you also need a working CUDA toolchain and compatible drivers on your system. (or at least just the nvcc compiler for local compilation).

---

## Configuration files

All simulation, SNN and optimisation hyperparameters are stored in two config files provided in the repository:

* `NES_CONFIG_RL.cfg` – configuration for **Reinforcement Learning** experiments
* `NES_CONFIG_SL.cfg` – configuration for **Supervised Learning** experiments

These config files centralise, among other things:

* **Simulation settings**

  * Total run time and time step (`dt`)
  * Run-time margins
  * Runner type (RL vs supervised), online vs offline modes

* **SNN architecture and topology**

  * Number of inputs, hidden layers and outputs
  * Layer sizes (e.g. `H1:32`, `H2:32`, …)
  * Connectivity patterns (`I->H1`, `H1->O`, `I->O`, recurrent links, etc.)

* **Encoding / decoding choices**

  * Encoder type (e.g. poisson, binomial, exact, rate, combinatorial, latency, direct, derivative, etc.)
  * Decoder type (e.g. rate, voltage, augmented, …)

* **Optimisable parameters for NeuroEvolution**

  * Which parameters are evolved (e.g. `params_to_update = weight` or also thresholds, voltages, delays, etc.)
  * Ranges and distributions for each parameter type (min/max, μ, σ, decays, …)

* **NES and optimisation settings**

  * Population size, initial sigma, temperature, mean decay
  * Optimisation objective (maximize, minimize, closest_to_zero)

To change the behaviour of the simulator (runtime, architecture, encoders/decoders, which SNN parameters are optimised, etc.), **edit `NES_CONFIG_RL.cfg` or `NES_CONFIG_SL.cfg`** rather than the Python code.

---

## Running the simulation

All examples are launched via `uv` from the `test` directory.

### Reinforcement Learning mode

From the `test` directory:

**CPU only:**

```bash
uv run test_RL.py --problem HalfCheetah --algo NES --nn SNN --nb_runs 1 --nb_generation 100 --nb_episodes 1 --record False --device cpu --nb_cpu 10 --config config/config_snn/NES_CONFIG_RL.cfg
```

**With GPU:**

```bash
uv run test_RL.py --problem HalfCheetah --algo NES --nn SNN --nb_runs 1 --nb_generation 100 --nb_episodes 1 --record False --device gpu --nb_gpu 1 --config config/config_snn/NES_CONFIG_RL.cfg
```

These commands will:

* Load the RL configuration from `NES_CONFIG_RL.cfg`
* Build the SNN defined in the config (architecture, neuron model, parameters)
* Encode observations according to the chosen encoder and observation type
* Decode SNN activity into actions via the selected decoder and action scaling
* Optimise the chosen SNN parameters with NES over several generations

---

### Supervised Learning mode

From the `test` directory:

**CPU only:**

```bash
uv run test_SL.py --problem MNIST --algo NES --nn SNN --nb_runs 1 --nb_generation 1000 --record False --device cpu --nb_cpu 10 --config config/config_snn/NES_CONFIG_SL.cfg
```

**With GPU:**

```bash
uv run test_SL.py --problem MNIST --algo NES --nn SNN --nb_runs 1 --nb_generation 1000 --record False --device gpu --nb_gpu 1 --config config/config_snn/NES_CONFIG_SL.cfg
```

These commands will:

* Load the SL configuration from `NES_CONFIG_SL.cfg`
* Use the SNN simulator in a supervised learning setting (e.g. MNIST)
* Encode inputs into spike trains, run the SNN over time, and decode outputs
* Use NES to optimise the SNN parameters according to the chosen objective

---

## Customisation

* To modify **simulation length**, time step or runner behaviour: edit the corresponding fields in `NES_CONFIG_RL.cfg` or `NES_CONFIG_SL.cfg`.
* To change the **network architecture** (number of layers, sizes, connectivity): update the relevant sections in the config files.
* To decide which SNN parameters are **optimised by NES** (weights only vs full SNN parameters): adjust `params_to_update` and the parameter sections in the config files.
* To experiment with different **encoding/decoding strategies**: change the `encoder` and `decoder` entries and their associated parameters.

The goal is that you can run new experiments by **only editing the config files**, without touching the simulator core.

## Citation

If you use this simulator in your research, please cite:

> Olivier Belan, **"SNN-sim: A configurable spiking neural network simulator"**, 2025.  
> GitHub repository: <https://github.com/OlivierBelan/SNN-Sim>

```bibtex
@misc{belan2025snnsim,
  author       = {Belan, Olivier},
  title        = {SNN-sim: A configurable spiking neural network simulator},
  year         = {2025},
  howpublished = {\url{<https://github.com/OlivierBelan/SNN-Sim>}},
  note         = {GitHub repository}
}

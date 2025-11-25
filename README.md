# SNN sim

This repository contains a simple implementation of clock based Spiking Neural Networks (SNNs) using cython and cuda. The code demonstrates the basic concepts of SNNs, the neuron model is leaky integrate-and-fire (LIF), all parameters are configurable (weighs, thresholds, time constants, etc). The code is optimized for performance using ray, cython and cuda, allowing for efficient simulation of "large-scale" or "fast" SNNs.

## Installation
This repository using uv env to manage dependencies. To install the required packages, run the following commands:

```bash
pip install uv
```
or any way you prefer to install uv.

Then, install the required dependencies using the setup file:

```bash
bash setup.sh
```


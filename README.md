A simple, efficient and cool Spiking Neural Network simulator made in python/cython (multi-threaded, work on cpu, gpu version coming very soon). It is designed to be easy to use and to be able to run on a wide range of problems, from supervised to reinforcement learning. Work well with NeuroEvolution algorithms (NEAT, CMA-ES, NES-evosax, etc...).

A more detailed README is coming soon, but for now, here is a simple example of how to use it:

Go on the test folder and run the following command:

python test.py NAME_OF_YOUR_ALGORITHM:

(for SUPERVISED Problems)
```bash
python test_SL.py NEAT 
python test_SL.py CMA-ES 
python test_SL.py NES-evosax
```

(for REINFORCEMENT Problems)
```bash
python test_RL.py NEAT 
python test_RL.py CMA-ES
python test_RL.py NES-evosax
```

More parameters are available, check the test_RL.py/test_SL.py file for more information.

To make a comparison a ANN runner is also available (made with pytorch), to run it, you need to uncomment "start_config_path = "./config/config_ann/SL/" in test_SL.py or test_RL.py and comment the other one "start_config_path = "./config/config_snn/SL/".
    
A more complete version with more algorithms and more examples is available at: https://github.com/OlivierBelan/Evo-Sim (mainly NeuroEvolution algorithms)

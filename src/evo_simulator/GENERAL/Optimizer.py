import numpy as np
from typing import Union
from evo_simulator.GENERAL.Population import Population
from evo_simulator.GENERAL.Genome import Genome
from typing import List, Dict, Union

# adopted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py


class Optimizer(object):
  def __init__(self, pi, epsilon:float=1e-08):
    self.pi = pi
    self.dim = pi.num_params
    self.epsilon = epsilon # learning rate
    self.t = 0
    self.stepsize:float = None

  def update(self, globalg):
    self.t += 1
    step = self._compute_step(globalg)
    theta = self.pi.mu
    ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
    self.pi.mu = theta + step
    return ratio

  def _compute_step(self, globalg):
    raise NotImplementedError


class BasicSGD(Optimizer):
  def __init__(self, pi, stepsize):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize

  def _compute_step(self, globalg):
    step = -self.stepsize * globalg
    return step

class SGD(Optimizer):
  def __init__(self, pi, stepsize, momentum=0.9):
    Optimizer.__init__(self, pi)
    self.v = np.zeros(self.dim, dtype=np.float32)
    self.stepsize, self.momentum = stepsize, momentum

  def _compute_step(self, globalg):
    self.v = self.momentum * self.v + (1. - self.momentum) * globalg
    step = -self.stepsize * self.v
    return step


class Adam(Optimizer):
  def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize
    self.beta1 = beta1
    self.beta2 = beta2
    self.m = np.zeros(self.dim, dtype=np.float32)
    self.v = np.zeros(self.dim, dtype=np.float32)

  def _compute_step(self, globalg):
    a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
    self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
    self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
    step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
    return step

# end of adopted code

def sort_population(population:Union[Population, Dict[int, Genome], List[Genome]], optimization_type:str, criteria:str) -> List[Genome]:
  if type(population) == Population:
    population_dict:Dict[int, Genome] = population.population
  else:
    population_dict:Dict[int, Genome] = population

  if criteria == "fitness":
      if optimization_type == "maximize":
          return sorted(population_dict.values(), key=lambda x: x.fitness.score, reverse=True)
      elif optimization_type == "minimize":
          return sorted(population_dict.values(), key=lambda x: x.fitness.score, reverse=False)
      elif optimization_type == "closest_to_zero":
          return sorted(population_dict.values(), key=lambda x: abs(x.fitness.score), reverse=False)

  else:
      if optimization_type == "maximize":
          return sorted(population_dict.values(), key=lambda x: x.info[criteria], reverse=True)
      elif optimization_type == "minimize":
          return sorted(population_dict.values(), key=lambda x: x.info[criteria], reverse=False)
      elif optimization_type == "closest_to_zero":
          return sorted(population_dict.values(), key=lambda x: abs(x.info[criteria]), reverse=False)
    

def minimize(a:float, b:float, return_value:bool=False) -> Union[bool, float]:
  if a < b:
    if return_value:
      return a
    else:
      return True

def maximize(a:float, b:float, return_value:bool=False) -> Union[bool, float]:
  if a > b:
    if return_value:
      return a
    else:
      return True

def closest_to_zero(a:float, b:float, return_value:bool=False) -> Union[bool, float]:
  if abs(a) < abs(b):
    if return_value:
      return a
    else:
      return True
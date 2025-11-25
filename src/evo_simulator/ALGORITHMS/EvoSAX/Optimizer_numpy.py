import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# TODO: Add gradient clipping - select leads to more compute
# "use_clip_by_global_norm": False,
# "clip_global_norm": 5,
# "use_clip_by_value": False,
# "clip_value": 5,


def exp_decay(
    param: np.ndarray, param_decay: float, param_limit: float
) -> float:
    """Exponentially decay parameter & clip by minimal value."""
    param = param * param_decay
    param = np.maximum(param, param_limit)
    return param


@dataclass
class OptState:
    lrate: float # Learning rate
    m: np.ndarray # Momentum
    v: Optional[np.ndarray] = None
    n: Optional[np.ndarray] = None 
    last_grads: Optional[np.ndarray] = None # Last gradients
    gen_counter: int = 0 # Generation counter


@dataclass
class OptParams:
    lrate_init: float = 0.01 # Initial learning rate
    lrate_decay: float = 0.999 # Learning rate decay
    lrate_limit: float = 0.001 # Learning rate limit
    momentum: Optional[float] = None
    beta_1: Optional[float] = None
    beta_2: Optional[float] = None
    beta_3: Optional[float] = None
    eps: Optional[float] = None
    max_speed: Optional[float] = None



class Optimizer(object):
    def __init__(self, num_dims: int):
        """Simple NumPy-Compatible Optimizer Class."""
        self.num_dims = num_dims

    @property
    def default_params(self) -> OptParams:
        """Return shared and optimizer-specific default parameters."""
        return OptParams(**self.params_opt)

    def initialize(self, params: OptParams) -> OptState:
        """Initialize the optimizer state."""
        return self.initialize_opt(params)

    def step(
        self,
        mean: np.ndarray,
        grads: np.ndarray,
        state: OptState,
        params: OptParams,
    ) -> Tuple[np.ndarray, OptState]:
        """Perform a gradient-based update step."""
        return self.step_opt(mean, grads, state, params)

    def update(self, state: OptState, params: OptParams) -> OptState:
        """Exponentially decay the learning rate if desired."""
        lrate = exp_decay(state.lrate, params.lrate_decay, params.lrate_limit)
        state.lrate = lrate
        return state

    @property
    def params_opt(self) -> Dict[str, float]:
        """Optimizer-specific hyperparameters."""
        raise NotImplementedError

    def initialize_opt(self, params: OptParams) -> OptState:
        """Optimizer-specific initialization of optimizer state."""
        raise NotImplementedError

    def step_opt(
        self,
        mean: np.ndarray,
        grads: np.ndarray,
        state: OptState,
        params: OptParams,
    ) -> Tuple[np.ndarray, OptState]:
        """Optimizer-specific step to update parameter estimates."""
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, num_dims: int):
        """Simple NumPy-Compatible SGD + Momentum optimizer."""
        super().__init__(num_dims)
        self.opt_name = "sgd"

    @property
    def params_opt(self) -> Dict[str, float]:
        """Return default SGD+Momentum parameters."""
        return {
            "momentum": 0.0,
        }

    def initialize_opt(self, params: OptParams) -> OptState:
        """Initialize the momentum trace of the optimizer."""
        return OptState(m=np.zeros(self.num_dims), lrate=params.lrate_init)

    def step_opt(
        self,
        mean: np.ndarray,
        grads: np.ndarray,
        state: OptState,
        params: OptParams,
    ) -> Tuple[np.ndarray, OptState]:
        """Perform a simple SGD + Momentum step."""
        # Update the momentum trace
        m = grads + params.momentum * state.m
        mean_new = mean - state.lrate * m
        state.m = m
        state.gen_counter += 1
        return mean_new, state


class Adam(Optimizer):
    def __init__(self, num_dims: int):
        """NumPy-Compatible Adam Optimizer (Kingma & Ba, 2015)
        Reference: https://arxiv.org/abs/1412.6980"""
        super().__init__(num_dims)
        self.opt_name = "adam"

    @property
    def params_opt(self) -> Dict[str, float]:
        """Return default Adam parameters."""
        return {
            "beta_1": 0.99,
            "beta_2": 0.999,
            "eps": 1e-8,
        }

    def initialize_opt(self, params: OptParams) -> OptState:
        """Initialize the m, v trace of the optimizer."""
        return OptState(
            m=np.zeros(self.num_dims),
            v=np.zeros(self.num_dims),
            lrate=params.lrate_init,
        )

    def step_opt(
        self,
        mean: np.ndarray,
        grads: np.ndarray,
        state: OptState,
        params: OptParams,
    ) -> Tuple[np.ndarray, OptState]:
        """Perform a simple Adam GD step."""

        m = (1 - params.beta_1) * grads + params.beta_1 * state.m
        v = (1 - params.beta_2) * (grads ** 2) + params.beta_2 * state.v
        mhat = m / (1 - params.beta_1 ** (state.gen_counter + 1))
        vhat = v / (1 - params.beta_2 ** (state.gen_counter + 1))

        mean_new = mean - state.lrate * mhat / (np.sqrt(vhat) + params.eps)
        state.m = m
        state.v = v
        state.gen_counter += 1
        return mean_new, state


class RMSProp(Optimizer):
    def __init__(self, num_dims: int):
        """NumPy-Compatible RMSProp Optimizer (Hinton et al., 2012)
        Reference: https://tinyurl.com/2sbbcnrv"""
        super().__init__(num_dims)
        self.opt_name = "rmsprop"

    @property
    def params_opt(self) -> Dict[str, float]:
        """Return default RMSProp parameters."""
        return {
            "momentum": 0.9,
            "beta_1": 0.99,
            "eps": 1e-8,
        }

    def initialize_opt(self, params: OptParams) -> OptState:
        """Initialize the m, v trace of the optimizer."""
        return OptState(
            m=np.zeros(self.num_dims),
            v=np.zeros(self.num_dims),
            lrate=params.lrate_init,
        )

    def step_opt(
        self,
        mean: np.ndarray,
        grads: np.ndarray,
        state: OptState,
        params: OptParams,
    ) -> Tuple[np.ndarray, OptState]:
        """Perform a simple RMSprop GD step."""
        v = (1 - params.beta_1) * (grads ** 2) + params.beta_1 * state.v
        m = params.momentum * state.m + grads / (np.sqrt(v) + params.eps)
        mean_new = mean - state.lrate * m
        state.m = m
        state.v = v
        state.gen_counter += 1
        return mean_new, state


class ClipUp(Optimizer):
    def __init__(self, num_dims: int):
        """NumPy-Compatible ClipUp Optimizer (Toklu et al., 2020)
        Reference: https://arxiv.org/abs/2008.02387"""
        super().__init__(num_dims)
        self.opt_name = "clipup"

    @property
    def params_opt(self) -> Dict[str, float]:
        """Return default ClipUp parameters."""
        return {
            "lrate_init": 0.15,
            "lrate_decay": 0.999,
            "lrate_limit": 0.05,
            "max_speed": 0.3,
            "momentum": 0.9,
        }

    def initialize_opt(self, params: OptParams) -> OptState:
        """Initialize the momentum trace of the optimizer."""
        return OptState(m=np.zeros(self.num_dims), lrate=params.lrate_init)

    def step_opt(
        self,
        mean: np.ndarray,
        grads: np.ndarray,
        state: OptState,
        params: OptParams,
    ) -> Tuple[np.ndarray, OptState]:
        """Perform a ClipUp step."""
        # Normalize length of gradients
        grad_magnitude = np.linalg.norm(grads)
        gradient = grads / (grad_magnitude + 1e-8)
        step = gradient * state.lrate
        velocity = params.momentum * state.m + step

        # Clip the update velocity
        vel_magnitude = np.linalg.norm(velocity)
        if vel_magnitude > params.max_speed:
            velocity = velocity * (params.max_speed / (vel_magnitude + 1e-8))

        mean_new = mean - state.lrate * velocity
        state.m = velocity
        state.gen_counter += 1
        return mean_new, state


class Adan(Optimizer):
    def __init__(self, num_dims: int):
        """NumPy-Compatible Adan Optimizer (Xi et al., 2022)
        Reference: https://arxiv.org/pdf/2208.06677.pdf"""
        super().__init__(num_dims)
        self.opt_name = "adan"

    @property
    def params_opt(self) -> Dict[str, float]:
        """Return default Adan parameters."""
        return {
            "beta_1": 0.98,
            "beta_2": 0.92,
            "beta_3": 0.99,
            "eps": 1e-8,
        }

    def initialize_opt(self, params: OptParams) -> OptState:
        """Initialize the m, v, n trace of the optimizer."""
        return OptState(
            m=np.zeros(self.num_dims),
            v=np.zeros(self.num_dims),
            n=np.zeros(self.num_dims),
            last_grads=np.zeros(self.num_dims),
            lrate=params.lrate_init,
        )

    def step_opt(
        self,
        mean: np.ndarray,
        grads: np.ndarray,
        state: OptState,
        params: OptParams,
    ) -> Tuple[np.ndarray, OptState]:
        """Perform a simple Adan GD step."""
        m = (1 - params.beta_1) * grads + params.beta_1 * state.m
        grad_diff = grads - state.last_grads
        v = (1 - params.beta_2) * grad_diff + params.beta_2 * state.v
        n = (1 - params.beta_3) * (
            grads + params.beta_2 * grad_diff
        ) ** 2 + params.beta_3 * state.n

        mhat = m / (1 - params.beta_1 ** (state.gen_counter + 1))
        vhat = v / (1 - params.beta_2 ** (state.gen_counter + 1))
        nhat = n / (1 - params.beta_3 ** (state.gen_counter + 1))
        mean_new = mean - state.lrate * (mhat + params.beta_2 * vhat) / (
            np.sqrt(nhat) + params.eps
        )
        state.m = m
        state.v = v
        state.n = n
        state.last_grads = grads
        state.gen_counter += 1
        return mean_new, state


GradientOptimizer = {
    "sgd": SGD,
    "adam": Adam,
    "rmsprop": RMSProp,
    "clipup": ClipUp,
    "adan": Adan,
}

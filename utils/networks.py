from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class FourierFeatures(nn.Module):
    """Sinusoidal positional encoding for the flow timestep t in [0, 1].

    Maps a scalar timestep to a high-dimensional embedding via:
        [cos(f_1*t), ..., cos(f_d*t), sin(f_1*t), ..., sin(f_d*t)]
    where f_i are either learned or fixed log-spaced frequencies (same scheme as
    the Transformer positional encoding / diffusion timestep embedding).

    This gives the vector field network a richer representation of "where in the
    flow trajectory" it currently is, compared to feeding raw t directly.
    """
    output_size: int = 64
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            # Learned random Fourier features: project t through a learned matrix
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            # Fixed log-spaced frequencies (same as Transformer / DDPM timestep embedding)
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)   # [10000^0, ..., 10000^{-1}]
            f = x * f                                  # broadcast: (batch, 1) * (half_dim,)
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)



class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class ActorVectorFieldMFM(nn.Module):
    """Average velocity network u_{r,t}(a_r | s) for flow map policies (Eq. 2).

    Predicts the average velocity over a flow interval [r, t]:
        input = concat[obs, a_r, fourier(r), fourier(t)]

    When r == t, reduces to instantaneous velocity (standard CFM, Eq. 3).
    When r < t, predicts the jump velocity: a_t = a_r + (t-r)·u_{r,t}.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        if self.use_fourier_features:
            self.ff_start = FourierFeatures(self.fourier_feature_dim)
            self.ff_end = FourierFeatures(self.fourier_feature_dim)

    @nn.compact
    def __call__(self, observations, actions, times=None, end_times=None,
                 is_encoded=False, **kwargs):
        """Predict u_{r,t}(a_r | s).

        Args:
            observations: State s. Shape: (batch, obs_dim).
            actions: Noisy action a_r. Shape: (batch, action_dim).
            times: Start time r ∈ [0, 1]. Shape: (batch, 1).
            end_times: End time t ∈ [0, 1]. Shape: (batch, 1).
            is_encoded: If True, skip encoder.

        Returns:
            Predicted average velocity. Shape: (batch, action_dim).
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        elif end_times is not None:
            if self.use_fourier_features:
                times = self.ff_start(times)
                end_times = self.ff_end(end_times)
            inputs = jnp.concatenate([observations, actions, times, end_times], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff_start(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        return self.mlp(inputs)


class ActorVectorField(nn.Module):
    """Neural network that predicts the velocity vector field v(x_t, t | s) for
    Conditional Flow Matching (CFM).

    Architecture summary
    --------------------
    The network learns a time-conditioned vector field that transports samples
    from a standard Gaussian x_0 ~ N(0, I) to the data distribution x_1
    (dataset actions) along straight-line interpolation paths:

        x_t = (1 - t) * x_0 + t * x_1,    t in [0, 1]

    Given the current state s (observations), the noisy action x_t, and the flow
    time t, the network predicts the velocity  v = dx/dt ≈ x_1 - x_0.

    Input construction:
        1. observations  ->  (optionally) encoded by a visual encoder
        2. times t       ->  (optionally) lifted to Fourier features for richer
                              frequency representation
        3. [observations, x_t, fourier(t)]  are concatenated into a single vector

    The concatenated vector is fed through a 4-layer MLP (default 512 units each)
    with GELU activations, producing a vector of size `action_dim` (or
    `action_dim * horizon_length` when using action chunking).

    During training the CFM loss is:
        L = E_{x_0, x_1, t} || v_theta(x_t, t, s) - (x_1 - x_0) ||^2

    At inference, actions are generated by numerically integrating the learned
    field from t=0 to t=1 via the Euler method (see `compute_flow_actions` in
    qc.py).

    Attributes:
        hidden_dims: MLP hidden layer sizes, e.g. (512, 512, 512, 512).
        action_dim: Output dimensionality (action_dim * horizon_length if chunking).
        layer_norm: Whether to apply LayerNorm after each hidden layer.
        encoder: Optional visual encoder applied to raw observations.
        use_fourier_features: If True, encode scalar t via FourierFeatures before concat.
        fourier_feature_dim: Dimensionality of the Fourier time embedding.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64

    def setup(self) -> None:
        # Final layer outputs action_dim with no activation (raw velocity prediction)
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Predict the velocity vector v(x_t, t | s).

        Args:
            observations: Environment observations s.  Shape: (batch, obs_dim).
            actions: Noisy actions x_t along the flow path.  Shape: (batch, action_dim).
            times: Flow time t in [0, 1].  Shape: (batch, 1). If None, time-unconditional.
            is_encoded: If True, skip the encoder (observations are pre-encoded).

        Returns:
            Predicted velocity v of shape (batch, action_dim).
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        if times is None:
            # Time-unconditional mode (used by the one-step distilled actor)
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)  # (batch, 1) -> (batch, fourier_feature_dim)
            # Concatenate: [s, x_t, embed(t)]
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v

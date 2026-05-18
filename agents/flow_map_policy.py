"""Flow Map Policy — offline pre-training (§3.2).

Trains a one-step flow map actor X_{0,1} using:
  1. L_Diag (Eq. 3): diagonal conditional flow matching (r = t).
  2. L_ESD  (Eq. 5): Eulerian self-distillation (off-diagonal, r < t)
     enforcing the flow map self-consistency via JVP.

The resulting pretrained policy serves as the offline anchor u^off_{r,1}
for FMQ online fine-tuning (see fmq.py).
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorFieldMFM, Value
from agents.flow_utils import sample_s_u


class FlowMapPolicy(flax.struct.PyTreeNode):
    """Flow map policy for offline pre-training (§3.2).

    Training: L_Diag (Eq. 3) + L_ESD (Eq. 5) with [r,t] curriculum.
    Inference: one-step flow map X_{0,1}(a₀|s) + best-of-N Q-selection.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

        next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)

        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """L_Diag + L_ESD (Eqs. 3, 5).

        1. L_Diag (r = t): ‖u_{t,t}(a_t|s) - (a₁ - a₀)‖²
        2. L_ESD  (r < t): Eulerian self-distillation via JVP with [r,t] curriculum.
        """
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_1 = batch_actions

        # ---- 1. L_Diag: CFM loss (diagonal: r = t) ----
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(
            batch['observations'], x_t, t, end_times=t, params=grad_params,
        )

        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2,
                    (batch_size, self.config["horizon_length"], self.config["action_dim"])
                ) * batch["valid"][..., None]
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        # ---- 2. L_ESD: Eulerian self-distillation (off-diagonal: r < t) ----
        rng, x_rng_esd, su_rng = jax.random.split(rng, 3)
        x_0_esd = jax.random.normal(x_rng_esd, (batch_size, action_dim))
        v_ss = x_1 - x_0_esd

        step = self.network.step
        s, u = sample_s_u(
            su_rng, batch_size, step,
            self.config['flow_map_warmup_steps'],
            self.config['flow_map_anneal_end_step'],
        )
        I_s = (1 - s) * x_0_esd + s * x_1

        def vsu_fn(s_in, u_in, x_in):
            return self.network.select('actor_bc_flow')(
                batch['observations'], x_in, s_in, end_times=u_in, params=grad_params,
            )

        distillation_type = self.config.get('distillation_type', 'mf')

        if distillation_type == 'lsd':
            def Xsu_fn(s_in, u_in, x_in):
                v = self.network.select('actor_bc_flow')(
                    batch['observations'], x_in, s_in, end_times=u_in, params=grad_params,
                )
                return x_in + (u_in - s_in) * v

            primals = (s, u, I_s)
            tangents = (jnp.zeros_like(s), jnp.ones_like(u), jnp.zeros_like(I_s))
            X_su, dXdu = jax.jvp(Xsu_fn, primals, tangents)

            v_uu = self.network.select('actor_bc_flow')(
                batch['observations'], jax.lax.stop_gradient(X_su), u, end_times=u, params=grad_params,
            )
            student = v_uu
            teacher = jax.lax.stop_gradient(dXdu)

        elif distillation_type == 'psd':
            rng, gamma_rng = jax.random.split(rng)
            gamma = jax.random.uniform(gamma_rng, (batch_size, 1))
            w = s + gamma * (u - s)

            v_sw = jax.lax.stop_gradient(vsu_fn(s, w, I_s))
            X_sw = jax.lax.stop_gradient(I_s + (w - s) * v_sw)

            student = vsu_fn(s, u, I_s)
            v_wu = vsu_fn(w, u, X_sw)
            teacher = jax.lax.stop_gradient(gamma * v_sw + (1 - gamma) * v_wu)

        else:  # 'mf' (default): Mean Flow / Eulerian self-distillation
            primals = (s, u, I_s)
            tangents = (jnp.ones_like(s), jnp.zeros_like(u), v_ss)
            v_su, jvp_val = jax.jvp(vsu_fn, primals, tangents)

            teacher = jax.lax.stop_gradient(v_ss + (u - s) * jvp_val)
            student = v_su

        if self.config["action_chunking"]:
            consistency_loss = jnp.mean(
                jnp.reshape(
                    (student - teacher) ** 2,
                    (batch_size, self.config["horizon_length"], self.config["action_dim"])
                ) * batch["valid"][..., None]
            )
        else:
            consistency_loss = jnp.mean(jnp.square(student - teacher))

        consistency_weight = jnp.where(step > self.config['flow_map_warmup_steps'], 1.0, 0.0)

        actor_loss = bc_flow_loss + consistency_weight * consistency_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'consistency_loss': consistency_loss,
            'consistency_weight': consistency_weight,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(self, observations, rng=None):
        """Best-of-N: generate N candidates via flow map, pick best by Q-value."""
        action_dim = self.config['action_dim'] * \
                    (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        M = self.config["actor_num_samples"]
        noises = jax.random.normal(
            rng,
            (*observations.shape[:-len(self.config['ob_dims'])], M, action_dim),
        )
        obs_M = jnp.repeat(observations[..., None, :], M, axis=-2)
        actions = jnp.clip(self.compute_flow_actions(obs_M, noises), -1, 1)

        if self.config["q_agg"] == "mean":
            q = self.network.select("critic")(obs_M, actions).mean(axis=0)
        else:
            q = self.network.select("critic")(obs_M, actions).min(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, M, action_dim))[
            jnp.arange(bsize), indices, :
        ].reshape(bshape + (action_dim,))
        return actions

    @jax.jit
    def compute_flow_actions(self, observations, noises):
        """One-step flow map action generation: a₁ = a₀ + u_{0,1}(a₀|s)."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)

        actions = noises
        n_steps = self.config['flow_map_steps']
        for i in range(n_steps):
            s = jnp.full((*observations.shape[:-1], 1), i / n_steps)
            u = jnp.full((*observations.shape[:-1], 1), (i + 1) / n_steps)
            vels = self.network.select('actor_bc_flow')(
                observations, actions, s, end_times=u, is_encoded=True,
            )
            actions = actions + (u - s) * vels

        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new FlowMapPolicy.

        Instantiates:
          - actor_bc_flow:  flow map velocity u_{r,t}(a_r | s) (Eq. 2)
          - critic:         ensemble Q-network Q_ϕ(s, a)
          - target_critic:  EMA copy Q̄_ϕ
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        )

        actor_bc_flow_def = ActorVectorFieldMFM(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times, ex_times)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():

    config = ml_collections.ConfigDict(
        dict(
            agent_name='offline',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            num_qs=2,
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            actor_type='best-of-n',
            actor_num_samples=32,
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            # Flow map training (§3.1–3.2)
            flow_map_steps=1,               # inference steps (1 = one-step generation)
            flow_map_warmup_steps=5000,     # diagonal-only CFM before off-diagonal
            flow_map_anneal_end_step=50000, # when [r,t] curriculum reaches full range
            distillation_type='mf',         # 'mf' (ESD), 'lsd' (Lagrangian), 'psd' (Progressive)
            # FMQ fine-tuning (used by --fmq_online, see agents/fmq.py)
            fmq_alpha=1.0,
            fmq_sigma_sq=1.0,
            fmq_normalize_grad=True,
            fmq_eta_override=-1.0,
            fmq_grad_at_online=False,
            fmq_adaptive_eta=False,
            fmq_beta=0.3,
            # QGBS inference (Algorithm 2, §3.4)
            qgbs_K=1,
            qgbs_B=4,
            qgbs_eta=0.3,
            qgbs_snr=1.5,
        )
    )
    return config

"""Flow Map Q-Guidance (FMQ) — online fine-tuning agent (§3.3, Algorithm 1).

Training (Eq. 12, Theorem 3.2):
    Regresses the average velocity u^θ_{r,1} onto the optimal trust-region target:
        u*_{r,1}(a_r|s) = u^off_{r,1}(a_r|s) + η · ∇_a Q_ϕ / ‖∇_a Q_ϕ‖

    Total loss: L_FMQ + λ_esd · L_ESD + λ_diag · L_Diag

Inference (Algorithm 2 — QGBS):
    See agents/qgbs.py for the Q-Guided Beam Search implementation.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.flow_utils import sample_s_u
from agents.qgbs import best_of_n, qgbs
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorFieldMFM, Value


class FMQAgent(flax.struct.PyTreeNode):
    """Flow Map Q-Guidance agent (§3.3, Algorithm 1).

    Components:
        actor (θ)           — average velocity network u^θ_{r,t}(a_r | s)
        critic (ϕ₁, ϕ₂)    — clipped double Q-networks Q_ϕ (Eq. 8)
        target_critic       — EMA target networks Q̄_ϕ
        frozen_actor_params — u^off_{r,1}: frozen offline anchor (Eq. 11)

    Online adaptation: Q-gradient evaluated at a₁ = X^off_{0,1}(a₀|s).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    frozen_actor_params: Any = None
    log_alpha: Any = None

    # ─── Critic ──────────────────────────────────────────────────────────

    def critic_loss(self, batch, grad_params, rng):
        """Clipped double Q-learning (Eq. 8).

        y = r + γ^H · min_j Q̄_ϕj(s', X_{0,1}(a'₀|s'))
        """
        batch_actions = self._get_actions(batch)

        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(
            batch['next_observations'][..., -1, :], rng=sample_rng)

        next_qs = self.network.select('target_critic')(
            batch['next_observations'][..., -1, :], actions=next_actions)
        next_q = next_qs.min(axis=0)

        target_q = (batch['rewards'][..., -1]
                    + (self.config['discount'] ** self.config['horizon_length'])
                    * batch['masks'][..., -1] * next_q)

        q = self.network.select('critic')(
            batch['observations'], actions=batch_actions, params=grad_params)
        loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return loss, {
            'critic_loss': loss,
            'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min(),
        }

    # ─── Actor ───────────────────────────────────────────────────────────

    def actor_loss(self, batch, grad_params, rng):
        """L_FMQ: trust-region velocity regression (Eq. 12, Algorithm 1 line 9).

        L = ‖u^θ_{r,1}(a_r|s) - sg(u^off_{r,1} + η_eff · ∇Q/‖∇Q‖)‖²
        """
        batch_actions = self._get_actions(batch)
        batch_size, action_dim = batch_actions.shape
        obs = batch['observations']

        alpha = (jnp.exp(self.log_alpha) if self.log_alpha is not None
                 else self.config['fmq_alpha'])
        eta = self._get_eta(alpha)

        # Swap in frozen offline actor params for trust-region anchor
        frozen_params = {k: v for k, v in grad_params.items()}
        frozen_params['modules_actor_bc_flow'] = self.frozen_actor_params

        # Alg 1 line 5: sample r ~ U[0,1)
        rng, t_rng = jax.random.split(rng)
        r = jax.random.uniform(t_rng, (batch_size, 1))

        fmq_loss, info = self._fmq_loss(
            batch, batch_actions, obs, grad_params, frozen_params,
            r, action_dim, batch_size, eta, rng)

        esd_w = self.config.get('esd_weight', 0.0)
        diag_w = self.config.get('diag_weight', 0.0)
        esd_loss = diag_loss = jnp.zeros(())

        if esd_w > 0:
            rng, sub_rng = jax.random.split(rng)
            esd_loss = self._esd_loss(
                batch, batch_actions, obs, grad_params, action_dim, batch_size, sub_rng)
        if diag_w > 0:
            rng, sub_rng = jax.random.split(rng)
            diag_loss = self._diag_loss(
                batch, batch_actions, obs, grad_params, action_dim, batch_size, sub_rng)

        total = fmq_loss + esd_w * esd_loss + diag_w * diag_loss
        info.update(actor_loss=total, fmq_loss=fmq_loss,
                    esd_loss=esd_loss, diag_loss=diag_loss,
                    fmq_alpha=alpha, fmq_eta=eta)
        return total, info

    def _fmq_loss(self, batch, batch_actions, obs, grad_params, frozen_params,
                  r, action_dim, B, eta, rng):
        """Core FMQ loss (Eq. 12)."""
        omr = 1.0 - r
        rng, eps_rng = jax.random.split(rng)
        eps = jax.random.normal(eps_rng, (B, action_dim))
        # Alg 1 line 5: a_r = (1-r)·a₀ + r·a_data
        a_r = omr * eps + r * batch_actions
        t1 = jnp.ones((B, 1))

        # u^θ_{r,1}(a_r | s): online average velocity
        u_theta = self.network.select('actor_bc_flow')(
            obs, a_r, r, end_times=t1, params=grad_params)
        a1_online = jnp.clip(a_r + omr * u_theta, -1, 1)

        # u^off_{r,1}(a_r | s): frozen offline average velocity
        u_off = self.network.select('actor_bc_flow')(
            obs, a_r, r, end_times=t1, params=frozen_params)
        a1_off = jnp.clip(a_r + omr * u_off, -1, 1)

        # Alg 1 line 7: g = ∇_a Q_ϕ(s, a₁) / (‖∇_a Q_ϕ‖ + κ₁)
        grad_at_online = self.config.get('fmq_grad_at_online', False)
        a_grad = a1_online if grad_at_online else a1_off

        def q_fn(a):
            return self.network.select('critic')(obs, a)[0].sum()
        g = jax.grad(q_fn)(a_grad)
        g_raw_norm = jnp.linalg.norm(g)

        if self.config.get('fmq_normalize_grad', True):
            g = g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)

        # Alg 1 line 8: η_eff = 1 / (1 + β·δ̃_critic)  (Eq. 13)
        adaptive_eta = self.config.get('fmq_adaptive_eta', False)
        if adaptive_eta:
            beta = self.config.get('fmq_beta', 1.0)
            a_sg = jax.lax.stop_gradient(a1_off)
            qs_all = self.network.select('critic')(obs, a_sg)
            q_std = qs_all.std(axis=0)
            q_std_rel = q_std / (q_std.mean() + 1e-8)
            eta_eff = (1.0 / (1.0 + beta * q_std_rel))[..., None]
        else:
            eta_eff = eta

        # Alg 1 line 9: θ ← θ - α·∇‖u^θ - sg(u^off + η_eff·g)‖²
        u_target = jax.lax.stop_gradient(
            u_off + eta_eff * jax.lax.stop_gradient(g))
        loss = self._regression_loss(batch, u_theta - u_target, B)

        return loss, {
            'q_mean_sampled': self.network.select('critic')(obs, a1_online).min(0).mean(),
            'grad_raw_norm': g_raw_norm,
            'eta_effective': jnp.mean(eta_eff) if adaptive_eta else eta,
            'delta_u_sq': jnp.mean(jnp.square(u_theta - jax.lax.stop_gradient(u_off))),
            'r_mean': r.mean(),
        }

    # ─── Auxiliary losses (§3.1) ─────────────────────────────────────────

    def _esd_loss(self, batch, batch_actions, obs, grad_params,
                  action_dim, B, rng):
        """L_ESD: Eulerian self-distillation (Eq. 5, off-diagonal r < t)."""
        rng, x_rng, rt_rng = jax.random.split(rng, 3)
        x0 = jax.random.normal(x_rng, (B, action_dim))
        v_rt = batch_actions - x0

        step = self.network.step
        r, t = sample_s_u(rt_rng, B, step,
                          self.config.get('esd_warmup_steps', 0),
                          self.config.get('esd_anneal_end_step', 50000))
        a_r = (1 - r) * x0 + r * batch_actions

        def u_rt_fn(r_in, t_in, x_in):
            return self.network.select('actor_bc_flow')(
                obs, x_in, r_in, end_times=t_in, params=grad_params)

        _, jvp_val = jax.jvp(u_rt_fn, (r, t, a_r),
                             (jnp.ones_like(r), jnp.zeros_like(t), v_rt))
        teacher = jax.lax.stop_gradient(v_rt + (t - r) * jvp_val)
        student = u_rt_fn(r, t, a_r)
        return self._regression_loss(batch, student - teacher, B)

    def _diag_loss(self, batch, batch_actions, obs, grad_params,
                   action_dim, B, rng):
        """L_Diag: diagonal flow matching loss (Eq. 3, r = t)."""
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x0 = jax.random.normal(x_rng, (B, action_dim))
        tc = jax.random.uniform(t_rng, (B, 1))
        a_t = (1 - tc) * x0 + tc * batch_actions
        vel = batch_actions - x0

        pred = self.network.select('actor_bc_flow')(
            obs, a_t, tc, end_times=tc, params=grad_params)
        return self._regression_loss(batch, pred - vel, B)

    # ─── Update loop ─────────────────────────────────────────────────────

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, a_rng, c_rng = jax.random.split(rng, 3)

        c_loss, c_info = self.critic_loss(batch, grad_params, c_rng)
        a_loss, a_info = self.actor_loss(batch, grad_params, a_rng)
        for k, v in c_info.items():
            info[f'critic/{k}'] = v
        for k, v in a_info.items():
            info[f'actor/{k}'] = v
        return c_loss + a_loss, info

    def _target_update(self, network):
        tp = jax.tree_util.tree_map(
            lambda p, t: p * self.config['tau'] + t * (1 - self.config['tau']),
            self.network.params['modules_critic'],
            self.network.params['modules_target_critic'])
        network.params['modules_target_critic'] = tp

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)
        new_net, info = agent.network.apply_loss_fn(
            loss_fn=lambda p: agent.total_loss(batch, p, rng=rng))
        agent._target_update(new_net)
        return agent.replace(network=new_net, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    # ─── Inference ───────────────────────────────────────────────────────

    @jax.jit
    def sample_actions(self, observations, rng=None):
        action_dim = self.config['action_dim'] * (
            self.config['horizon_length'] if self.config['action_chunking'] else 1)

        if self.config['encoder'] is not None:
            obs_enc = self.network.select('actor_bc_flow_encoder')(observations)
        else:
            obs_enc = observations

        actor_type = self.config['actor_type']

        if actor_type == 'best-of-n':
            return self._best_of_n(observations, obs_enc, action_dim, rng)
        elif actor_type == 'qgbs':
            return self._qgbs(observations, obs_enc, action_dim, rng)
        else:
            noises = jax.random.normal(
                rng, (*observations.shape[:-len(self.config['ob_dims'])], action_dim))
            return jnp.clip(self._flow_map(obs_enc, noises, action_dim), -1, 1)

    def _best_of_n(self, obs, obs_enc, action_dim, rng):
        return best_of_n(self.network, obs, obs_enc, action_dim,
                         self.config, rng, self._flow_map)

    def _qgbs(self, obs, obs_enc, action_dim, rng):
        return qgbs(self.network, obs, obs_enc, action_dim,
                    self.config, rng, self._flow_map)

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _get_actions(self, batch):
        if self.config['action_chunking']:
            return jnp.reshape(batch['actions'], (batch['actions'].shape[0], -1))
        return batch['actions'][..., 0, :]

    def _get_eta(self, alpha):
        eta_override = self.config.get('fmq_eta_override', -1.0)
        if eta_override >= 0:
            return float(eta_override)
        return self.config['fmq_sigma_sq'] / (2.0 * alpha)

    def _flow_map(self, obs_enc, noises, action_dim):
        """One-step flow map: a₁ = a₀ + u^θ_{0,1}(a₀ | s) (Eq. 7)."""
        s0 = jnp.zeros((*obs_enc.shape[:-1], 1))
        u1 = jnp.ones((*obs_enc.shape[:-1], 1))
        v = self.network.select('actor_bc_flow')(
            obs_enc, noises, s0, end_times=u1, is_encoded=True)
        return noises + v

    def _regression_loss(self, batch, residual, B):
        if self.config['action_chunking']:
            return jnp.mean(
                jnp.reshape(residual ** 2,
                            (B, self.config['horizon_length'], self.config['action_dim']))
                * batch['valid'][..., None])
        return jnp.mean(jnp.square(residual))

    # ─── Construction ────────────────────────────────────────────────────

    @classmethod
    def create_from_pretrained(cls, pretrained_agent, fmq_config=None):
        """Wrap a pretrained offline agent, freezing u^off_{r,1} as anchor (Eq. 11)."""
        frozen = jax.tree_util.tree_map(
            lambda x: x, pretrained_agent.network.params['modules_actor_bc_flow'])

        config = dict(pretrained_agent.config)
        config['agent_name'] = 'fmq'
        if fmq_config:
            config.update(fmq_config)

        for k, v in _DEFAULTS.items():
            config.setdefault(k, v)

        valid = ('best-of-n', 'qgbs')
        if config.get('actor_type') not in valid:
            config['actor_type'] = 'best-of-n'

        return cls(
            rng=pretrained_agent.rng,
            network=pretrained_agent.network,
            config=flax.core.FrozenDict(**config),
            frozen_actor_params=frozen,
            log_alpha=jnp.log(jnp.array(config['fmq_alpha'], dtype=jnp.float32)),
        )

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create from scratch (for deserialization / standalone training)."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        ex_times = ex_actions[..., :1]
        full_actions = (jnp.concatenate([ex_actions] * config['horizon_length'], axis=-1)
                        if config['action_chunking'] else ex_actions)
        full_action_dim = full_actions.shape[-1]

        encoders = {}
        if config['encoder'] is not None:
            enc = encoder_modules[config['encoder']]
            encoders = {k: enc() for k in ('critic', 'actor_bc_flow', 'actor_onestep_flow')}

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'))

        actor_def = ActorVectorFieldMFM(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config['use_fourier_features'],
            fourier_feature_dim=config['fourier_feature_dim'])

        flow_args = (ex_observations, full_actions, ex_times, ex_times)

        network_info = dict(
            actor_bc_flow=(actor_def, flow_args),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        if encoders.get('actor_bc_flow'):
            network_info['actor_bc_flow_encoder'] = (encoders['actor_bc_flow'], (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)

        tx = (optax.adamw(config['lr'], weight_decay=config['weight_decay'])
              if config['weight_decay'] > 0 else optax.adam(config['lr']))
        params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, params, tx=tx)
        network.params['modules_target_critic'] = network.params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        frozen = jax.tree_util.tree_map(lambda x: x, network.params['modules_actor_bc_flow'])

        return cls(rng, network=network,
                   config=flax.core.FrozenDict(**config),
                   frozen_actor_params=frozen,
                   log_alpha=jnp.log(jnp.array(config.get('fmq_alpha', 1.0), dtype=jnp.float32)))


_DEFAULTS = dict(
    fmq_alpha=1.0, fmq_sigma_sq=1.0,
    fmq_normalize_grad=True,
    fmq_eta_override=-1.0, fmq_grad_at_online=False,
    fmq_adaptive_eta=False, fmq_beta=0.3,
    esd_weight=0.0, diag_weight=0.0,
    esd_warmup_steps=0, esd_anneal_end_step=50000,
)


def get_config():
    config = ml_collections.ConfigDict(dict(
        agent_name='fmq',
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
        q_agg='min',
        num_qs=2,
        encoder=ml_collections.config_dict.placeholder(str),
        horizon_length=ml_collections.config_dict.placeholder(int),
        action_chunking=True,
        use_fourier_features=False,
        fourier_feature_dim=64,
        weight_decay=0.,
        # Flow map offline pre-training (§3.2)
        flow_map_steps=1,
        flow_map_warmup_steps=5000,
        flow_map_anneal_end_step=50000,
        # Inference action selection
        actor_type='best-of-n',         # 'best-of-n' or 'qgbs'
        actor_num_samples=32,           # M: beam width / initial candidates
        # FMQ trust-region (§3.3, Theorem 3.2)
        fmq_alpha=1.0,                  # η = σ²/(2α), controls trust-region radius
        fmq_sigma_sq=1.0,              # σ² in η computation
        fmq_normalize_grad=True,        # ∇Q/‖∇Q‖ (Theorem 3.2)
        fmq_eta_override=-1.0,          # override computed η if >= 0
        fmq_grad_at_online=False,       # ∇Q at a₁^off (default) vs a₁^on
        fmq_adaptive_eta=False,         # η_eff via Eq. 13
        fmq_beta=1.0,                   # β in Eq. 13 (critic disagreement sensitivity)
        # Off-diagonal regularization (§3.1)
        esd_weight=0.0,                 # λ_ESD: Eulerian self-distillation (Eq. 5)
        diag_weight=0.0,                # λ_Diag: diagonal CFM (Eq. 3)
        esd_warmup_steps=0,
        esd_anneal_end_step=50000,
        # QGBS inference (Algorithm 2, §3.4)
        qgbs_K=1,                       # K: beam search iterations
        qgbs_B=4,                       # B: branches per particle
        qgbs_eta=0.3,                   # η: trust-region step size
        qgbs_snr=1.5,                   # ρ: SNR → t' = ρ/(1+ρ) (Eq. 14)
    ))
    return config

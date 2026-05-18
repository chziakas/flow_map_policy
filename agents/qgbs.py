"""Q-Guided Beam Search (QGBS) — inference-time action selection (Algorithm 2).

Iteratively refines flow map samples via SNR-based renoising (Eq. 14),
beam selection, and trust-region Q-gradient projection (Theorem 3.2).
Deployed only at inference; does not affect training.

NFE = M(1 + K·B) per action selection.
"""

import math
import jax
import jax.numpy as jnp


def best_of_n(network, obs, obs_enc, action_dim, config, rng, flow_map_fn):
    """Best-of-N: sample M from X_{0,1}, return argmax Q (K=0 baseline)."""
    M = config['actor_num_samples']
    bshape = obs.shape[:-len(config['ob_dims'])]
    noises = jax.random.normal(rng, (*bshape, M, action_dim))
    obs_enc_M = jnp.repeat(obs_enc[..., None, :], M, axis=-2)
    obs_raw_M = jnp.repeat(obs[..., None, :], M, axis=-2)

    actions = jnp.clip(flow_map_fn(obs_enc_M, noises, action_dim), -1, 1)
    q = network.select('critic')(obs_raw_M, actions).min(axis=0)
    idx = jnp.argmax(q, axis=-1)

    return jnp.reshape(actions, (-1, M, action_dim))[
        jnp.arange(idx.reshape(-1).shape[0]), idx.reshape(-1), :
    ].reshape(*bshape, action_dim)


def qgbs(network, obs, obs_enc, action_dim, config, rng, flow_map_fn):
    """Q-Guided Beam Search (Algorithm 2, §3.4).

    Hyperparameters (from config):
        M   = actor_num_samples — beam width (particles retained per step)
        B   = qgbs_B            — branches per particle (re-noise copies)
        K   = qgbs_K            — number of explore-exploit iterations
        η   = qgbs_eta          — trust-region step size (Theorem 3.2)
        ρ   = qgbs_snr          — SNR for renoising: t' = ρ/(1+ρ) (Eq. 14)
    """
    M = config.get('actor_num_samples', 32)
    B_mc = config.get('qgbs_B', 4)
    K = config.get('qgbs_K', 1)
    eta = config.get('qgbs_eta', 0.3)
    snr = config.get('qgbs_snr', 1.5)
    norm_g = config.get('fmq_normalize_grad', True)

    bshape = obs.shape[:-len(config['ob_dims'])]
    MK = M * B_mc
    t_prime = snr / (1.0 + snr)
    sigma_tp = 1.0 - t_prime

    # Line 2: sample M initial candidates a₁ = X_{0,1}(a₀|s)
    rng, noise_rng = jax.random.split(rng)
    noises = jax.random.normal(noise_rng, (*bshape, M, action_dim))
    obs_enc_M = jnp.repeat(obs_enc[..., None, :], M, axis=-2)
    actions = jnp.clip(flow_map_fn(obs_enc_M, noises, action_dim), -1, 1)

    # Lines 3–10: K iterations of explore (renoising) + exploit (beam + projection)
    for _ in range(K):
        rng, rn_rng = jax.random.split(rng)
        eps = jax.random.normal(rn_rng, (*bshape, MK, action_dim))

        # Line 6: re-noise a_{t'} = t'·a₁ + (1-t')·ε  (Eq. 14, B copies)
        x_tp = t_prime * jnp.repeat(actions, B_mc, axis=-2) + sigma_tp * eps

        obs_enc_MK = jnp.repeat(obs_enc[..., None, :], MK, axis=-2)
        obs_raw_MK = jnp.repeat(obs[..., None, :], MK, axis=-2)

        # Line 6: denoise â₁ = a_{t'} + (1-t')·u_{t',t'}(a_{t'}|s)
        t_tens = jnp.full((*bshape, MK, 1), t_prime)
        v = network.select('actor_bc_flow')(
            obs_enc_MK, x_tp, t_tens, end_times=t_tens, is_encoded=True)
        comp = jnp.clip(x_tp + sigma_tp * v, -1, 1)

        # Line 8: score with Q_ϕ and select top-M (beam selection)
        q_all = network.select('critic')(obs_raw_MK, comp).min(axis=0)
        flat_b = math.prod(bshape) if bshape else 1
        top_idx = jnp.argsort(q_all.reshape(flat_b, MK), axis=-1)[..., -M:]
        actions = comp.reshape(flat_b, MK, action_dim)[
            jnp.arange(flat_b)[:, None], top_idx, :
        ].reshape(*bshape, M, action_dim)

        # Line 9: trust-region projection a ← a + η·∇Q/‖∇Q‖ (Theorem 3.2)
        obs_raw_M = jnp.repeat(obs[..., None, :], M, axis=-2)

        def q_step(a):
            return network.select('critic')(obs_raw_M, a).min(axis=0).sum()
        g = jax.grad(q_step)(actions)
        if norm_g:
            g = g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
        actions = jnp.clip(actions + eta * g, -1, 1)

    # K=0 with η>0: single trust-region projection without renoising
    if K == 0 and eta > 0:
        obs_raw_M = jnp.repeat(obs[..., None, :], M, axis=-2)

        def q_k0(a):
            return network.select('critic')(obs_raw_M, a).min(axis=0).sum()
        g = jax.grad(q_k0)(actions)
        if norm_g:
            g = g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
        actions = jnp.clip(actions + eta * g, -1, 1)

    # Line 11: return argmax_m Q_ϕ(s, a₁_m)
    obs_raw_M = jnp.repeat(obs[..., None, :], M, axis=-2)
    q_final = network.select('critic')(obs_raw_M, actions).min(axis=0)
    best = jnp.argmax(q_final, axis=-1)
    return jnp.reshape(actions, (-1, M, action_dim))[
        jnp.arange(best.reshape(-1).shape[0]), best.reshape(-1), :
    ].reshape(*bshape, action_dim)

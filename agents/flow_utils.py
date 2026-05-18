"""Shared sampling utilities for flow map policy training.

Implements the off-diagonal training curriculum that allows the flow map
actor to learn arbitrary-size jumps X_{r,t} over intervals [r, t] ⊂ [0, 1].
See §3.1 (Eqs. 3–6) in the paper.
"""

import jax
import jax.numpy as jnp


def sample_r_t(rng, batch_size, step, warmup_steps, anneal_end_step):
    """Sample off-diagonal time pair (r, t) with annealing curriculum.

    Controls which flow map intervals X_{r,t} are trained:
      Phase 1 (step < warmup):    r = t (diagonal only, standard CFM)
      Phase 2 (warmup..anneal):   interval [r, t] grows gradually
      Phase 3 (step >= anneal):   full random [r, t] ⊂ [0, 1]

    Returns (r, t) each shape (batch_size, 1), 0 <= r <= t <= 1.
    """
    rng1, rng2 = jax.random.split(rng)
    t1 = jax.random.uniform(rng1, (batch_size, 1))
    t2 = jax.random.uniform(rng2, (batch_size, 1))

    t_min = jnp.minimum(t1, t2)
    t_max = jnp.maximum(t1, t2)
    mid = (t_min + t_max) / 2
    dist = t_max - t_min

    anneal_duration = jnp.maximum(anneal_end_step - warmup_steps, 1)
    progress = jnp.clip((step - warmup_steps) / anneal_duration, 0.0, 1.0)
    max_step_size = jnp.where(step < warmup_steps, 0.0, progress)

    r = mid - max_step_size * dist / 2
    t = mid + max_step_size * dist / 2
    return r, t


# Backward-compatible alias
sample_s_u = sample_r_t

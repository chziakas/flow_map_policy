"""Two-phase offline-to-online RL training pipeline for FMQ.

FMQ (Flow-Map Q Fine-Tuning): MFM offline then trust-region Q-steering online
                        (agent=offline_pretraining.py offline -> fmq.py online)

Pipeline:
  Phase 1 — Offline RL (--offline_steps):
      Train the flow-map actor (CFM/MFM losses) and critic (clipped double Q-learning)
      on a fixed offline dataset.  Supports periodic dataset rotation for
      large OGBench datasets (--ogbench_dataset_dir).

  Phase 2 — Online RL (--online_steps):
      Fine-tune with environment interaction.  When --fmq_online is set,
      the agent is replaced by FMQAgent which freezes the pretrained
      drift and regresses toward v_target = v_base + eta * grad_a Q.

Flags:
  --agent           Agent config file (agents/flow_map_policy.py)
  --fmq_online      Switch to FMQ trust-region fine-tuning at online boundary
  --skip_offline    Skip offline phase, restore from --restore_path
  --eval_only       Run evaluation only (no training)
  --horizon_length  Action chunking horizon H (default 5)
"""
import os, sys
import yaml

_gpu_from_cli = None
_config_path = None
for arg in sys.argv[1:]:
    if arg.startswith('--gpu='):
        _gpu_from_cli = arg.split('=', 1)[1]
    if arg.startswith('--config='):
        _config_path = arg.split('=', 1)[1]

if _gpu_from_cli is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_from_cli
elif _config_path is not None:
    with open(_config_path, 'r') as _f:
        _cfg = yaml.safe_load(_f)
    if 'training' in _cfg and 'gpu' in _cfg['training']:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_cfg['training']['gpu'])

if '--benchmark_cpu' in sys.argv or '--benchmark_cpu=True' in sys.argv:
    os.environ['JAX_PLATFORMS'] = 'cpu'

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

import glob, tqdm, wandb, json, random, time, jax, flax
from absl import app, flags
from ml_collections import config_flags
from utils.log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent, restore_agent_with_file
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('gpu', None, 'GPU device ID (e.g. "1" or "0,1"). Sets CUDA_VISIBLE_DEVICES.')
flags.DEFINE_string('wandb_project', 'fmq', 'Weights & Biases project name.')
flags.DEFINE_string('exp_suffix', '', 'Suffix appended to experiment name (e.g. "cluster").')
flags.DEFINE_string('run_group', 'default', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/flow_map_policy.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

flags.DEFINE_bool('eval_only', False, 'Run evaluation only (skip training), requires --restore_path')
flags.DEFINE_string('restore_path', None, 'Path to checkpoint file (.pkl) to restore agent from')
flags.DEFINE_bool('benchmark_cpu', False, 'Run inference benchmark on CPU without JIT. Use with --eval_only.')
flags.DEFINE_string('online_actor_type', None,
    'Actor type to use during online phase. If None, keeps the same as agent.actor_type.')
flags.DEFINE_bool('fmq_online', False,
    'Switch to FMQAgent (FMQ fine-tuning) for the online phase.')
flags.DEFINE_bool('restore_base_ckpt', False,
    'Restore a base checkpoint, then convert to FMQ agent. '
    'Use with --fmq_online --eval_only for offline pretrained checkpoints.')
flags.DEFINE_bool('skip_offline', False,
    'Skip offline training and restore from --restore_path, then proceed to online phase.')
flags.DEFINE_string('config', None,
    'Path to YAML config file. Values override flag defaults; CLI flags override YAML.')


def _apply_yaml_config(yaml_path):
    """Load a YAML config and apply values as flag defaults (CLI flags take precedence)."""
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    _YAML_TO_FLAGS = {
        'env.env_name': 'env_name',
        'env.horizon_length': 'horizon_length',
        'env.sparse': 'sparse',
        'training.seed': 'seed',
        'training.gpu': 'gpu',
        'training.offline_steps': 'offline_steps',
        'training.online_steps': 'online_steps',
        'training.buffer_size': 'buffer_size',
        'training.start_training': 'start_training',
        'training.utd_ratio': 'utd_ratio',
        'training.discount': 'discount',
        'training.fmq_online': 'fmq_online',
        'logging.wandb_project': 'wandb_project',
        'logging.run_group': 'run_group',
        'logging.exp_suffix': 'exp_suffix',
        'logging.save_dir': 'save_dir',
        'logging.log_interval': 'log_interval',
        'logging.eval_interval': 'eval_interval',
        'logging.save_interval': 'save_interval',
        'evaluation.eval_episodes': 'eval_episodes',
        'evaluation.video_episodes': 'video_episodes',
        'evaluation.video_frame_skip': 'video_frame_skip',
        'dataset.dataset_proportion': 'dataset_proportion',
        'dataset.dataset_replace_interval': 'dataset_replace_interval',
        'dataset.ogbench_dataset_dir': 'ogbench_dataset_dir',
    }

    _YAML_TO_AGENT = {
        'model.actor_hidden_dims': 'actor_hidden_dims',
        'model.value_hidden_dims': 'value_hidden_dims',
        'model.layer_norm': 'layer_norm',
        'model.actor_layer_norm': 'actor_layer_norm',
        'model.num_qs': 'num_qs',
        'model.q_agg': 'q_agg',
        'model.encoder': 'encoder',
        'model.use_fourier_features': 'use_fourier_features',
        'model.fourier_feature_dim': 'fourier_feature_dim',
        'model.weight_decay': 'weight_decay',
        'model.action_chunking': 'action_chunking',
        'flow_map.flow_map_steps': 'flow_map_steps',
        'flow_map.flow_map_warmup_steps': 'flow_map_warmup_steps',
        'flow_map.flow_map_anneal_end_step': 'flow_map_anneal_end_step',
        'flow_map.distillation_type': 'distillation_type',
        'optimizer.lr': 'lr',
        'optimizer.batch_size': 'batch_size',
        'fmq.fmq_alpha': 'fmq_alpha',
        'fmq.fmq_sigma_sq': 'fmq_sigma_sq',
        'fmq.fmq_normalize_grad': 'fmq_normalize_grad',
        'fmq.fmq_eta_override': 'fmq_eta_override',
        'fmq.fmq_grad_at_online': 'fmq_grad_at_online',
        'fmq.fmq_adaptive_eta': 'fmq_adaptive_eta',
        'fmq.fmq_beta': 'fmq_beta',
        'inference.actor_type': 'actor_type',
        'inference.actor_num_samples': 'actor_num_samples',
        'qgbs.qgbs_K': 'qgbs_K',
        'qgbs.qgbs_B': 'qgbs_B',
        'qgbs.qgbs_eta': 'qgbs_eta',
        'qgbs.qgbs_snr': 'qgbs_snr',
    }

    agent_overrides = {}
    for yaml_key, flag_name in _YAML_TO_FLAGS.items():
        section, key = yaml_key.split('.', 1)
        if section in cfg and key in cfg[section]:
            val = cfg[section][key]
            if val is not None and not FLAGS[flag_name].present:
                FLAGS[flag_name].value = val

    for yaml_key, agent_key in _YAML_TO_AGENT.items():
        section, key = yaml_key.split('.', 1)
        if section in cfg and key in cfg[section]:
            val = cfg[section][key]
            if val is not None:
                agent_overrides[agent_key] = val

    return agent_overrides


class LoggingHelper:
    """Unified logger that writes to both per-prefix CSV files and Weights & Biases.

    Tracks training speed (iter/s) for offline and online phases, and records
    evaluation results for the final results.json summary.
    """

    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()
        self.speed_history = {'offline': [], 'online': []}
        self.eval_history = []

    def get_speed(self, interval, phase='offline'):
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now
        if elapsed > 0:
            speed = interval / elapsed
            self.speed_history[phase].append(speed)
            return speed
        return 0.0

    def record_eval(self, eval_info, step, phase):
        entry = {'step': int(step), 'phase': phase}
        for k, v in eval_info.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                entry[k] = float(v)
        self.eval_history.append(entry)

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

def main(_):
    if FLAGS.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
        os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

    method_tag = FLAGS.run_group.replace('benchmark_', '')
    exp_name = get_exp_name(FLAGS.seed, env_name=FLAGS.env_name, method=method_tag)
    if FLAGS.exp_suffix:
        exp_name = exp_name + '_' + FLAGS.exp_suffix
    run = setup_wandb(project=FLAGS.wandb_project, group=FLAGS.run_group, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent

    if FLAGS.config is not None:
        agent_overrides = _apply_yaml_config(FLAGS.config)
        for k, v in agent_overrides.items():
            if k in config:
                existing = config[k]
                if existing is not None:
                    v = type(existing)(v)
            config[k] = v
        if 'actor_type' in agent_overrides and agent_overrides['actor_type'] == 'qgbs':
            FLAGS.online_actor_type = 'qgbs'

    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0
    
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    # handle dataset
    def process_train_dataset(ds):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        if is_robomimic_env(FLAGS.env_name):
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)
        
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    if FLAGS.eval_only:
        assert FLAGS.restore_path is not None, '--restore_path is required when using --eval_only'
        _FMQ_KEYS = (
            'fmq_alpha', 'fmq_sigma_sq', 'fmq_normalize_grad',
            'fmq_eta_override', 'fmq_grad_at_online',
            'fmq_adaptive_eta', 'fmq_beta',
            'actor_type', 'actor_num_samples',
            'qgbs_K', 'qgbs_B', 'qgbs_eta', 'qgbs_snr',
        )
        if FLAGS.restore_base_ckpt and FLAGS.fmq_online:
            agent = restore_agent_with_file(agent, FLAGS.restore_path)
            print(f"Restored base checkpoint (before FMQ conversion)")
            from agents.fmq import FMQAgent
            fmq_overrides = {k: config[k] for k in _FMQ_KEYS if k in config}
            if FLAGS.online_actor_type is not None:
                fmq_overrides['actor_type'] = FLAGS.online_actor_type
            agent = FMQAgent.create_from_pretrained(agent, fmq_config=fmq_overrides)
            print(f"Converted to FMQAgent for eval_only (actor_type={fmq_overrides.get('actor_type')})")
        else:
            if FLAGS.fmq_online:
                from agents.fmq import FMQAgent
                fmq_overrides = {k: config[k] for k in _FMQ_KEYS if k in config}
                if FLAGS.online_actor_type is not None:
                    fmq_overrides['actor_type'] = FLAGS.online_actor_type
                agent = FMQAgent.create_from_pretrained(agent, fmq_config=fmq_overrides)
                print(f"Converted to FMQAgent for eval_only (actor_type={fmq_overrides.get('actor_type')})")
            agent = restore_agent_with_file(agent, FLAGS.restore_path)
        eval_info, trajs, renders = evaluate(
            agent=agent,
            env=eval_env,
            action_dim=example_batch["actions"].shape[-1],
            num_eval_episodes=FLAGS.eval_episodes,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
            benchmark=FLAGS.benchmark_cpu,
        )
        summary = []
        summary.append('=' * 40)
        summary.append('         FINAL SUMMARY')
        summary.append('=' * 40)
        summary.append(f'Environment:    {FLAGS.env_name}')
        summary.append(f'Agent:          {config.get("agent_name", "unknown")}')
        summary.append(f'Seed:           {FLAGS.seed}')
        summary.append(f'\nEvaluation ({FLAGS.eval_episodes} episodes):')
        summary.append(f'  Success Rate:       {eval_info.get("success", 0) * 100:.1f}%')
        summary.append(f'  Mean Return:        {eval_info.get("episode.return", 0):.1f}')
        if 'inference_ms_mean' in eval_info:
            summary.append(f'\nInference (CPU, no JIT):')
            summary.append(f'  Latency:            {eval_info["inference_ms_mean"]:.1f} +/- {eval_info.get("inference_ms_std", 0):.1f} ms/action')
        summary.append('=' * 40)

        summary_text = '\n'.join(summary)
        print('\n' + summary_text)

        save_dir = os.path.dirname(FLAGS.restore_path)
        summary_file = os.path.join(save_dir, 'benchmark_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(summary_text + '\n')
        print(f'\nSaved to {summary_file}')
        return

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes},
        wandb_logger=wandb,
    )

    if FLAGS.skip_offline:
        assert FLAGS.restore_path is not None, '--restore_path is required when using --skip_offline'
        agent = restore_agent_with_file(agent, FLAGS.restore_path)
        print(f"Skipped offline training; restored agent from {FLAGS.restore_path}")
        log_step += FLAGS.offline_steps

    offline_init_time = time.time()
    # ======================== Phase 1: Offline RL ========================
    # Train actor (CFM + optional MFM self-consistency) and critic (clipped
    # double Q-learning) on the offline dataset.  For large OGBench datasets,
    # shards are cycled every dataset_replace_interval steps to limit memory.
    for i in tqdm.tqdm(range(1, 1 if FLAGS.skip_offline else FLAGS.offline_steps + 1)):
        log_step += 1

        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
            )
            train_dataset = process_train_dataset(train_dataset)

        batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)

        agent, offline_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            offline_info['iter_per_sec'] = logger.get_speed(FLAGS.log_interval, 'offline')
            logger.log(offline_info, "offline_agent", step=log_step)
        
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

        # eval
        if i == FLAGS.offline_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            # during eval, the action chunk is executed fully
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)
            logger.record_eval(eval_info, log_step, 'offline')

    save_agent(agent, FLAGS.save_dir, 'offline')

    # ============== Offline-to-Online Transition: SFQ V2 Agent Swap ==============
    # When --fmq_online is set, replace the offline agent with FMQAgent.
    # create_from_pretrained() snapshots the current actor weights as a frozen
    # anchor (u^off), so the online trust-region loss regresses toward
    # u^off + eta * normalize(grad_a Q).  FMQ config keys are forwarded.
    if FLAGS.fmq_online:
        from agents.fmq import FMQAgent
        fmq_overrides = {}
        for key in ('fmq_alpha', 'fmq_sigma_sq', 'fmq_normalize_grad',
                    'fmq_eta_override', 'fmq_grad_at_online',
                    'fmq_adaptive_eta', 'fmq_beta',
                    'qgbs_K', 'qgbs_B', 'qgbs_eta', 'qgbs_snr',
                    'actor_type', 'actor_num_samples',
                    'esd_weight', 'diag_weight',
                    'esd_warmup_steps', 'esd_anneal_end_step'):
            if key in config:
                fmq_overrides[key] = config[key]
        if FLAGS.online_actor_type is not None:
            fmq_overrides['actor_type'] = FLAGS.online_actor_type
        agent = FMQAgent.create_from_pretrained(
            agent, fmq_config=fmq_overrides
        )
        print(f"Switched to FMQAgent for online FMQ fine-tuning")
    elif FLAGS.online_actor_type is not None:
        new_config = dict(agent.config)
        new_config['actor_type'] = FLAGS.online_actor_type
        agent = agent.replace(config=flax.core.FrozenDict(**new_config))
        print(f"Switched actor_type to '{FLAGS.online_actor_type}' for online phase")

    # Seed the replay buffer with the offline dataset so online training
    # mixes offline and freshly collected transitions.
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
        
    ob, _ = env.reset()
    
    action_queue = []
    action_dim = example_batch["actions"].shape[-1]

    # ======================== Phase 2: Online RL ========================
    # Interact with the environment, collect transitions into the replay
    # buffer, and update the agent.  Action chunks are executed open-loop:
    # the full H-step chunk is queued and popped one action at a time.
    # With UTD > 1 the batch is reshaped into (utd_ratio, batch_size, ...)
    # and processed via jax.lax.scan in batch_update for efficiency.
    update_info = {}

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)
        
        # during online rl, the action chunk is executed fully
        if len(action_queue) == 0:
            action = agent.sample_actions(observations=ob, rng=key)

            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)
        
        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))
        
        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        # Shift rewards to be non-positive: D4RL antmaze gives +1 on goal,
        # robomimic gives +1 on success — subtract 1 so that the offline
        # dataset (which already uses this convention) and online rewards match.
        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            int_reward = int_reward - 1.0
        elif is_robomimic_env(FLAGS.env_name):
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)
        
        # done
        if done:
            ob, _ = env.reset()
            action_queue = []  # reset the action queue
        else:
            ob = next_ob

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio, 
                        sequence_length=FLAGS.horizon_length, discount=discount)
            batch = jax.tree.map(lambda x: x.reshape((
                FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)

            agent, update_info["online_agent"] = agent.batch_update(batch)
            
        if i % FLAGS.log_interval == 0:
            speed = logger.get_speed(FLAGS.log_interval, 'online')
            for key, info in update_info.items():
                info['iter_per_sec'] = speed
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)
            logger.record_eval(eval_info, log_step, 'online')

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

    end_time = time.time()

    # Inference latency benchmark: 5 warmup calls to trigger JIT compilation,
    # then 50 timed calls with block_until_ready to measure wall-clock latency.
    inference_times = []
    inference_rng = jax.random.PRNGKey(42)
    dummy_obs = np.zeros_like(example_batch['observations'])
    for _warmup in range(5):
        inference_rng, k = jax.random.split(inference_rng)
        _ = agent.sample_actions(observations=dummy_obs, rng=k)
    for _trial in range(50):
        inference_rng, k = jax.random.split(inference_rng)
        t0 = time.time()
        _ = agent.sample_actions(observations=dummy_obs, rng=k)
        jax.block_until_ready(_)
        inference_times.append((time.time() - t0) * 1000.0)
    inference_ms_mean = float(np.mean(inference_times))
    inference_ms_std = float(np.std(inference_times))

    save_agent(agent, FLAGS.save_dir, 'online')

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {"steps": np.array(data["steps"]),
                 "qpos": np.stack(data["qpos"], axis=0), 
                 "qvel": np.stack(data["qvel"], axis=0), 
                 "obs": np.stack(data["obs"], axis=0), 
                 "offline_time": online_init_time - offline_init_time,
                 "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    offline_time = online_init_time - offline_init_time
    online_time = end_time - online_init_time
    total_time = end_time - offline_init_time
    if not logger.speed_history['offline'] and offline_time > 0:
        logger.speed_history['offline'].append(FLAGS.offline_steps / offline_time)
    if not logger.speed_history['online'] and online_time > 0:
        logger.speed_history['online'].append(FLAGS.online_steps / online_time)
    avg_offline_speed = np.mean(logger.speed_history['offline']) if logger.speed_history['offline'] else 0
    avg_online_speed = np.mean(logger.speed_history['online']) if logger.speed_history['online'] else 0
    all_speeds = logger.speed_history['offline'] + logger.speed_history['online']
    avg_total_speed = np.mean(all_speeds) if all_speeds else 0

    summary = []
    summary.append('=' * 40)
    summary.append('         FINAL SUMMARY')
    summary.append('=' * 40)
    summary.append(f'Environment:    {FLAGS.env_name}')
    summary.append(f'Agent:          {config.get("agent_name", "unknown")}')
    summary.append(f'Seed:           {FLAGS.seed}')
    summary.append(f'\nEvaluation ({FLAGS.eval_episodes} episodes):')
    summary.append(f'  Success Rate:       {eval_info.get("success", 0) * 100:.1f}%')
    summary.append(f'  Mean Return:        {eval_info.get("episode.return", 0):.1f}')
    summary.append(f'\nTraining Speed (iter/s):')
    summary.append(f'  Offline:            {avg_offline_speed:.1f}')
    summary.append(f'  Online:             {avg_online_speed:.1f}')
    summary.append(f'  Overall:            {avg_total_speed:.1f}')
    summary.append(f'\nWall-Clock Time:')
    summary.append(f'  Offline:            {offline_time / 60:.1f} min')
    summary.append(f'  Online:             {online_time / 60:.1f} min')
    summary.append(f'  Total:              {total_time / 60:.1f} min')
    summary.append(f'\nInference (GPU, JIT):')
    summary.append(f'  Latency:            {inference_ms_mean:.2f} +/- {inference_ms_std:.2f} ms/action')
    summary.append('=' * 40)

    summary_text = '\n'.join(summary)
    print('\n' + summary_text)
    with open(os.path.join(FLAGS.save_dir, 'summary.txt'), 'w') as f:
        f.write(summary_text + '\n')

    results = {
        'method': FLAGS.run_group.replace('benchmark_', ''),
        'env': FLAGS.env_name,
        'seed': FLAGS.seed,
        'agent_name': config.get('agent_name', 'unknown'),
        'offline_steps': FLAGS.offline_steps,
        'online_steps': FLAGS.online_steps,
        'horizon_length': FLAGS.horizon_length,
        'offline_time_s': round(offline_time, 2),
        'online_time_s': round(online_time, 2),
        'total_time_s': round(total_time, 2),
        'offline_speed_iter_s': {
            'mean': round(float(avg_offline_speed), 2),
            'std': round(float(np.std(logger.speed_history['offline'])), 2) if logger.speed_history['offline'] else 0.0,
        },
        'online_speed_iter_s': {
            'mean': round(float(avg_online_speed), 2),
            'std': round(float(np.std(logger.speed_history['online'])), 2) if logger.speed_history['online'] else 0.0,
        },
        'eval_history': logger.eval_history,
        'final_success': float(eval_info.get('success', 0)),
        'final_return': float(eval_info.get('episode.return', 0)),
        'wandb_url': run.url,
    }
    results['inference_ms'] = {
        'mean': inference_ms_mean,
        'std': inference_ms_std,
    }

    with open(os.path.join(FLAGS.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

if __name__ == '__main__':
    app.run(main)

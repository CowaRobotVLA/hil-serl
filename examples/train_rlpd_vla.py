#!/usr/bin/env python3
"""
Training script for PI05Agent (VLA+RL).
Based on train_rlpd.py but adapted for Flow Matching VLA agent.
"""

import warnings

warnings.filterwarnings("ignore")

import logging

logging.getLogger("asyncio").setLevel(logging.ERROR)

import glob
import time
import numpy as np
import tqdm
from absl import app, flags

import os
import copy
from typing import Optional
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

import torch
from torch.utils.tensorboard import SummaryWriter

from serl_launcher.serl_launcher_torch.agents.continuous.pi05 import PI05Agent

from serl_launcher.serl_launcher_torch.utils.timer_utils import Timer
from serl_launcher.serl_launcher_torch.utils.train_utils import (
    concat_batches,
    state_dict_to_numpy,
    numpy_to_state_dict,
    print_green,
)

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.serl_launcher_torch.data.data_store import (
    MemoryEfficientReplayBufferDataStore,
)
from serl_launcher.serl_launcher_torch.utils.launcher import (
    make_trainer_config,
    make_pi05_agent,
)

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "cowa_vla_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", True, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string(
    "demo_path",
    None,
    "Path to the demo data.",
)
flags.DEFINE_string("checkpoint_path", "rlpd_vla_ckpt", "Path to save checkpoints.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean("use_classifier", True, "Use classifier to compute reward.")
flags.DEFINE_boolean("debug", False, "Debug mode.")


def actor(agent: PI05Agent, data_store, intvn_data_store, env, device: str = "cuda"):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    agent.eval()
    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        """Update agent parameters from server"""
        state_dict = numpy_to_state_dict(params, device)
        agent.load_state_dict(state_dict, strict=False)

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = {k: torch.as_tensor(v, device=device) for k, v in obs.items()}
                    actions = agent.sample_actions(
                        observations=obs_tensor,
                        argmax=False,
                    )
                actions = actions.cpu().numpy()

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            reward = np.asarray(reward, dtype=np.float32)

            # override the action with the intervention action
            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            running_return += reward

            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )

            if "grasp_penalty" in info:
                transition["grasp_penalty"] = info["grasp_penalty"]
            # All data goes into replay buffer
            data_store.insert(transition)

            transitions.append(copy.deepcopy(transition))
            if already_intervened:
                intvn_data_store.insert(transition)

            obs = next_obs
            if done or truncated:
                if "episode" in info:
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps

                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            # dump to pickle file
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


def learner(
    agent: PI05Agent,
    replay_buffer: MemoryEfficientReplayBufferDataStore,
    demo_buffer: Optional[MemoryEfficientReplayBufferDataStore] = None,
    device: str = "cuda",
):
    agent.train()

    # Create TensorBoard log directory
    log_dir = os.path.join(FLAGS.checkpoint_path, "logs") if FLAGS.checkpoint_path else "./logs"
    os.makedirs(log_dir, exist_ok=True)
    tb_logger = SummaryWriter(log_dir=log_dir)

    step = 0
    global_step = 0  # For noise annealing

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if "environment" in payload and "episode" in payload["environment"]:
            episode_info = payload["environment"]["episode"]
            for key, value in episode_info.items():
                if isinstance(value, (int, float)):
                    tb_logger.add_scalar(f"environment/{key}", value, step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(state_dict_to_numpy(agent.state_dict()))
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    if demo_buffer:
        single_buffer_batch_size = config.batch_size // 2
        demo_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": single_buffer_batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=device,
        )
    else:
        single_buffer_batch_size = config.batch_size
        demo_iterator = None

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": single_buffer_batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=device,
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    pbar = tqdm.tqdm(
        total=config.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    # PI05Agent only has "model" to update
    networks_to_update = frozenset({"model"})

    for step in tqdm.tqdm(range(config.max_steps), dynamic_ncols=True, desc="learner"):
        # Sample from replay buffer
        with timer.context("sample_replay_buffer"):
            batch = next(replay_iterator)
            if demo_iterator:
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

        # Update agent
        with timer.context("train"):
            # Update global_step for noise annealing
            agent.config["global_step"] = global_step
            update_info = agent.update(batch, networks_to_update=networks_to_update)
            global_step += 1

        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            torch.cuda.synchronize()
            with torch.no_grad():
                state_dict = agent.state_dict()
                numpy_params = state_dict_to_numpy(state_dict)
            server.publish_network(numpy_params)
            del state_dict, numpy_params
            torch.cuda.empty_cache()

        if step % config.log_period == 0:
            # Log training info to TensorBoard
            for key, value in update_info.items():
                if isinstance(value, (int, float)):
                    tb_logger.add_scalar(f"train/{key}", value, step)

            # Log timer info to TensorBoard
            timer_stats = timer.get_average_times()
            for key, value in timer_stats.items():
                if isinstance(value, (int, float)):
                    tb_logger.add_scalar(f"timer/{key}", value, step)

        if step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
            checkpoint_file = os.path.join(FLAGS.checkpoint_path, f"checkpoint_{step}.pt")
            with torch.no_grad():
                torch.save(
                    {"step": step, "model_state_dict": agent.state_dict()},
                    checkpoint_file,
                )
            print_green(f"Saved checkpoint to {checkpoint_file}")
            torch.cuda.empty_cache()

        pbar.update(len(replay_buffer) - pbar.n)
        step += 1

    # Close TensorBoard writer
    tb_logger.close()


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_green(f"Using device: {device}")

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=FLAGS.use_classifier and FLAGS.actor,
    )
    env = RecordEpisodeStatistics(env)

    # Create PI05Agent (no sample_obs/sample_action needed - loads from config)
    agent: PI05Agent = make_pi05_agent(
        seed=FLAGS.seed,
    )
    print_green(
        f"PI05Agent created with config: action_chunk={agent.config.get('action_chunk')}, num_steps={agent.config.get('num_steps')}"
    )

    # Move to device
    agent = agent.to(device)

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        checkpoint_files = glob.glob(os.path.join(FLAGS.checkpoint_path, "checkpoint_*.pt"))
        if checkpoint_files:
            input("Checkpoint path already exists. Press Enter to resume training.")
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            ckpt = torch.load(latest_checkpoint, map_location=device)
            agent.load_state_dict(ckpt["model_state_dict"], strict=False)
            print_green(f"Loaded previous checkpoint at step {ckpt['step']} from {latest_checkpoint}.")
        else:
            print_green(f"Checkpoint directory exists but no checkpoint files found.")

    if FLAGS.learner:
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=False,
            device="cpu",
        )

        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=False,
            device="cpu",
        )
        print_green("replay buffer created")

        if FLAGS.demo_path:
            for path in FLAGS.demo_path:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        transitions = pkl.load(f)
                        for transition in transitions:
                            if "infos" in transition and "grasp_penalty" in transition["infos"]:
                                transition["grasp_penalty"] = transition["infos"]["grasp_penalty"]
                            demo_buffer.insert(transition)
                    print_green(f"Loaded demo data from {path}")
                else:
                    print_green(f"Demo path {path} not found, skipping.")
        else:
            print_green("No demo path provided. Creating empty demo buffer.")
            demo_buffer = None

        if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer")):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "demo_buffer")):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}")

        print_green(f"demo buffer size: {len(demo_buffer) if demo_buffer else 0}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        # learner loop
        print_green("starting learner loop")
        learner(agent, replay_buffer=replay_buffer, demo_buffer=demo_buffer, device=device)

    elif FLAGS.actor:
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        demo_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(agent, data_store, demo_data_store, env, device=device)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)

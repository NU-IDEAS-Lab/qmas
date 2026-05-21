from __future__ import annotations

import functools
from typing import Dict, Any, Optional

import numpy as np

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec
from pettingzoo.utils.conversions import turn_based_aec_to_parallel
from gymnasium import spaces
from pettingzoo.classic import hanabi_v5


# ======================================================================
# Argument Helpers
# ======================================================================

def add_args(parser):
    parser.add_argument("--hanabi_colors", type=int, default=5)
    parser.add_argument("--hanabi_ranks", type=int, default=5)
    parser.add_argument("--hanabi_players", type=int, default=2)
    parser.add_argument("--hanabi_hand_size", type=int, default=5)
    parser.add_argument("--hanabi_max_information_tokens", type=int, default=8)
    parser.add_argument("--hanabi_max_life_tokens", type=int, default=3)
    parser.add_argument("--hanabi_observation_type", type=str,
        choices=["minimal", "card_knowledge", "seer"],
        default="seer")
    parser.add_argument("--hanabi_random_start_player", action="store_true")
    parser.add_argument("--hanabi_log_file", type=str, default="")
    parser.add_argument("--render_mode", type=str, default="none",
        choices=parallel_env.metadata["render_modes"])


def validate_args(parsed_args):
    parsed_args.num_agents = parsed_args.hanabi_players


def env(*args, **kwargs):
    return parallel_env(*args, **kwargs)


def raw_env(*args, **kwargs):
    return parallel_to_aec(parallel_env(*args, **kwargs))


# ======================================================================
#                   MINIMAL OBSERVATION VERSION
# ======================================================================

class parallel_env(ParallelEnv):

    metadata = {
        "name": "hanabi_v5_minimal_obs",
        "render_modes": ["human", "ansi", "none"],
    }

    def __init__(
        self,
        colors=5,
        ranks=5,
        players=2,
        hand_size=5,
        max_information_tokens=8,
        max_life_tokens=3,
        hanabi_observation_type="seer",
        random_start_player=False,
        hanabi_log_file="",
        max_cycles=-1,
        render_mode="none"
    ):
        super().__init__()

        self.players = players
        self.hand_size = hand_size
        self.max_information_tokens = max_information_tokens
        self.max_life_tokens = max_life_tokens
        self.colors = colors
        self.ranks = ranks
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.log_file=hanabi_log_file

        # ---- base env with MINIMAL observation ----
        base = hanabi_v5.env(
            colors=colors,
            ranks=ranks,
            players=players,
            hand_size=hand_size,
            max_information_tokens=max_information_tokens,
            max_life_tokens=max_life_tokens,
            observation_type= hanabi_observation_type,
            random_start_player=random_start_player,
            render_mode=render_mode if render_mode != "none" else None,
        )

        # Convert AEC → Parallel
        self._base_env: ParallelEnv = turn_based_aec_to_parallel(base)

        self.possible_agents = base.possible_agents[:]
        self.agents = []
        self.step_count = 0
        self._last_obs: Optional[Dict[str, Any]] = None
        self.dones = {a: False for a in self.possible_agents}
        
        if self.log_file != "":
            with open(self.log_file, 'w') as f:
                f.write("Final_Score,Game_Lost,Total_Steps\n")
        # ============================
        # Determine OBS_SIZE SAFELY
        # ============================
        any_agent = self.possible_agents[0]
        obs_space = self._base_env.observation_space(any_agent)["observation"]
        self.OBS_SIZE = obs_space.shape[0]

        # ============================
        # Compute observation encoding offsets
        # ============================
        bits_per_card = colors * ranks
        max_deck_size = sum([3, 2, 2, 2, 1]) * colors

        hands_section = (players - 1) * hand_size * bits_per_card + players
        board_section = (max_deck_size - players * hand_size) + \
                        (colors * ranks) + max_information_tokens + max_life_tokens
        discard_section = max_deck_size
        last_action_section = players + 4 + players + colors + ranks + \
                              hand_size + hand_size + bits_per_card + 2

        self._observation_type = hanabi_observation_type
        self._card_knowledge_start = hands_section + board_section + \
                                     discard_section + last_action_section
        self._bits_per_card = bits_per_card
        self._bits_per_knowledge = bits_per_card + colors + ranks

        self._firework_start = hands_section + (max_deck_size - players * hand_size)
        self._life_token_start = self._firework_start + colors * ranks + max_information_tokens

        # ---- Gym spaces ----
        self.observation_spaces = {
            agent: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.OBS_SIZE,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: self._base_env.action_space(agent)
            for agent in self.possible_agents
        }

    # ----------------------------------------------------------------------
    # Required API
    # ----------------------------------------------------------------------

    @functools.cache
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.cache
    def action_space(self, agent):
        return self.action_spaces[agent]

    @property
    @functools.cache
    def state_space(self):
        # MINIMAL MODE: state == observation
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.OBS_SIZE,),
            dtype=np.float32
        )

    # ----------------------------------------------------------------------
    # Visibility mask: hide own card identity, keep hints visible
    # ----------------------------------------------------------------------

    def _compute_visibility_mask(self, agent):
        obs = self._last_obs[agent]["observation"]
        mask = np.ones_like(obs, dtype=bool)

        if self._observation_type != "seer":
            return mask

        ck_start = self._card_knowledge_start
        for card_idx in range(self.hand_size):
            card_offset = ck_start + card_idx * self._bits_per_knowledge
            mask[card_offset : card_offset + self._bits_per_card] = False

        return mask

    # ----------------------------------------------------------------------
    # Strip action mask
    # ----------------------------------------------------------------------

    def _strip_action_mask(self, obs_dict):
        return {
            a: full["observation"].astype(np.float32)
            for a, full in obs_dict.items()
        }

    # ----------------------------------------------------------------------
    # reset()
    # ----------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self.step_count = 0

        obs, _ = self._base_env.reset(seed=seed, options=options)
        self._last_obs = obs
        self.agents = list(obs.keys())
        self.dones = {a: False for a in self.possible_agents}
        self.score = {a: 0 for a in self.possible_agents}
        
        clean = self._strip_action_mask(obs)
        
        info = {
            a: {"visibility_mask": self._compute_visibility_mask(a)}
            for a in self.possible_agents
        }

        return clean, info

    # ----------------------------------------------------------------------
    # state(agent) = its own observation
    # ----------------------------------------------------------------------

    def state(self):
        return self._last_obs[self.possible_agents[0]]["observation"].astype(np.float32)

    # ----------------------------------------------------------------------
    # available_actions
    # ----------------------------------------------------------------------

    def available_actions(self, agent):
        return self._last_obs[agent]["action_mask"].astype(np.int8)

    @functools.cache
    def available_actions_space(self, agent):
        n = self.action_space(agent).n
        return spaces.MultiBinary(n)

    # ----------------------------------------------------------------------
    # observe(agent)
    # ----------------------------------------------------------------------

    def observe(self, agent):
        obs = self._last_obs[agent]["observation"].astype(np.float32)
        return obs, self._compute_visibility_mask(agent)

    # ----------------------------------------------------------------------
    # step()
    # ----------------------------------------------------------------------

    def step(self, action_dict, lastStep=False):
       
        if action_dict is None:
            action_dict = {}

        self.step_count += 1

        # FIX: Convert flattened actions back to integers for base env
        converted_actions = {}
        for agent, action in action_dict.items():
            if isinstance(action, np.ndarray):
                # Extract the integer from the array
                converted_actions[agent] = int(action[0])
            else:
                converted_actions[agent] = action

        obs, reward_dict, term, trunc, _ = self._base_env.step(converted_actions)
        self._last_obs = obs
        self.agents = list(obs.keys())

        for a in self.possible_agents:
            if term.get(a, False):
                self.dones[a] = True
            
            # Increment score.
            self.score[a] += reward_dict.get(a, 0)

        force_trunc = lastStep or (
            self.max_cycles >= 0 and self.step_count >= self.max_cycles
        )

        done_dict = self.dones
        trunc_dict = {a: bool(trunc.get(a, False)) for a in self.possible_agents}

        if force_trunc:
            for a in self.possible_agents:
                trunc_dict[a] = True

        episode_end = force_trunc or all(
            done_dict[a] or trunc_dict[a] for a in self.possible_agents
        )

        clean = self._strip_action_mask(obs)

        # Compute average score.
        score_np = np.array([self.score[a] for a in self.possible_agents])
        score_avg = np.mean(score_np)

        assert np.all(score_np == self._base_env.unwrapped.hanabi_env.game_state.returns()) , "Internal score representation mismatch."
        assert np.all(score_np == score_avg) , "Scores across agents should be identical in Hanabi."
        assert np.all(score_np >= 0) , "Scores should be non-negative in Hanabi."
        assert np.all(score_np <= self.colors * self.ranks) , "Scores should not exceed maximum possible in Hanabi."

        game_lost = self._is_game_lost()

        if episode_end:
            if self.log_file != "":
                with open(self.log_file, 'a') as f:
                    f.write(f"{score_avg},{game_lost},{self.step_count}\n")
            self.agents = []

        info = {}

        mask = self._compute_visibility_mask(self.possible_agents[0])
        info["state_visibility_mask"] = mask
        info["score"] = score_avg
        info["game_lost"] = game_lost
        info["steps_to_finish"] = self.step_count

        return clean, reward_dict, done_dict, trunc_dict, info

    # ----------------------------------------------------------------------
    # Check if game is lost (all life tokens depleted)
    # ----------------------------------------------------------------------

    def _is_game_lost(self):
        obs0 = self._last_obs[self.possible_agents[0]]["observation"]
        life_tokens = obs0[self._life_token_start:self._life_token_start + self.max_life_tokens]
        return life_tokens.sum() == 0

    # ----------------------------------------------------------------------
    # render() / close()
    # ----------------------------------------------------------------------

    def render(self, *args, **kwargs):
        return self._base_env.render(*args, **kwargs)

    def close(self):
        self._base_env.close()
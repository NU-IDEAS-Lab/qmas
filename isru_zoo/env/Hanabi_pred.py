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
        self.log_file="hanabi_results1.txt"

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
        self.count = 0
        self.total = 0
        self.step_count = 0
        self._last_obs: Optional[Dict[str, Any]] = None
        self.dones = {a: False for a in self.possible_agents}
        
        # Track previous score for reward calculation
        self._prev_score = 0
        self._cumulative_reward = 0
        with open(self.log_file, 'w') as f:
             f.write("Episode,Final_Score,Game_Lost,Total_Steps\n")
        # ============================
        # Determine OBS_SIZE SAFELY
        # ============================
        any_agent = self.possible_agents[0]
        obs_space = self._base_env.observation_space(any_agent)["observation"]
        self.OBS_SIZE = obs_space.shape[0]

        # ---- Gym spaces ----
        self.observation_spaces = {
            agent: spaces.Box(
                low=0.0,
                high=1.0,
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
    # Visibility mask = all ones
    # ----------------------------------------------------------------------

    def _compute_visibility_mask(self, agent):
        obs = self._last_obs[agent]["observation"]
        mask = np.ones_like(obs, dtype=bool)

        # Own cards' true identity bits are firs

        # Mask index 308-657 as False
        mask[308:658] = False

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
        self._prev_score = 0
        self._cumulative_reward = 0

        obs, _ = self._base_env.reset(seed=seed, options=options)
        self._last_obs = obs
        self.agents = list(obs.keys())
        self.dones = {a: False for a in self.possible_agents}
        
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
        return self.state(), self._compute_visibility_mask(agent)

    # ----------------------------------------------------------------------
    # compute firework score (sum of highest rank in each color's firework)
    # ----------------------------------------------------------------------

    def _compute_firework_score(self):
        """
        Compute the current firework score.
        The score is the sum of values in each constructed firework.
        For example, if Blue has cards 1,2 played, Red has 1, Green has 1,2,3:
        Score = 2 + 1 + 3 = 6
        """
        obs0 = self._last_obs[self.possible_agents[0]]["observation"]
        # Firework encoding starts at index 175 for default settings
        # Each color has 'ranks' bits indicating which ranks have been played
        base = 175
        score = 0
        for i in range(self.colors):
            seg = obs0[base + self.ranks * i : base + self.ranks * i + self.ranks]
            if seg.sum() > 0:
                # The highest rank played is the firework value for this color
                highest = np.where(seg == 1)[0].max() + 1
                score += highest
        return score

    # ----------------------------------------------------------------------
    # check if game is lost (all life tokens depleted)
    # ----------------------------------------------------------------------

    def _is_game_lost(self):
        """
        Check if the game is lost by examining life tokens.
        Game is lost when all life tokens are depleted.
        """
        obs0 = self._last_obs[self.possible_agents[0]]["observation"]
        # Life tokens are encoded after information tokens in the observation
        # For default settings: info tokens at 200-207, life tokens at 208-210
        # Calculate positions based on settings
        life_token_start = 175 + (self.colors * self.ranks) + self.max_information_tokens
        life_token_end = life_token_start + self.max_life_tokens
        life_tokens = obs0[life_token_start:life_token_end]
        
        # If all life token bits are 0, game is lost
        return life_tokens.sum() == 0

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

        obs, rew, term, trunc, _ = self._base_env.step(converted_actions)
        self._last_obs = obs
        self.agents = list(obs.keys())

        for a in self.possible_agents:
            if term.get(a, False):
                self.dones[a] = True

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

        # Calculate reward as change in firework score
        current_score = self._compute_firework_score()
        step_reward = current_score - self._prev_score
        self._cumulative_reward += step_reward
        self._prev_score = current_score

        # Check if game is lost
        game_lost = self._is_game_lost()
        
        final_score = current_score
        final_reward = step_reward
        
        if episode_end:
            if game_lost:
                # If game is lost, final score is 0
                # Final reward is negation of all rewards received so far
                final_score = 0
                final_reward = -self._cumulative_reward + step_reward  # Negate previous rewards
            
            
            self.total += final_score
            # print(f"[Episode End Final Score = {final_score}, Game Lost = {game_lost}")
            # print(f"[Steps to finish the game] Total Steps = {self.step_count}")
            with open(self.log_file, 'a') as f:
                f.write(f"{self.count},{final_score},{game_lost},{self.step_count}\n")
            self.agents = []

        # All agents get the same reward (cooperative game)
        reward_dict = {a: float(final_reward) for a in self.possible_agents}

        info = {a: [reward_dict[a]] for a in self.possible_agents}
        
        mask = self._compute_visibility_mask(self.possible_agents[0])
        info["state_visibility_mask"] = mask
        info["score"] = final_score
        info["num of steps"] = self.step_count

        return clean, reward_dict, done_dict, trunc_dict, info

    # ----------------------------------------------------------------------
    # render() / close()
    # ----------------------------------------------------------------------

    def render(self, *args, **kwargs):
        return self._base_env.render(*args, **kwargs)

    def close(self):
        self._base_env.close()
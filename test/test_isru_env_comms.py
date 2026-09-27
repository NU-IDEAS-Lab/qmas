import unittest

import numpy as np
from gymnasium import spaces

from isru_zoo.env.isru_env import parallel_env
from onpolicy.envs.pettingzoo.Pettingzoo_Env import PettingzooEnv


def make_env(num_agents=2, communication_mode="broadcast", **kwargs):
    return parallel_env(
        num_extractors=num_agents,
        num_haulers=0,
        num_prospectors=0,
        num_superbots=0,
        world_size=10,
        num_obstacles=0,
        num_resources=2,
        communication_mode=communication_mode,
        **kwargs,
    )


def make_action(environment, agent, request, relative_position=(0.0, 0.0)):
    ''' Builds a flattened action for `agent`: stationary movement (Moore index 4),
    plus a communication request bit (and, for "nearest" mode, a relative position). '''
    action_space = environment.action_space(agent)
    action = {"movement": np.array([4], dtype=np.int32)}
    if "communication" in action_space.spaces:
        comm = {"request": np.array([1 if request else 0], dtype=np.int32)}
        if "relative_position" in action_space.spaces["communication"].spaces:
            comm["relative_position"] = np.array(relative_position, dtype=np.float32)
        action["communication"] = comm
    return spaces.flatten(action_space, action)


class TestReceivedComms(unittest.TestCase):
    ''' Verifies that isru_env.py's `received_comms` info flag (consumed by
    onpolicy's `_apply_ex_env_comms_merge` to actually merge/share predicted state
    between agents, mirroring patrolling_zoo.py) is populated correctly. '''

    def test_reset_received_comms_false(self):
        environment = make_env()
        _, info = environment.reset(seed=42)
        for agent in environment.agents:
            self.assertIs(info[agent]["received_comms"], False)

    def test_get_reset_info_received_comms_false(self):
        environment = make_env()
        environment.reset(seed=42)
        info = environment.get_reset_info()
        for agent in environment.agents:
            self.assertIs(info[agent]["received_comms"], False)

    def test_broadcast_received_comms_tracks_request_bit(self):
        environment = make_env(num_agents=2, communication_mode="broadcast")
        environment.reset(seed=42)
        agent0, agent1 = environment.agents

        action_dict = {
            agent0: make_action(environment, agent0, request=True),
            agent1: make_action(environment, agent1, request=False),
        }
        _, _, _, _, info = environment.step(action_dict)

        # agent0 requested and has another agent to hear from -> received_comms True.
        self.assertIs(info[agent0]["received_comms"], True)
        # agent1 didn't request -> falls into the no_communication branch, never
        # enters comms_requests_explicit/relative -> received_comms False.
        self.assertIs(info[agent1]["received_comms"], False)

    def test_broadcast_no_request_received_comms_false(self):
        environment = make_env(num_agents=2, communication_mode="broadcast")
        environment.reset(seed=42)
        agent0, agent1 = environment.agents

        action_dict = {
            agent0: make_action(environment, agent0, request=False),
            agent1: make_action(environment, agent1, request=False),
        }
        _, _, _, _, info = environment.step(action_dict)

        for agent in environment.agents:
            self.assertIs(info[agent]["received_comms"], False)

    def test_nearest_single_agent_request_never_satisfied(self):
        ''' With only one agent, there is no one else to communicate with. The
        `len(self.possible_agents) > 1` guard in step() means the request never
        even enters comms_requests_relative, so received_comms must be False even
        though the agent's request bit was set -- this is the "a request was made
        but nothing was actually received" case that received_comms must not
        conflate with "a request succeeded". '''
        environment = make_env(num_agents=1, communication_mode="nearest")
        environment.reset(seed=42)
        agent = environment.agents[0]

        action_dict = {agent: make_action(environment, agent, request=True)}
        _, _, _, _, info = environment.step(action_dict)

        self.assertIs(info[agent]["received_comms"], False)

    def test_info_wrapper_stacks_received_comms(self):
        ''' Exercises Pettingzoo_Env.py's `_info_wrapper`, which is what actually
        surfaces `received_comms` to onpolicy's runner as a top-level (n_agents,)
        array for `_apply_ex_env_comms_merge` to consume. '''
        environment = make_env(num_agents=2, communication_mode="broadcast")
        environment.reset(seed=42)
        agent0, agent1 = environment.agents

        action_dict = {
            agent0: make_action(environment, agent0, request=True),
            agent1: make_action(environment, agent1, request=False),
        }
        _, _, _, _, info = environment.step(action_dict)

        wrapper = object.__new__(PettingzooEnv)
        wrapper.env = environment
        # This env's observation_space/state_space are plain Dicts (no Graph leaves),
        # so the real PettingzooEnv.__init__ would set both of these True.
        wrapper.flatten_observations = True
        wrapper.flatten_observations_global = True
        wrapper._info_wrapper(info)

        self.assertIn("received_comms", info)
        expected = np.array(
            [info[a]["received_comms"] for a in environment.possible_agents],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(info["received_comms"], expected)
        np.testing.assert_array_equal(info["received_comms"], np.array([1.0, 0.0], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
from gymnasium import spaces

from isru_zoo.env.isru_env import parallel_env, parallel_env_graph_obs, parallel_env_partial_obs


def make_env(cls=parallel_env_graph_obs, **kwargs):
    defaults = dict(
        num_extractors=1,
        num_haulers=1,
        num_prospectors=1,
        num_superbots=0,
        world_size=20,
        num_obstacles=0,
        num_resources=2,
        communication_mode="full",
        movement_mode="velocity",
    )
    defaults.update(kwargs)
    return cls(**defaults)


def step_random(environment):
    action_dict = {
        agent: spaces.flatten(environment.action_space(agent), environment.action_space(agent).sample())
        for agent in environment.agents
    }
    return environment.step(action_dict)


class TestRandomObservationRadius(unittest.TestCase):
    ''' Verifies the randomized observation radius, which mirrors patrolling_zoo.py:
    one radius per episode, shared by all agents, drawn uniformly from
    [observation_radius_random_min, observation_radius_random_max]. '''

    def test_disabled_by_default(self):
        environment = make_env(cls=parallel_env, observation_radius=7)
        for seed in range(5):
            environment.reset(seed=seed)
            for agent in environment.possible_agents:
                self.assertEqual(agent.observation_radius, 7)

    def test_radius_shared_and_in_range(self):
        environment = make_env(observation_radius_random_min=0.0, observation_radius_random_max=15.0)
        radii = []
        for seed in range(50):
            environment.reset(seed=seed)
            episode_radii = {agent.observation_radius for agent in environment.possible_agents}
            self.assertEqual(len(episode_radii), 1)
            radius = episode_radii.pop()
            self.assertGreaterEqual(radius, 0.0)
            self.assertLessEqual(radius, 15.0)
            radii.append(radius)
        self.assertGreater(len(set(radii)), 1)

    def test_radius_fixed_within_episode(self):
        environment = make_env(observation_radius_random_min=0.0, observation_radius_random_max=15.0)
        environment.reset(seed=3)
        radius = environment.possible_agents[0].observation_radius
        for _ in range(10):
            step_random(environment)
            for agent in environment.possible_agents:
                self.assertEqual(agent.observation_radius, radius)

    def test_map_unchanged_by_randomization(self):
        ''' The radius is drawn after map generation, so a seeded map is the same
        whether or not randomization is enabled. '''
        fixed = make_env()
        randomized = make_env(observation_radius_random_min=0.0, observation_radius_random_max=15.0)
        fixed.reset(seed=11)
        randomized.reset(seed=11)
        np.testing.assert_array_equal(fixed.map_obstacles, randomized.map_obstacles)
        for a, b in zip(fixed.possible_agents, randomized.possible_agents):
            np.testing.assert_array_equal(a.position, b.position)

    def test_observation_space_unchanged_when_disabled(self):
        for dense in (False, True):
            environment = make_env(gnn_use_dense_obs=dense)
            agent = environment.possible_agents[0]
            self.assertNotIn("observation_radius", environment.observation_space(agent).spaces)

    def test_dense_obs_reports_radius(self):
        environment = make_env(gnn_use_dense_obs=True,
                               observation_radius_random_min=0.0, observation_radius_random_max=15.0)
        observation, info = environment.reset(seed=5)
        _, _, _, _, step_info = step_random(environment)
        for agent in environment.possible_agents:
            space = environment.observation_space(agent)
            self.assertIn("observation_radius", space.spaces)
            self.assertTrue(space.contains(observation[agent]))
            self.assertAlmostEqual(float(observation[agent]["observation_radius"][0]), agent.observation_radius, places=5)
            np.testing.assert_array_equal(info[agent]["visibility_mask"]["observation_radius"], [1.0])
            # The global (infinite-radius) view still reports the agent's actual radius.
            global_obs, _ = environment.get_observation_and_comms_situated(agent)
            self.assertAlmostEqual(float(global_obs["observation_radius"][0]), agent.observation_radius, places=5)
        state, _ = environment._state()
        self.assertTrue(environment.state_space.contains(state))

    def test_sparse_obs_reports_radius(self):
        environment = make_env(gnn_use_dense_obs=False,
                               observation_radius_random_min=0.0, observation_radius_random_max=15.0)
        observation, info = environment.reset(seed=5)
        for agent in environment.possible_agents:
            self.assertAlmostEqual(float(observation[agent]["observation_radius"][0]), agent.observation_radius, places=5)
            self.assertEqual(len(info[agent]["visibility_mask"]), len(observation[agent]))
            self.assertTrue(info[agent]["visibility_mask"].all())

    def test_radius_controls_visibility(self):
        ''' The drawn radius is what actually limits each agent's view. '''
        environment = make_env(gnn_use_dense_obs=True,
                               observation_radius_random_min=0.0, observation_radius_random_max=15.0)
        environment.reset(seed=7)
        for agent in environment.possible_agents:
            expected = environment._get_observation_radius_mask(agent.grid_position, agent.observation_radius)
            np.testing.assert_array_equal(agent.mask_observed, expected)

    def test_partial_obs_rejects_randomization(self):
        with self.assertRaises(ValueError):
            make_env(cls=parallel_env_partial_obs,
                     observation_radius_random_min=0.0, observation_radius_random_max=15.0)

    def test_sparse_obs_filters_by_radius(self):
        ''' With no comms, the sparse graph obs only contains the agents within the drawn radius. '''
        environment = make_env(gnn_use_dense_obs=False, communication_mode="bernoulli",
                               communication_probability=0.0, num_resources=0,
                               observation_radius_random_min=0.0, observation_radius_random_max=15.0)
        for seed in range(20):
            observation, _ = environment.reset(seed=seed)
            for agent in environment.possible_agents:
                graph = observation[agent]["graph"]
                role_dim = int(max(environment.NODE_TYPE)) + 1
                roles = graph.x[:, :role_dim].argmax(dim=1).tolist()
                num_agent_nodes = sum(r < int(environment.NODE_TYPE.RESOURCE) for r in roles)  # Agent roles precede entity types.
                expected = sum(
                    np.linalg.norm(other.position - agent.position) <= agent.observation_radius
                    for other in environment.possible_agents
                )
                self.assertEqual(num_agent_nodes, expected)


if __name__ == "__main__":
    unittest.main()

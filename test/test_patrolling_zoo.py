import os
import unittest
from pettingzoo.test import parallel_api_test
from patrolling_zoo.patrolling_zoo_v0 import PatrolGraph, env
import patrolling_zoo.graphs

class TestEnvironment(unittest.TestCase):

    def test_parallel_api(self):
        environment = env()
        parallel_api_test(environment, num_cycles=10)
    
    def test_path_length_from_node0(self):
        environment = env(
            graph_file = os.path.join(os.path.dirname(patrolling_zoo.graphs.__file__), "4nodes.graph"),
            num_agents=1
        )
        agent = environment.agents[0]
        agent.reset()
        agent.position = environment.pg.getNodePosition(0)
        agent.lastNode = 0

        path = environment._getPathToNode(agent, 0)
        pathLen = environment._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 0.0)

        path = environment._getPathToNode(agent, 1)
        pathLen = environment._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 40.0)

        path = environment._getPathToNode(agent, 2)
        pathLen = environment._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 25.0)

        path = environment._getPathToNode(agent, 3)
        pathLen = environment._getAgentPathLength(agent, path)
        self.assertAlmostEqual(pathLen, 47.2, places=1)

    def test_path_length_from_node1(self):
        environment = env(
            graph_file = os.path.join(os.path.dirname(patrolling_zoo.graphs.__file__), "4nodes.graph"),
            num_agents=1
        )
        agent = environment.agents[0]
        agent.reset()
        agent.position = environment.pg.getNodePosition(1)
        agent.lastNode = 1

        path = environment._getPathToNode(agent, 0)
        pathLen = environment._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 40.0)

        path = environment._getPathToNode(agent, 1)
        pathLen = environment._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 0.0)

        path = environment._getPathToNode(agent, 2)
        pathLen = environment._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 65.0)

        path = environment._getPathToNode(agent, 3)
        pathLen = environment._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 25.0)
    
    def test_path_length_from_edge13(self):
        environment = env(
            graph_file = os.path.join(os.path.dirname(patrolling_zoo.graphs.__file__), "4nodes.graph"),
            num_agents=1
        )
        agent = environment.agents[0]
        agent.reset()
        pos = environment.pg.getNodePosition(1)
        agent.position = (pos[0], pos[1] + 5.0)
        agent.edge = (1, 3)
        agent.lastNode = 1

        path = environment._getPathToNode(agent, 0)
        pathLen = environment._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 45.0)

        path = environment._getPathToNode(agent, 1)
        pathLen = environment._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 5.0)

        path = environment._getPathToNode(agent, 2)
        pathLen = environment._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 60.0)

        path = environment._getPathToNode(agent, 3)
        pathLen = environment._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 20.0)
    
    def test_state(self):
        environment = env(
            graph_file = os.path.join(os.path.dirname(patrolling_zoo.graphs.__file__), "4nodes.graph"),
            num_agents=1
        )
        agent = environment.agents[0]
        environment.reset(seed=42)
        pos = environment.pg.getNodePosition(1)
        agent.position = (pos[0], pos[1] + 5.0)
        agent.edge = (1, 3)
        agent.lastNode = 1

        state0 = environment.state()[agent]
        self.assertEqual(state0["vertex_state"][0], 1.0)
        self.assertEqual(state0["vertex_state"][1], 1.0)
        self.assertEqual(state0["vertex_state"][2], 1.0)
        self.assertEqual(state0["vertex_state"][3], 1.0)
        self.assertAlmostEqual(state0["vertex_distances"][agent][0], 0.954, places=3)
        self.assertAlmostEqual(state0["vertex_distances"][agent][1], 0.106, places=3)
        self.assertAlmostEqual(state0["vertex_distances"][agent][2], 1.272, places=3)
        self.assertAlmostEqual(state0["vertex_distances"][agent][3], 0.424, places=3)

        self.assertEqual(len(state0), 3)
        self.assertEqual(len(state0["vertex_state"]), 4)
        self.assertEqual(len(state0["vertex_distances"]), 1)
        self.assertEqual(len(state0["vertex_distances"][agent]), 4)
    
    def test_movement_node0_node1(self):
        environment = env(
            graph_file = os.path.join(os.path.dirname(patrolling_zoo.graphs.__file__), "4nodes.graph"),
            num_agents=1
        )
        agent = environment.agents[0]
        environment.reset(seed=42)
        pos = environment.pg.getNodePosition(0)
        agent.position = pos
        agent.edge = None
        agent.lastNode = 0

        # Move agent one step towards node 1.
        reached, stepSize = environment._moveTowardsNode(agent, 1, 1.0)
        self.assertEqual(agent.position, (pos[0] + 1.0, pos[1]))
        self.assertEqual(agent.edge, (0, 1))
        self.assertEqual(agent.lastNode, 0)
        self.assertEqual(stepSize, 0.0)
        self.assertEqual(reached, False)

        # Move agent one step towards node 1.
        reached, stepSize = environment._moveTowardsNode(agent, 1, 1.0)
        self.assertEqual(agent.position, (pos[0] + 2.0, pos[1]))
        self.assertEqual(agent.edge, (0, 1))
        self.assertEqual(agent.lastNode, 0)
        self.assertEqual(stepSize, 0.0)
        self.assertEqual(reached, False)

        # Move agent one step towards node 0.
        reached, stepSize = environment._moveTowardsNode(agent, 0, 1.0)
        self.assertEqual(agent.position, (pos[0] + 1.0, pos[1]))
        self.assertEqual(agent.edge, (0, 1))
        self.assertEqual(agent.lastNode, 0)
        self.assertEqual(stepSize, 0.0)
        self.assertEqual(reached, False)

        # Move agent one step towards node 1.
        reached, stepSize = environment._moveTowardsNode(agent, 1, 1.0)
        self.assertEqual(agent.position, (pos[0] + 2.0, pos[1]))
        self.assertEqual(agent.edge, (0, 1))
        self.assertEqual(agent.lastNode, 0)
        self.assertEqual(stepSize, 0.0)
        self.assertEqual(reached, False)

        # Move agent 40 steps towards node 1.
        reached, stepSize = environment._moveTowardsNode(agent, 1, 40.0)
        self.assertEqual(agent.position, environment.pg.getNodePosition(1))
        self.assertEqual(agent.edge, None)
        self.assertEqual(agent.lastNode, 1)
        self.assertEqual(stepSize, 2.0)
        self.assertEqual(reached, True)


if __name__ == '__main__':
    unittest.main()
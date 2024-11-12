import unittest
from pettingzoo.test import parallel_api_test
from patrolling_zoo.patrolling_zoo_v0 import env
from qmas.communication.communication_wrapper import CommunicationWrapper
from qmas.communication.model.model_boolean_observation import BooleanObservationModel

class TestEnvironment(unittest.TestCase):

    def test_parallel_api(self):
        comms_model = BooleanObservationModel()
        environment = env()
        environment = CommunicationWrapper(environment, comms_model)
        parallel_api_test(environment, num_cycles=10)
    

if __name__ == '__main__':
    unittest.main()
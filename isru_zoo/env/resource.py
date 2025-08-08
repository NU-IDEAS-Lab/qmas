class Resource:
    ''' This class represents a quantity of certain type of resource. '''
    def __init__(self, quantity):
        self.quantity = quantity


class TestResource1(Resource):
    ''' We are using this as the test resource during development. '''
    
    reward_deposit = 5.0


class TestResource2(Resource):
    ''' We are using this as the test resource during development. '''

    reward_deposit = 7.0
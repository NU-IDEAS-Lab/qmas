class Resource:
    ''' This class represents a quantity of certain type of resource. '''

    resource_count = 0

    def __init__(self, quantity, resource_id=None):
        self.quantity = quantity
        
        if resource_id == None:
            self.resource_id = Resource.resource_count
            Resource.resource_count += 1
        else:
            self.resource_id = resource_id


class TestResource1(Resource):
    ''' We are using this as the test resource during development. '''
    
    reward_deposit = 1.5


class TestResource2(Resource):
    ''' We are using this as the test resource during development. '''

    reward_deposit = 3
class Resource:
    ''' This class represents a quantity of certain type of resource. '''

    resource_count = 0

    def __init__(self, quantity_max, resource_id=None):
        self.quantity_max = quantity_max
        self.quantity = quantity_max
        
        if resource_id == None:
            self.resource_id = Resource.resource_count
            Resource.resource_count += 1
        else:
            self.resource_id = resource_id
        
        assert self.resource_id > 0, "Resource ID must be greater than 0."
    
    def reset(self, quantity=None):
        ''' Resets the resource to its initial state. '''
        if quantity is not None:
            self.quantity = quantity

    def __repr__(self):
        return f"Resource(id={self.resource_id}, quantity={self.quantity})"

class TestResource1(Resource):
    ''' We are using this as the test resource during development. '''
    
    reward_deposit = 1.5
    reward_extraction = 0.1


class TestResource2(Resource):
    ''' We are using this as the test resource during development. '''

    reward_deposit = 3
    reward_extraction = 0.5
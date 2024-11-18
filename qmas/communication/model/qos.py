import random

class LinkQoSBase:
    def __init__(self, sender, receiver):
        self.sender = sender
        self.receiver = receiver

    def can_communicate(self):
        raise NotImplementedError("getQoS not implemented")

    def mangle_message(self, message):
        return message
    
    def on_step(self):
        pass


class LinkQoSBernoulli(LinkQoSBase):
    def __init__(self, sender, receiver, p):
        super().__init__(sender, receiver)
        self.p = p

    def can_communicate(self):
        return random.random() <= self.p
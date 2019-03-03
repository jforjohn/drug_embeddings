class Pair(object):

    def __init__(self, *, entity_1, entity_2, type, id=None):
        self.id = id
        self.entity_1 = entity_1
        self.entity_2 = entity_2
        self.type = type

    def __repr__(self):
        return f'<Pair {self.id} {self.entity_1} {self.entity_2} {self.type}>'


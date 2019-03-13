class DrugPair(object):

    def __init__(self, *, entity_1, entity_2, dp_type, dp_id=None):
        self.id = dp_id
        self.entity_1 = entity_1
        self.entity_2 = entity_2
        self.type = dp_type

    def __repr__(self):
        return f'<DrugPair {self.id} {self.entity_1} {self.entity_2} {self.type}>'


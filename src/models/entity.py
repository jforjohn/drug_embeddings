class Entity(object):

    def __init__(self, *, char_offset, type, text, id=None):
        self.id = id
        self.char_offset = char_offset
        self.type = type
        self.text = text
    
    def to_output(self):
        return f'{self.char_offset}|{self.text}|{self.type}'

    def __repr__(self):
        return f'<Pair {self.id} {self.char_offset} {self.text} {self.type}>'
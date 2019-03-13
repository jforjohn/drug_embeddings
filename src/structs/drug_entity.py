class DrugEntity(object):

    def __init__(self, *, offsets, de_type, text, de_id=None):
        self.id = de_id
        self.offsets = offsets
        self.type = de_type
        self.text = text

    def _offset_text(self):
        # for discontinuous entities, we only get the first span
        # (will not work, but there are few of them)
        '''
        return ';'.join(
            [
                '-'.join([str(o) for o in offset])
                for offset in self.offsets
            ]
        )
        '''
        return '-'.join([str(o) for o in self.offsets])
    
    def to_output(self):
        return f'{self._offset_text()}|{self.text}|{self.type}'

    def __repr__(self):
        return f'<DrugEntity {self.id} {self._offset_text()} {self.text} {self.type}>'
from os import listdir
from xml.dom.minidom import parse

import pandas as pd

from structs import DrugEntity
from structs import DrugPair


class Parser(object):
    def __init__(self, base_path):
        self.base_path = base_path

    def call(self):
        sentences = []
        for f in listdir(self.base_path):
            tree = parse(self.base_path + '/' + f)

            for s in tree.getElementsByTagName('sentence'):
                sid = s.attributes['id'].value
                text = s.attributes['text'].value

                entities = []
                for e in s.getElementsByTagName('entity'):
                    # for discontinuous entities, we only get the first span
                    # (will not work, but there are few of them)
                    offsets = [
                        [int(index) for index in offset.split('-')]
                        for offset in e.attributes['charOffset'].value.split(';')
                    ][0]
                    entities.append(DrugEntity(
                        de_id=e.attributes['id'].value,
                        offsets=offsets,
                        de_type=e.attributes['type'].value,
                        text=e.attributes['text'].value
                    ))

                pairs = []
                for p in s.getElementsByTagName('pair'):
                    pairs.append(DrugPair(
                        dp_id=p.attributes['id'].value,
                        entity_1=p.attributes['e1'].value,
                        entity_2=p.attributes['e2'].value,
                        dp_type=p.attributes['type'].value
                        if 'type' in p.attributes else None,
                    ))
                sentences.append([sid, text, entities, pairs])

        return pd.DataFrame(
            sentences,
            columns=['id', 'sentence', 'parsed_drugs', 'parsed_pairs']
        )

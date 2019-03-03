from os import listdir
from xml.dom.minidom import parse

import pandas as pd

from models import Entity
from models import Pair


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
                    entities.append(Entity(
                        id=e.attributes['id'].value,
                        char_offset=e.attributes['charOffset'].value,
                        type=e.attributes['type'].value,
                        text=e.attributes['text'].value
                    ))

                pairs = []
                for p in s.getElementsByTagName('pair'):
                    pairs.append(Pair(
                        id=p.attributes['id'].value,
                        entity_1=p.attributes['e1'].value,
                        entity_2=p.attributes['e2'].value,
                        type=p.attributes['type'].value
                        if 'type' in p.attributes else None,
                    ))
                sentences.append([sid, text, entities, pairs])

        return pd.DataFrame(
            sentences,
            columns=['id', 'sentence', 'parsed_drugs', 'parsed_pairs']
        )

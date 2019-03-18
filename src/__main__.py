from input_output.parser import Parser
from input_output.writer import Writer
from preprocessing.tokenizer import tokenize
from preprocessing.transformations import CRF_get_tag, CRFfeatureTransformer

test_base_folder = '../resources/Test-DDI/DrugBank'
train_base_folder = '../resources/Train/DrugBank'

train = Parser(train_base_folder).call()

train['tokens'] = train['sentence'].apply(tokenize)
train['crf_features'] = train['tokens'].apply(CRFfeatureTransformer().fit_transform)
train['crf_tags'] = train[['tokens', 'parsed_drugs']].apply(CRF_get_tag, axis=1)

print(train.head())

test = Parser(test_base_folder).call()

test['tokens'] = test['sentence'].apply(tokenize)
test['crf_features'] = test['tokens'].apply(CRFfeatureTransformer().fit_transform)

print(test.head())

from models.crf import CRFClassifier

clf = CRFClassifier().fit(train['crf_features'], train['crf_tags'])
test['crf_tags'] = clf.predict(test['crf_features'])

from structs import DrugEntity

drugs = []
for tokens, crf_tags in zip(test['tokens'], test['crf_tags']):
    current_drugs = []
    current_token = None
    for token, crf_tag in zip(tokens, crf_tags):
        if crf_tag == 'O':
            if current_token is not None:
                current_drugs.append(current_token)
                current_token = None
        else:
            if current_token == None:
                current_token = DrugEntity(
                    offsets=token['char_offset'],
                    de_type=crf_tag.split('-')[-1],
                    text=token['text']
                )
            else:
                current_token.offsets = [current_token.offsets[0], token['char_offset'][1]]
                current_token.text = current_token.text + ' ' + token['text']

    drugs.append(current_drugs)

test['drugs'] = drugs

print(test.head())

Writer('../out.txt').call(test)
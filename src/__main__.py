from input_output.parser import Parser
from input_output.writer import Writer
from tokenizer import tokenize
from pipeline.rules.token_classifier import classify_token

base_folder = '../resources/Test-DDI/DrugBank'

df = Parser(base_folder).call()

df['tokens'] = df['sentence'].apply(tokenize)

get_drugs = lambda tokens: [t for t in  [classify_token(t) for t in tokens] if t is not None]
df['drugs'] = df['tokens'].apply(get_drugs)


Writer('../out.txt').call(df)

print(df.head())

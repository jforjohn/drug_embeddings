from input_output.parser import Parser
from input_output.writer import Writer
from tokenizer import tokenize
from pipeline.rules.token_classifier import classify_token, classify_tokens

base_folder = '../resources/Test-DDI/DrugBank'

df = Parser(base_folder).call()

df['tokens'] = df['sentence'].apply(tokenize)
df['drugs'] = df['tokens'].apply(classify_tokens)


Writer('../out.txt').call(df)

print(df.head())

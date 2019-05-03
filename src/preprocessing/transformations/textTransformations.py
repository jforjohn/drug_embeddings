import numpy as np
import re
from nltk import pos_tag
from nltk import download
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences


'''
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import treebank
download('treebank')
train_data = treebank.tagged_sents()

# Perceptron tagger
per = PerceptronTagger(load='false')
per.train(train_data)
'''

#download('wordnet')
#download('averaged_perceptron_tagger')
wnl = WordNetLemmatizer()

def lemmatize(p):
  if p[1][0] in {'N','V', 'J'}:
      return wnl.lemmatize(p[0].lower(), pos=p[1][0].lower())
  return p[0].lower()

def lemmas(sents):
  new_sent = []
  for sent in sents:
      pairs = pos_tag(sent)
      #pairs = per.tag(sent)
      new_sent.append([lemmatize(pair) for pair in pairs])
  return new_sent

def sentClean(sent, removals=False):
  #new_sent = []
  if sent is np.NaN:
      pass#print(sent)
  if removals:
      #mod_sent = re.sub(r'[^A-Z a-z]', ' ', sent)
      mod_sent = re.sub(r'[^\w]|[0-9]', ' ', sent)
  else:
      mod_sent = re.sub(r'[^\w]', ' ', sent)  # |\b\w\b -> to take out single chars
  clean_sent = re.sub(r'[ ]+', ' ', mod_sent.strip())
  if len(clean_sent) == 0:
      clean_sent = sent
  #new_sent.append(clean_sent)
  return clean_sent.lower()

def getCharOffsets(drug_entity):
    char_offsets = []
    for ent in drug_entity:
        char_offsets.append(ent.offsets)
    return char_offsets

def removeEmptyRows(df):
    remove_row = df['sentence'].apply(lambda x: x if len(x)<=1 else None).dropna(how='all')

    df.drop(remove_row.index, inplace=True)
    return df

def removeSpecialCases(df):
    remove_specialrow = df[df['crf_tags'].apply(len) != df['tokens_emb'].apply(len)]
    df.drop(remove_specialrow.index, inplace=True)
    return df

def padList(col, val, config):
    return pad_sequences(sequences=col, maxlen=config.get('MAX_LEN', 100), padding="post", value=val).tolist()

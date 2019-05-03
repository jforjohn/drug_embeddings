import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words('english')
whitelist_sw = ['no', 'of', 'because', 'to', 'if' 'as', 'about', 'between', 'against', 'after', 'from',  'over', 'again', 'then', 'more', 'most', 'such', 'not', 'so', 'can', 'should', 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def tokenize(sentence):
    offset = 0
    tks = []
    # word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(sentence):
        offset = sentence.find(t, offset)
        tks.append({
            'text': t.lower(),
            'char_offset': [offset, offset+len(t)-1]
        })
        offset += len(t)
    return tks

def tokenize_embed(tokens, **config):
    if config:
        tk = Tokenizer(num_words=config.get('NB_WORDS', 10000),
                       filters='',
                       lower=True,
                       split=" ")
    else:
        tk = Tokenizer(lower=True,
                       filters='',
                       split=" ")
    #words = col.apply(lambda el_lst: pd.Series([el['text'] for el in el_lst])).stack().unique()
    #word2idx = {w: i + 1 for i, w in enumerate(words)}
    sentences = tokens.apply(lambda el_lst: (' ').join([el['text'] for el in el_lst]))
    tk.fit_on_texts(sentences)
    return tk, tk.texts_to_sequences(sentences)

def tokens2sent(sentence, removals=False):
    # word_tokenize splits words, taking into account punctuations, numbers, etc.
    tokens = word_tokenize(sentence)
    if removals:
        tokens = [token for token in tokens if (
            token not in stop_words or token in whitelist_sw) and len(token)>1]
    return (' ').join(tokens)

def labelEncode(label):
    tk = Tokenizer(lower=True,
                       filters='',
                       split=" ")
    labels = label.apply(lambda el_lst: (' ').join([el for el in el_lst]))
    tk.fit_on_texts(labels)
    tk.word_index = {k:v-1 for k,v in tk.word_index.items()}
    text2seq = tk.texts_to_sequences(labels)
    label_seq = [list(map(lambda x: x-1,label)) for label in text2seq]
    return tk, label_seq

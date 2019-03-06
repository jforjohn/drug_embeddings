from nltk.tokenize import word_tokenize


def tokenize(sentence):
    offset = 0
    tks = []
    # word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(sentence):
        offset = sentence.find(t, offset)
        tks.append({
            'text': t,
            'char_offset': [offset, offset+len(t)-1]
        })
        offset += len(t)

    return tks

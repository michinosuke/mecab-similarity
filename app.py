import os
import sys
import json
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

dicdir = os.path.join(os.getcwd(), 'ipadic', 'dicdir')
rcfile = os.path.join(os.getcwd(), "ipadic", "dicdir", "mecabrc")
tagger = MeCab.Tagger("-r{} -d{}".format(rcfile, dicdir))

def get_nouns(str: str):
    nouns = ''
    for row in str.split("\n"):
        word = row.split("\t")[0]
        if word == "EOS":
            break
        type = row.split("\t")[1].split(',')[0]
        if type == "名詞":
            nouns += " " + word
    return nouns

def get_tf_idf(word_list):
    docs = np.array(word_list)
    print(docs)
    vectorizer = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    vectors = vectorizer.fit_transform(docs)
    vectors_array = vectors.toarray()
    return vectors_array

def get_cos_sim(v1,v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def handler(event, context):
    a = event.get('a', '')
    b = event.get('b', '')
    parsed_a = tagger.parse(a)
    parsed_b = tagger.parse(b)
    nouns_a = get_nouns(parsed_a)
    nouns_b = get_nouns(parsed_b)
    print('a', json.dumps([nouns_a, nouns_b]))
    tf_idf = get_tf_idf([nouns_a, nouns_b])
    print(tf_idf)
    print(json.dumps(tf_idf.tolist()))
    print(np.array(tf_idf.tolist()))
    cos_sim = get_cos_sim(tf_idf[1], tf_idf[0])

    return {
        "a": a,
        "b": b,
        "similarity": cos_sim
    }

if __name__ == '__main__':
    event = {
        "a": "ZIGがVTuberグループ「流星群」のオーディションを開始",
        "b": "「流星群プロジェクト」第１号さん、始動"
    }
    print(handler(event, {}))
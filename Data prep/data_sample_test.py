# import scipy.io
# mat = scipy.io.loadmat('data/syl1.mat')
# import pandas as pd
# print(mat)
import os
from mat4py import loadmat
import pprint
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import numpy as np

data = loadmat('data/phn1.mat')
# print(data)

def syl_data(data):
    duration = []
    sylStart = [0]
    for i in range(len(data['spurtSylTimes'])):
        duration.append(data['spurtSylTimes'][i][1] - data['spurtSylTimes'][i][0])
        sylStart.append(data['spurtSylTimes'][i][0])
    data['spurtSylTimes'] = duration
    data['sylStart'] = sylStart
    out = []
    count=0
    for i in range(500):
        if data['sylStart'][count]*1000 == i:
            count+=1
            out.append(1)
        else: out.append(0)
    # print(out)
    return data['spurtSyl']
# print(data)


def phn_data(data):

    duration = []
    phnStart = [0]
    for i in range(len(data['phnTimes'])):
        duration.append(round(data['phnTimes'][i][1] - data['phnTimes'][i][0],2))
        phnStart.append(data['phnTimes'][i][0])
    data['phnTimes'] = duration
    data['phnStart'] = phnStart

    out_ph = []
    count_ph = 0

    for i in range(500):
        if data['phnStart'][count_ph]*1000 == i:
            count_ph+=1

        else:
            out_ph.append(data['phnTimes'][count_ph-1]*1000)
    print(out_ph)
    return out_ph
# print(data)
# new = pd.DataFrame.from_dict(data)

# print(new)

# linguistic = ["phn","syl","wav"]
# folder = os.listdir('data/phn')

# print(folder)

# for files in folder:
#     # cur_folder = os.listdir(folder + "/")
#     print(files)
## populating sample frequency
# out = []
# count=0
# for i in range(500):
#     if data['sylStart'][count]*1000 == i:
#         count+=1
#         out.append(1)
#     else: out.append(0)
# print(out)

## population phoneme duration



# print(out_ph)
## To iterate over all the files
files  = os.listdir("data/syl")
outs = []
for file in files:
    data = loadmat("data/syl/"+file)
    outs.append(syl_data(data))

# pprint.pprint(outs)

## read from text file and vectorize
with open('data/transcript.txt', 'r', encoding='utf-8') as file:
    text = file.read()

## Tokenize text 
tokens = word_tokenize(text)
print(tokens)

# print(text[1])
sentences = []
words = []
for chars in tokens:
    if chars != '.':
        words.append(chars)
    else:
        sentences.append(words)
        words = []
        continue
print(sentences)

## vectorizing sentences 
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
model = Word2Vec(sentences=sentences, vector_size=10, window=1, min_count=1, workers=4)
model.save("word2vec.model")
word2vec_model = model

def get_vector(token):
    try:
        return word2vec_model.wv[token]
    except KeyError:
        # Handle the case when the token is not in the vocabulary
        return np.zeros(word2vec_model.vector_size)
vectors = []
for token in tokens:
    if token=='.':
        print("hi") 
        continue
    else : vectors.append(get_vector(token))
# vectors = [get_vector(token) for token in tokens]

print(vectors[:10])
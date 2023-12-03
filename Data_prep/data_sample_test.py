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

data = loadmat('data/syl1.mat')
print(data)

def syl_data(data,data_len):
    duration = []
    sylStart = []
    for i in range(len(data['spurtSylTimes'])):
        duration.append(round(data['spurtSylTimes'][i][1] - data['spurtSylTimes'][i][0],2))
        sylStart.append(round(data['spurtSylTimes'][i][0],2))
    data['spurtSylTimes'] = duration
    data['sylStart'] = sylStart
    out = []
    count=0
    factor = int(data_len/(sylStart[-1]*100))
    cut_off = len(sylStart)
    print(factor)
    for i in range(data_len):
        if count >= cut_off : count -= 1
        if int(data['sylStart'][count]*100*factor) == i:
            count+=1
            out.append(1)
        else: out.append(0)
    print(data)
    return out
# print(data)


def phn_data(data,data_len):

    duration = []
    phnStart = [0]
    for i in range(len(data['phnTimes'])):
        duration.append(round(data['phnTimes'][i][1] - data['phnTimes'][i][0],2))
        phnStart.append(round(data['phnTimes'][i][0],2))
    data['phnTimes'] = duration
    data['phnStart'] = phnStart

    out_ph = []
    count_ph = 0
    factor = int(data_len/(phnStart[-1]*100))
    cut_off = len(phnStart)
    # (print(cut_off))
    # print(factor,phnStart[-1])
    for i in range(data_len):
        
        if int(data['phnStart'][count_ph]*100*factor) == i:
            # print(data['phnStart'][count_ph]*100*factor,i)
            if(count_ph == cut_off-1) : 
                # print(i)
                out_ph.append(data['phnTimes'][count_ph-1]*1000)
                continue
            else :
                count_ph+=1
                # print(i,data['phnTimes'][count_ph-1]*1000)
                out_ph.append(data['phnTimes'][count_ph-1]*1000)

        else:
            out_ph.append(data['phnTimes'][count_ph-1]*1000)
            # print(i)
    # print()
    return out_ph
# print(data)
## population phoneme duration

# print(out_ph)
## To iterate over all the files
files  = os.listdir("data/syl")
outs = []
data_len = 1200
for i,file in enumerate(files):
    if i == 1: break
    data = loadmat("data/syl/"+file)
    outs.append(syl_data(data,data_len))

print(len(outs[0]))

### Read from text file and vectorize
with open('data/transcript.txt', 'r', encoding='utf-8') as file:
    text = file.read()

## Tokenize text 
tokens = word_tokenize(text)
# print(tokens)

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
# print(sentences)

## vectorizing sentences 
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
model = Word2Vec(sentences=sentences, vector_size=32, window=1, min_count=1, workers=4)
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
        # print("hi") 
        continue
    else : vectors.append(get_vector(token))
# vectors = [get_vector(token) for token in tokens]

sentence_vectors = []
word_vecs = []
for i in range(len(sentences)):
    for token in sentences[i]:
        word_vecs.append(get_vector(token))
    sentence_vectors.append(word_vecs)
    word_vecs = []

# print(sentence_vectors[0])

def syl_val(vector,sample_freq,data_len):
    zero_val = np.zeros(32)
    out_vec= []
    count = 0
    print(sample_freq[0])
    print(vector)
    print("hi")
    for i in range(data_len):
        if sample_freq[i] == 1:
            print(count)
            if count == len(vector):
                out_vec.append(zero_val)
                continue
            else:
                out_vec.append(vector[count])
                count +=1
        else: out_vec.append(zero_val)
    return out_vec


syl_v = syl_val(sentence_vectors[0],outs[0],data_len)
print(len(syl_v))







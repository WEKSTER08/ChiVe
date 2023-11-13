import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_sequence, unpack_sequence
from cwrnn_torch import ClockworkRNNLayer    
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random

class CHIVE(nn.Module):
    def __init__(self, latent_space_dim,input_size):
        super(CHIVE, self).__init__()
        self.latent_space_dim = latent_space_dim
        self.layers = nn.ModuleList()
        
        self.hidden_size = 32
        self.frnn_layer =  ClockworkRNNLayer(input_size[0], hidden_size= self.hidden_size)
        self.phrnn_layer =  ClockworkRNNLayer(input_size[1], hidden_size= self.hidden_size)
        self.sylrnn_layer =  ClockworkRNNLayer(input_size[2], hidden_size =self.hidden_size)
        self.phrnn_decd = ClockworkRNNLayer(self.hidden_size, hidden_size =self.hidden_size)
        self.phrnn_dur = ClockworkRNNLayer(self.hidden_size, hidden_size =1)
        self.frnn_c = ClockworkRNNLayer(self.hidden_size, hidden_size =12)
        self.frnn_f = ClockworkRNNLayer(self.hidden_size, hidden_size =1)
        self.sylrnn_dur = ClockworkRNNLayer(self.hidden_size, hidden_size =1)



        self.bottle_neck_representation = None
        # print(self.parameters)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        

    def forward(self, x):
        length = len(x)
        frnn_inp, phrnn_inp, sylrnn_inp,sample_freq = x
        def encoder(frnn_inp, phrnn_inp, sylrnn_inp,sample_freq):
            frnn_inp, phrnn_inp, sylrnn_inp,sample_freq = x
            #frame rate rnn intialization
            frnn_seq = frnn_inp[0]
            frnn_clock = frnn_inp[1]
            h_frnn = torch.zeros(self.hidden_size)

            #phone rate rnn initialisation
            phrnn_seq = phrnn_inp[0]
            phrnn_clock = phrnn_inp[1]
            h_phrnn = torch.zeros(self.hidden_size)

            h_sylrnn = torch.zeros(self.hidden_size)
            freq_count = 0
            # self.frnn_layer = ClockworkRNNLayer(input_size=frnn_seq[0].size(),hidden_size=self.hidden_size,clock_val=)
            for t in range(len(sample_freq)):
                ##Asynchronously adding frame_rate and phone rate rnn layers
                h_frnn = self.frnn_layer(x=frnn_seq[t],h_prev = h_frnn,timestep= t, clock_val =frnn_clock[t])
                h_frnn = self.frnn_layer(x=frnn_seq[t],h_prev = h_frnn,timestep= t, clock_val =frnn_clock[t])
                h_phrnn = self.phrnn_layer(x=phrnn_seq[t],h_prev = h_phrnn,timestep= t, clock_val =phrnn_clock[t])
                h_phrnn = self.phrnn_layer(x=phrnn_seq[t],h_prev = h_phrnn,timestep= t, clock_val =phrnn_clock[t])
                ## Passing the frame rate and phone rate rnn layers to the Syllable rate rnn layers along with linguistic features
                if sample_freq[t] == 1:
                    syl_clock = torch.randint(2, self.hidden_size-2, (1,)).item()
                    # print("Lens-",(h_frnn.shape), (h_phrnn.shape), (sylrnn_inp[t].shape))
                    # max_length = max((h_frnn.shape[1]), (h_phrnn.shape[1]), (sylrnn_inp[t].shape[1]))
                   

                    # h_frnn = F.pad(h_frnn, pad=(0, max_length - (h_frnn.shape[1])))
                    # h_phrnn = F.pad(h_phrnn, pad=(0, max_length - (h_phrnn.shape[1])))
                    # # sylrnn_inp[t] = F.pad(sylrnn_inp[t], pad=(0, max_length - (sylrnn_inp[t].shape[1])))
                    # syl_pad = torch.tensor(np.zeros( max_length - (sylrnn_inp[t].shape[1])))
                    # syl_inp = torch.cat((sylrnn_inp[t],syl_pad.unsqueeze(-1).t()), dim=1)
                    # syl_inp = syl_inp.to(torch.float32)
                    # result_tensor = torch.stack([h_frnn, h_phrnn, syl_inp], dim=0)
                    ##Adding the frnn output value, phrnn output value and linguistic feature value
                    result_tensor = h_frnn+h_phrnn+sylrnn_inp[t]
                    h_sylrnn = self.sylrnn_layer(x= result_tensor,h_prev = h_sylrnn,timestep= freq_count, clock_val =syl_clock)
                    yield h_sylrnn 

        ##define bottleneck representations
        bottle_neck_representation = encoder(frnn_inp, phrnn_inp, sylrnn_inp,sample_freq)
        
        def decoder(bottle_neck_representation):
            h_sylrnn_decd = torch.zeros(self.hidden_size)
            h_phrnn_decd = torch.zeros(self.hidden_size)
            h_phrnn_dur = torch.zeros(1)
            h_frnn_c = torch.zeros(12)
            h_frnn_f = torch.zeros(1)
            h_sylrnn_dur = torch.zeros(1)
            for i,x_inp in enumerate(bottle_neck_representation):
                timestep = i
                # print(i)
                ## syllable rate rnn;phone rate rnn; phone rate duration; syllable rate duration;phone rate rnn -> f0,c0
                if i>0 and (sample_freq[i-1]==1 or h_sylrnn_dur==1):
                    ##syllable rate rnn hidden state init
                    h_sylrnn_decd = torch.zeros(self.hidden_size)
                    ##phone rate rnn hiddenstate init
                    h_phrnn_decd = torch.zeros(self.hidden_size)
                    clock_decd = torch.randint(2, self.hidden_size-2, (1,)).item()
                    ##syllable rate rnn
                    h_sylrnn_decd = self.sylrnn_layer(x = x_inp,h_prev= h_sylrnn_decd,timestep = timestep,clock_val=clock_decd)
                    # print("Shape of this tensor",h_sylrnn_decd.shape,"Shape of input tensor--",x_inp.shape)
                    ##phone rate rnn
                    h_phrnn_decd = self.phrnn_decd(x = h_sylrnn_decd,h_prev = h_phrnn_decd,timestep=timestep,clock_val = clock_decd)
                    ##phone rate duration
                    h_phrnn_dur = self.phrnn_dur(x =h_phrnn_decd,h_prev = h_phrnn_dur,timestep=timestep,clock_val = 1)
                    ##syllable rate duration
                    h_sylrnn_dur=  self.sylrnn_dur(x = h_sylrnn_decd,h_prev = h_sylrnn_dur,timestep=timestep,clock_val = 1)

                ##f0 values
                h_frnn_f = self.frnn_f(x=h_phrnn_decd,h_prev=h_frnn_f,timestep=timestep,clock_val = 1)
                clock_decd_c = torch.randint(2,10, (1,)).item()
                ##co values
                h_frnn_c = self.frnn_c(x = h_phrnn_decd,h_prev = h_frnn_c,timestep=timestep,clock_val=clock_decd_c)

            return h_frnn_f,h_frnn_c,h_phrnn_dur,h_sylrnn_dur
                    # self.layers.append(h_sylrnn)

        # decoder_out = decoder(bottle_neck_representation)
        return decoder(bottle_neck_representation)
    def summary(self):
        print("\nModel Summary:")
        print("=" * 50)
        print("Clockwork RNN Layer:")
        print("-" * 30)
        total_params = 0
        for name, param in self.named_parameters():
            print(f"{name}: {param.size()}")
            total_params += param.numel()
        print("=" * 50)
        print(f"Total Trainable Parameters: {total_params}")


    def train(self, x_train, y_train, batch_size, num_epochs):
        frnn_train,frnn_clock, phrnn_train,phrnn_clock,sylrnn_train,seq_train = x_train
        # print(frnn_train)
        frnn_train = frnn_train.to(torch.float32).requires_grad_(True)
        frnn_clock = frnn_clock.to(torch.float32).requires_grad_(True)
        phrnn_train = phrnn_train.to(torch.float32).requires_grad_(True)
        phrnn_clock = phrnn_clock.to(torch.float32).requires_grad_(True)
        sylrnn_train = sylrnn_train.to(torch.float32).requires_grad_(True)
        seq_train = torch.tensor(seq_train,dtype=torch.float32).requires_grad_(True).clone()
        # sylrnn_train = torch.tensor(sylrnn_train, dtype=torch.float32)
        y_train = torch.tensor(y_train,dtype=torch.float32).requires_grad_(True)

        dataset = torch.utils.data.TensorDataset(frnn_train,frnn_clock, phrnn_train,phrnn_clock, sylrnn_train,seq_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.loss_function = nn.MSELoss()
        for epoch in range(num_epochs):
            
            for i,data in enumerate(train_loader):
                frnn_batch,frnn_clock, phrnn_batch,phrnn_clock, sylrnn_batch,seq_batch, y_batch = data
                # print(frnn_batch.shape)
                frnn_c_batch, frnn_f_batch = frnn_batch[:, :12],frnn_batch[:, 12:]
                phrnn_dur_batch = phrnn_batch[:,2:]
                # print(phrnn_dur_batch.shape,frnn_c_batch.shape, frnn_f_batch.shape)
                # print("Shapes--",frnn_batch.shape,phrnn_batch.shape,seq_batch.shape)
                self.optimizer.zero_grad()
                length = len(train_loader) -i
                if length/batch_size >1 :  batch_len = batch_size
                else: batch_len = length%batch_size
                total_loss = Variable(torch.zeros(1), requires_grad=True)
                for i in range(batch_len):
                    h_frnn_f,h_frnn_c,h_phrnn_dur,h_sylrnn_dur = self.forward([(frnn_batch,frnn_clock), (phrnn_batch,phrnn_clock), sylrnn_batch,seq_batch])
                    # print("shapes of outputs",h_frnn_c.view(12).shape,frnn_c_batch[i,:,:].view(12).shape)
                # print(output.shape,y_batch.shape)
                

                    loss_f = self.loss_function(h_frnn_f.view(1),frnn_f_batch[i].view(1))
                    loss_c = self.loss_function(h_frnn_c.view(12),frnn_c_batch[i].view(12))
                    loss_ph_dur = self.loss_function(h_phrnn_dur.view(1),phrnn_dur_batch[i].view(1))
                    loss_syl_dur = self.loss_function(h_sylrnn_dur.view(1),seq_batch[i].view(1))
                    total_loss = total_loss.clone()
                    total_loss += (loss_c+loss_f+loss_ph_dur+loss_syl_dur)/4
                total_loss = total_loss/batch_len
                total_loss.backward()
                self.optimizer.step()
            print("Epoch--",epoch+1,"| Loss --",total_loss.item())



if __name__ == "__main__":

    
    # chive.summary()

    # num_samples = 64
    frnn_sequence_length = 13
    phrnn_sequence_length = 3
    sylrnn_sequence_length = 32
    input_size = [frnn_sequence_length,phrnn_sequence_length,sylrnn_sequence_length]
    chive = CHIVE(latent_space_dim=1,input_size=input_size)

    def generate_non_repeating_subset(start, end, subset_size):
        if subset_size > (end - start):
            raise ValueError("Subset size cannot be greater than the range size.")

        numbers = list(range(start, end))
        random.shuffle(numbers)
        subset = numbers[:subset_size]
        return subset

    # frnn_seq = torch.tensor(np.random.rand(num_samples,1, 13))
    # clock_frnn = torch.tensor([np.random.uniform(1, 6) for i in range(num_samples)])
    # phrnn_seq = torch.tensor(np.random.rand(num_samples,1, 3))
    # clock_phrnn = torch.tensor([np.random.uniform(1, 6) for i in range(num_samples)])
    # sylrnn_val = torch.tensor(np.random.rand(int(num_samples - num_samples/2),1,1))
    # seq_data = generate_non_repeating_subset(0,int(num_samples), int(num_samples/2))
    # seq_timesteps = torch.zeros(num_samples)
    # sylrnn_data = torch.zeros_like(frnn_seq) 
    # syl_seq_count = 0
    # for i in range(len(seq_timesteps)):
    #     if i in seq_data:
    #         sylrnn_data[i] =  torch.tensor(np.random.rand(1,1, 13))
    #         seq_timesteps[i]=torch.tensor(1)
    #         syl_seq_count+=1
    #     else: seq_timesteps[i]=torch.tensor(0)
    # print(seq_timesteps)

    # print(frnn_seq)
    # print(len(frnn_seq),len(phrnn_seq),len(sylrnn_data),(seq_timesteps[99]),len(clock_frnn),len(clock_phrnn),len(sylrnn_val),len(seq_data))


    ### Real data prep 


    ## Frnn seq, Phrnn seq......................................................................................................
    ## To equal the length of the f0 and c0 values by averaging
    def average_reduce(input_list, target_length):
        original_length = len(input_list)
        dimension = len(input_list[0])  # Assuming all elements have the same dimensionality
        step = original_length // target_length

        reduced_list = [
            np.mean(input_list[i:i+step], axis=0) for i in range(0, original_length, step)
        ]

        return reduced_list[:target_length]

    def average_reduce_ph(input_list, target_length):
        original_length = len(input_list)
        dimension = len(input_list[0])  # Assuming all elements have the same dimensionality
        step = original_length // target_length

        reduced_list = [
            np.mean(input_list[i]) for i in range(0, original_length, step)
        ]
        print(reduced_list[:10])

        return reduced_list[:target_length]
    # Load the audio file
    # file_path = 'path/to/your/audio/file.wav'
    def extract_f_c(data_path,ph):
        y, sr = librosa.load(data_path,sr=16000,)

        # Compute the short-time Fourier transform (STFT)
        D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

        # Extract the frequency values
        frequencies = librosa.fft_frequencies(sr=16000)

        mfcc_val = librosa.feature.mfcc(y=y, n_mfcc = 12, sr=sr,hop_length= 10)
        mfcc_val = mfcc_val.T
        f0, voiced_flag, voiced_probs = librosa.pyin(y, sr = sr, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C7'),frame_length= 20, hop_length= 10 )
        
        if ph ==1 : 
            # print("ph")
            reduced_mfcc = average_reduce_ph(mfcc_val,len(frequencies))
        else: 
            # print("No ph")
            reduced_mfcc = average_reduce(mfcc_val,len(frequencies))
        frnn_inp = []
        for i in range(len(reduced_mfcc)):

            frnn_inp.append(np.hstack((reduced_mfcc[i],frequencies[i])))
            # frnn_inp.append([vals,frequencies[i]])
        # print(reduced_mfcc[:5])
        frequencies = frequencies.reshape(-1,1)
        # frnn_inp = np.concatenate((reduced_mfcc,frequencies),axis =1)
        return frnn_inp

    f_c_inp = extract_f_c('../Data_prep/data/wav/ISLE_SESS0003_BLOCKD01_01_sprt1.wav',0)
    f_cav_inp = extract_f_c('../Data_prep/data/wav/ISLE_SESS0003_BLOCKD01_01_sprt1.wav',1)
    print((f_c_inp[:10]),len(f_cav_inp))

    ## Phrnn dur, Sample_freq, Sylrnn inp ...................................................................................

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

    
    # print(data)

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
        # print(factor)
        for i in range(data_len):
            if count >= cut_off : count -= 1
            if int(data['sylStart'][count]*100*factor) == i:
                count+=1
                out.append(1)
            else: out.append(0)
        # print(data)
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
    files  = os.listdir("../Data_prep/data/syl")
    outs = []
    data_len = len(f_c_inp)
    for i,file in enumerate(files):
        if i == 1: break
        data = loadmat("../Data_prep/data/syl/"+file)
        outs.append(syl_data(data,data_len))

    # print(len(outs[0]))

    files  = os.listdir("../Data_prep/data/phn")
    outs_ph = []
    data_len = len(f_c_inp)
    for i,file in enumerate(files):
        if i == 1: break
        data = loadmat("../Data_prep/data/phn/"+file)
        outs_ph.append(phn_data(data,data_len))

    # print(len(outs_ph[0]))

    ### Read from text file and vectorize
    with open('../Data_prep/data/transcript.txt', 'r', encoding='utf-8') as file:
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
        # print(sample_freq[0])
        # print(vector)
        # print("hi")
        for i in range(data_len):
            if sample_freq[i] == 1:
                # print(count)
                if count == len(vector):
                    out_vec.append(zero_val)
                    continue
                else:
                    out_vec.append(vector[count])
                    count +=1
            else: out_vec.append(zero_val)
        return out_vec


    syl_v = syl_val(sentence_vectors[0],outs[0],data_len)
    # print(len(syl_v))







    print(len(f_c_inp),len(f_cav_inp),len(syl_v),len(outs_ph[0]),len(outs[0]))
    num_samples = data_len
    clock_frnn = torch.tensor([np.random.uniform(1, 6) for i in range(num_samples)])
    clock_phrnn = torch.tensor([np.random.uniform(1, 6) for i in range(num_samples)])
    
    # for i in f_c_inp: i = torch.tensor(i[0])
    
#     f_c_inp_t = [
#     [torch.tensor(arr) if isinstance(arr, np.ndarray) else torch.tensor(arr) for arr in inner_list]
#     for inner_list in f_c_inp
# ]
#     print(f_c_inp_t[:10])
    frnn_seq = torch.tensor(f_c_inp)
    phrnn_seq = []
    for i in range(len(f_cav_inp)):
        phrnn_seq.append(np.hstack((f_cav_inp[i],outs_ph[0][i])))
    # phrnn_seq = np.concatenate((f_cav_inp,outs_ph[0]),axis =1)
    print(phrnn_seq[:2])
    phrnn_seq = torch.tensor(phrnn_seq)
    sylrnn_data = torch.tensor(syl_v)
    seq_timesteps = torch.tensor(outs[0])
    dummy_targets = np.random.rand(num_samples,32)
    # model = MyModel()
    total_params = sum(p.numel() for p in chive.parameters())
    print(f"Number of parameters: {total_params}")
    # chive.summary()
    chive.train(x_train=[frnn_seq,clock_frnn, phrnn_seq,clock_phrnn,sylrnn_data,seq_timesteps], y_train=dummy_targets, batch_size=16, num_epochs=16)

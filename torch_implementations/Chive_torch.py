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
import os
from mat4py import loadmat
import pprint
import nltk
import pysptk
from scipy.io import wavfile
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import numpy as np
class BottleneckLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BottleneckLayer, self).__init__()
        
        # Define the fully connected layers
        self.fc_mean = nn.Linear(input_size, hidden_size)
        self.fc_logvar = nn.Linear(input_size, hidden_size)
        
    def forward(self, x):
        # Forward pass through the bottleneck layer
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        
        return mean, logvar

class CHIVE(nn.Module):
    def __init__(self, latent_space_dim,input_size):
        super(CHIVE, self).__init__()
        self.latent_space_dim = latent_space_dim
        self.layers = nn.ModuleList()
        
        self.hidden_size = 32
        self.activation = nn.GELU()
        self.frnn_layer0 =  ClockworkRNNLayer(input_size[0], hidden_size= self.hidden_size)
        self.frnn_layer1 =  ClockworkRNNLayer(self.hidden_size, hidden_size= self.hidden_size)
        self.phrnn_layer0 =  ClockworkRNNLayer(input_size[1], hidden_size= self.hidden_size)
        self.phrnn_layer1 =  ClockworkRNNLayer(self.hidden_size, hidden_size= self.hidden_size)
        self.sylrnn_layer =  ClockworkRNNLayer(input_size[2], hidden_size =self.hidden_size)
        self.phrnn_decd = ClockworkRNNLayer(self.hidden_size, hidden_size =self.hidden_size)
        self.phrnn_dur = ClockworkRNNLayer(self.hidden_size, hidden_size =1)
        self.frnn_c = ClockworkRNNLayer(self.hidden_size, hidden_size =12)
        self.frnn_f = ClockworkRNNLayer(self.hidden_size, hidden_size =1)
        self.sylrnn_dur = ClockworkRNNLayer(self.hidden_size, hidden_size =1)
        self.bottleneck = BottleneckLayer(input_size=self.hidden_size,hidden_size = self.hidden_size)



        self.bottle_neck_representation = None
        # print(self.parameters)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        length = len(x)
        frnn_inp, phrnn_inp, sylrnn_inp,sample_freq = x
        def encoder(frnn_inp, phrnn_inp, sylrnn_inp,sample_freq):
            frnn_inp, phrnn_inp, sylrnn_inp,sample_freq = x
            #frame rate rnn intialization
            frnn_seq = frnn_inp[0]
            frnn_clock = frnn_inp[1]
            h_frnn = torch.zeros(self.hidden_size)
            h_frnn0 = torch.zeros(self.hidden_size)

            #phone rate rnn initialisation
            phrnn_seq = phrnn_inp[0]
            phrnn_clock = phrnn_inp[1]
            h_phrnn = torch.zeros(self.hidden_size)
            h_phrnn0 = torch.zeros(self.hidden_size)

            h_sylrnn = torch.zeros(self.hidden_size)
            freq_count = 0
            # self.frnn_layer = ClockworkRNNLayer(input_size=frnn_seq[0].size(),hidden_size=self.hidden_size,clock_val=)
            for t in range(len(sample_freq)):
                ##Asynchronously adding frame_rate and phone rate rnn layers
                h_frnn0 = self.frnn_layer0(x=frnn_seq[t],h_prev = h_frnn0,timestep= t, clock_val =frnn_clock[t])
                h_frnn = self.frnn_layer1(x=h_frnn0,h_prev = h_frnn,timestep= t, clock_val =frnn_clock[t])
                h_frnn = self.activation(h_frnn)
                h_phrnn0 = self.phrnn_layer0(x=phrnn_seq[t],h_prev = h_phrnn0,timestep= t, clock_val =phrnn_clock[t])
                h_phrnn = self.phrnn_layer1(x=h_phrnn0,h_prev = h_phrnn,timestep= t, clock_val =phrnn_clock[t])
                h_phrnn = self.activation(h_phrnn)
                ## Passing the frame rate and phone rate rnn layers to the Syllable rate rnn layers along with linguistic features
                if sample_freq[t] == 1:
                    syl_clock = torch.randint(2, self.hidden_size-2, (1,)).item()
                    # if padding is required
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
                    h_sylrnn = self.activation(h_sylrnn)
                    yield h_sylrnn 

        ##define bottleneck representations
        bottle_neck_representation = encoder(frnn_inp, phrnn_inp, sylrnn_inp,sample_freq)

        def decoder(z):
            h_sylrnn_decd = torch.zeros(self.hidden_size)
            h_phrnn_decd = torch.zeros(self.hidden_size)
            h_phrnn_dur = torch.zeros(1)
            kl_loss = torch.zeros(1)
            h_frnn_c = torch.zeros(12)
            h_frnn_f = torch.zeros(1)
            h_sylrnn_dur = torch.zeros(1)
            for i,x_inp in enumerate(bottle_neck_representation):
                mean, logvar = self.bottleneck(x_inp)
                x_inp = self.reparameterize(mean, logvar)
                kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
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
                    h_sylrnn_decd = self.activation(h_sylrnn_decd)
                    # print("Shape of this tensor",h_sylrnn_decd.shape,"Shape of input tensor--",x_inp.shape)
                    ##phone rate rnn
                    h_phrnn_decd = self.phrnn_decd(x = h_sylrnn_decd,h_prev = h_phrnn_decd,timestep=timestep,clock_val = clock_decd)
                    h_phrnn_decd = self.activation(h_phrnn_decd)
                    ##phone rate duration
                    # h_phrnn_dur = self.phrnn_dur(x =h_phrnn_decd,h_prev = h_phrnn_dur,timestep=timestep,clock_val = 1)
                    # h_phrnn_dur = self.activation(h_phrnn_dur)
                    # ##syllable rate duration
                    # h_sylrnn_dur=  self.sylrnn_dur(x = h_sylrnn_decd,h_prev = h_sylrnn_dur,timestep=timestep,clock_val = 1)
                    # h_sylrnn_dur = self.activation(h_sylrnn_dur)

                ##f0 values
                h_frnn_f = self.frnn_f(x=h_phrnn_decd,h_prev=h_frnn_f,timestep=timestep,clock_val = 1)
                h_frrn_f = self.activation(h_frnn_f)
                clock_decd_c = torch.randint(2,10, (1,)).item()
                ##co values
                h_frnn_c = self.frnn_c(x = h_phrnn_decd,h_prev = h_frnn_c,timestep=timestep,clock_val=clock_decd_c)
                h_frrn_c = self.activation(h_frnn_c)

            return (h_frnn_f,h_frnn_c),kl_loss
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
        text_file = open("train_log.txt", "w")
        
        for epoch in range(num_epochs):
            
            for i,data in enumerate(train_loader):
                frnn_batch,frnn_clock, phrnn_batch,phrnn_clock, sylrnn_batch,seq_batch, y_batch = data
                # print(frnn_batch.shape)
                frnn_c_batch, frnn_f_batch = frnn_batch[:, :12],frnn_batch[:, 12:]
                # phrnn_dur_batch = phrnn_batch[:,2:]
                # print
                # print(phrnn_dur_batch.shape,frnn_c_batch.shape, frnn_f_batch.shape)
                # print("Shapes--",frnn_batch.shape,phrnn_batch.shape,seq_batch.shape)
                self.optimizer.zero_grad()
                length = len(train_loader) -i
                if length/batch_size >=1 :  batch_len = batch_size
                else: batch_len = length%batch_size
                total_loss =  Variable(torch.zeros(1), requires_grad=True)
                for i in range(batch_len):
                    (h_frnn_f,h_frnn_c),kl_loss = self.forward([(frnn_batch,frnn_clock), (phrnn_batch,phrnn_clock), sylrnn_batch,seq_batch])
                    # print("shapes of outputs",h_frnn_c.view(12).shape,frnn_c_batch[i,:,:].view(12).shape)
                # print(output.shape,y_batch.shape)
                

                    loss_f = self.loss_function(h_frnn_f.view(1),frnn_f_batch[i].view(1))
                    loss_c = self.loss_function(h_frnn_c.view(12),frnn_c_batch[i].view(12))
                    # loss_ph_dur = self.loss_function(h_phrnn_dur.view(1),phrnn_dur_batch[i].view(1))
                    # loss_syl_dur = self.loss_function(h_sylrnn_dur.view(1),seq_batch[i].view(1))
                    total_loss = total_loss.clone()
                    total_loss += (loss_c+loss_f+kl_loss)/3
                total_loss = total_loss/batch_len
                total_loss.backward()
                self.optimizer.step()
            print("Epoch--",epoch+1,"| Loss --",total_loss.item())
            text_file.write("Loss: %f price %f" % (total_loss.item(), epoch+1))
            
        text_file.close()


if __name__ == "__main__":

    def standardize_list(input_list):
        # Convert the list to a NumPy array
        data_array = np.array(input_list)

        # Calculate mean and standard deviation
        mean_value = np.mean(data_array)
        std_dev = np.std(data_array)

        # Standardize the array
        standardized_array = (data_array - mean_value) / std_dev

        # Convert the NumPy array back to a list
        standardized_list = standardized_array

        return standardized_list
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
        # print(reduced_list[:10])

        return reduced_list[:target_length]
    # Load the audio file
    # file_path = 'path/to/your/audio/file.wav'
    def extract_f_c(data_path,ph):
        y, sr = librosa.load(data_path,sr=16000)
        sampling_rate, x = wavfile.read(data_path)

        # Compute the short-time Fourier transform (STFT)
        D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

        # Extract the frequency values
        frequencies = librosa.fft_frequencies(sr=16000)

        mfcc_val = librosa.feature.mfcc(y=y, n_mfcc = 12, sr=sr,hop_length= 10)
        mfcc_val = mfcc_val.T
        f0 = pysptk.swipe(x.astype(float), fs=sampling_rate/2, hopsize=10, min=60, max=240)
        mfcc_val = mfcc_val[:len(f0)]
        # f0, voiced_flag, voiced_probs = librosa.pyin(y, sr = sr, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C7'),frame_length= 20, hop_length= 10 )
        frequencies = frequencies.reshape(-1,1)
        print(len(f0),len(mfcc_val))
        pad_length = len(mfcc_val) -len(frequencies)
        # for i in range(pad_length): frequencies = np.vstack((frequencies,[0]))
        frequencies = frequencies.reshape(-1,1)
        # print(frequencies.shape)
        if ph ==1 : 
            # print("ph")
            reduced_mfcc = average_reduce_ph(mfcc_val,4096)
        else: 
            # print("No ph")
            reduced_mfcc = average_reduce(mfcc_val,4096)
        frnn_inp = []
        for i in range(len(reduced_mfcc)):

            frnn_inp.append(np.hstack((reduced_mfcc[i],f0[i])))
            # frnn_inp.append([vals,frequencies[i]])
        # print()

        # print(len(frnn_inp))
        # frnn_inp = np.concatenate((reduced_mfcc,frequencies),axis =1)
        frnn_inp = standardize_list(frnn_inp)
        print(frnn_inp.shape)
        return frnn_inp

    files  = os.listdir("../Data_prep/data/wav")
    f_c_inp  = []
    f_cav_inp  = []
    
    
    for i,file in enumerate(files):
        if i == 7: break
        print(i)
        f_c_inp.append(extract_f_c("../Data_prep/data/wav/"+file,0))
        # print(len(f_c_inp),len(f_c_inp[0]))
        f_cav_inp.append(extract_f_c("../Data_prep/data/wav/"+file,1))
    # print(f_c_inp)   
    f_c_inp_np = np.array(f_c_inp)
    reshaped_arr = f_c_inp_np.reshape((f_c_inp_np.shape[0] * f_c_inp_np.shape[1],f_c_inp_np.shape[2] ))
    f_c_inp = reshaped_arr.tolist()
    
    f_cav_inp_np = np.array(f_cav_inp)
    reshaped_arr = f_cav_inp_np.reshape((f_cav_inp_np.shape[0] * f_cav_inp_np.shape[1],f_cav_inp_np.shape[2] ))
    f_cav_inp = reshaped_arr.tolist()
    # f_c_inp = extract_f_c('../Data_prep/data/wav/ISLE_SESS0003_BLOCKD01_01_sprt1.wav',0)
    # f_cav_inp = extract_f_c('../Data_prep/data/wav/ISLE_SESS0003_BLOCKD01_01_sprt1.wav',1)
    # print(len(f_c_inp),len(f_cav_inp))
    
    data_len = len(f_c_inp)


    
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
        phnStart = []
        # print(data)
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
        ## The factor helps relation between milliseconds and len of elements in the f0, c0
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
        
        return out_ph
    # print(data)
    ## population phoneme duration

    # print(out_ph)
    ## To iterate over all the files
    files  = os.listdir("../Data_prep/data/syl")
    outs = []
    data_len = 4096
    for i,file in enumerate(files):
        if i == 7: break
        data = loadmat("../Data_prep/data/syl/"+file)
        outs.append(syl_data(data,data_len))

    # print(outs)
 

    # print(len(outs))

    files  = os.listdir("../Data_prep/data/phn")
    outs_ph = []
    data_len = 4096
    for i,file in enumerate(files):
        if i == 7: break
        # print(file)
        data = loadmat("../Data_prep/data/phn/"+file)
        outs_ph.append(phn_data(data,data_len))
    # outs_ph = outs_ph.reshape(len(outs_ph)*len(outs_ph[0]),len(outs_ph[0][0]))
    # Make it an array first to reshape
    # print(outs_ph)
    outs_ph_np = np.array(outs_ph)
    reshaped_arr = outs_ph_np.reshape((outs_ph_np.shape[0] * outs_ph_np.shape[1], 1))
    outs_ph = reshaped_arr.tolist()
    # print(len(outs_ph))

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
    syl_v = []
    for i in range(len(outs)):
        syl_v.append(syl_val(sentence_vectors[i], outs[i], data_len))
        
    # p4rint((syl_v[:1]))

    syl_v_np = np.array(syl_v)
    reshaped_arr = syl_v_np.reshape((syl_v_np.shape[0] * syl_v_np.shape[1], syl_v_np.shape[2]))
    syl_v = reshaped_arr.tolist()
    # print("syl_v -- ",len(syl_v))

    outs_np = np.array(outs)
    reshaped_arr = outs_np.reshape((outs_np.shape[0] * outs_np.shape[1], 1))
    outs = reshaped_arr.tolist()
    # print("Outs -- ",len(outs))





    print(len(f_c_inp),len(f_cav_inp),len(syl_v),len(outs_ph),len(outs))
    num_samples = len(f_c_inp)
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
        phrnn_seq.append(np.hstack((f_cav_inp[i],outs_ph[i])))
    # phrnn_seq = np.concatenate((f_cav_inp,outs_ph[0]),axis =1)
    # print(phrnn_seq[:2])
    phrnn_seq = torch.tensor(phrnn_seq)
    sylrnn_data = torch.tensor(syl_v)
    seq_timesteps = torch.tensor(outs)
    dummy_targets = np.random.rand(num_samples,32)
    # model = MyModel()
    total_params = sum(p.numel() for p in chive.parameters())
    print(f"Number of parameters: {total_params}")
    # chive.summary()
    chive.train(x_train=[frnn_seq,clock_frnn, phrnn_seq,clock_phrnn,sylrnn_data,seq_timesteps], y_train=dummy_targets, batch_size=16, num_epochs=16)





## To generate random data
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
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_sequence, unpack_sequence
from cwrnn_torch import ClockworkRNNLayer
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
                #Asynchronously adding frame_rate and phone rate rnn layers
                h_frnn = self.frnn_layer(x=frnn_seq[t],h_prev = h_frnn,timestep= t, clock_val =frnn_clock[t])
                # self.layers.append(h_frnn)
                h_phrnn = self.phrnn_layer(x=phrnn_seq[t],h_prev = h_phrnn,timestep= t, clock_val =phrnn_clock[t])
                # self.layers.append( h_phrnn)
                # Passing the frame rate and phone rate rnn layers to the Syllable rate rnn layers along with linguistic features
                if sample_freq[t] == 1:
                    # print(t)
                    syl_clock = torch.randint(2, self.hidden_size-2, (1,)).item()
                    # sylrnn_inp[t] = sylrnn_inp[t].view(1)
                    
                    # syl_inp = [h_frnn,h_phrnn,sylrnn_inp[t]]
                    max_length = max((h_frnn.shape[1]), (h_phrnn.shape[1]), (sylrnn_inp[t].shape[1]))
                    # print("Lens-",(h_frnn.shape[1]), (h_phrnn.shape[1]), (sylrnn_inp[t].shape[1]))

                    h_frnn = F.pad(h_frnn, pad=(0, max_length - (h_frnn.shape[1])))
                    h_phrnn = F.pad(h_phrnn, pad=(0, max_length - (h_phrnn.shape[1])))
                    # sylrnn_inp[t] = F.pad(sylrnn_inp[t], pad=(0, max_length - (sylrnn_inp[t].shape[1])))
                    syl_pad = torch.tensor(np.zeros( max_length - (sylrnn_inp[t].shape[1])))
                    syl_inp = torch.cat((sylrnn_inp[t],syl_pad.unsqueeze(-1).t()), dim=1)
                    syl_inp = syl_inp.to(torch.float32)
                    # result_tensor = torch.stack([h_frnn, h_phrnn, syl_inp], dim=0)
                    result_tensor = h_frnn+h_phrnn+syl_inp
                    # print("result tesnor------",result_tensor.shape)
                    h_sylrnn = self.sylrnn_layer(x= result_tensor,h_prev = h_sylrnn,timestep= freq_count, clock_val =syl_clock)
                    yield h_sylrnn 

        #define bottleneck representations
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
                if i>0 and (sample_freq[i-1]==1 or h_sylrnn_dur==1):
                    # print("HI")
                    h_sylrnn_decd = torch.zeros(self.hidden_size)
                    h_phrnn_decd = torch.zeros(self.hidden_size)
                    # print(x_inp.shape)
                    clock_decd = torch.randint(2, self.hidden_size-2, (1,)).item()
                    h_sylrnn_decd = self.sylrnn_layer(x = x_inp,h_prev= h_sylrnn_decd,timestep = timestep,clock_val=clock_decd)
                    # print("Shape of this tensor",h_sylrnn_decd.shape,"Shape of input tensor--",x_inp.shape)
                    h_phrnn_decd = self.phrnn_decd(x = h_sylrnn_decd,h_prev = h_phrnn_decd,timestep=timestep,clock_val = clock_decd)
                    h_phrnn_dur = self.phrnn_dur(x =h_phrnn_decd,h_prev = h_phrnn_dur,timestep=timestep,clock_val = 1)
                    h_sylrnn_dur=  self.sylrnn_dur(x = h_sylrnn_decd,h_prev = h_sylrnn_dur,timestep=timestep,clock_val = 1)
                    # frnn_f_inp = torch.cat((h_sylrnn_decd,h_phrnn_decd),dim=0)
                h_frnn_f = self.frnn_f(x=h_phrnn_decd,h_prev=h_frnn_f,timestep=timestep,clock_val = 1)
                clock_decd_c = torch.randint(2,10, (1,)).item()
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
            
            for data in train_loader:
                frnn_batch,frnn_clock, phrnn_batch,phrnn_clock, sylrnn_batch,seq_batch, y_batch = data
                frnn_c_batch, frnn_f_batch = torch.split(frnn_batch, [12, 1], dim=2)
                _, phrnn_dur_batch = torch.split(phrnn_batch, [2,1], dim=2)
                # print("Shapes--",frnn_batch.shape,phrnn_batch.shape,seq_batch.shape)
                self.optimizer.zero_grad()
                for i in range(batch_size):
                    h_frnn_f,h_frnn_c,h_phrnn_dur,h_sylrnn_dur = self.forward([(frnn_batch,frnn_clock), (phrnn_batch,phrnn_clock), sylrnn_batch,seq_batch])
                    # print("shapes of outputs",h_frnn_c.view(12).shape,frnn_c_batch[i,:,:].view(12).shape)
                # print(output.shape,y_batch.shape)
                

                    loss_f = self.loss_function(h_frnn_f.view(1),frnn_f_batch[i,:,:].view(1))
                    loss_c = self.loss_function(h_frnn_c.view(12),frnn_c_batch[i,:,:].view(12))
                    loss_ph_dur = self.loss_function(h_phrnn_dur.view(1),phrnn_dur_batch[i,:,:].view(1))
                    loss_syl_dur = self.loss_function(h_sylrnn_dur.view(1),seq_batch[i].view(1))
                    total_loss = (loss_c+loss_f+loss_ph_dur+loss_syl_dur)/4
                total_loss = total_loss/batch_size
                total_loss.backward()
                self.optimizer.step()
            print("Epoch--",epoch+1,"| Loss --",total_loss.item())



if __name__ == "__main__":

    
    # chive.summary()

    num_samples = 64
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

    frnn_seq = torch.tensor(np.random.rand(num_samples,1, 13))
    clock_frnn = torch.tensor([np.random.uniform(1, 6) for i in range(num_samples)])
    phrnn_seq = torch.tensor(np.random.rand(num_samples,1, 3))
    clock_phrnn = torch.tensor([np.random.uniform(1, 6) for i in range(num_samples)])
    sylrnn_val = torch.tensor(np.random.rand(int(num_samples - num_samples/2),1,1))
    seq_data = generate_non_repeating_subset(0,int(num_samples), int(num_samples/2))
    seq_timesteps = torch.zeros(num_samples)
    sylrnn_data = torch.zeros_like(frnn_seq) 
    syl_seq_count = 0
    for i in range(len(seq_timesteps)):
        if i in seq_data:
            sylrnn_data[i] =  torch.tensor(np.random.rand(1,1, 13))
            seq_timesteps[i]=torch.tensor(1)
            syl_seq_count+=1
        else: seq_timesteps[i]=torch.tensor(0)
    # print(seq_timesteps)

    # print(frnn_seq)
    # print(len(frnn_seq),len(phrnn_seq),len(sylrnn_data),(seq_timesteps[99]),len(clock_frnn),len(clock_phrnn),len(sylrnn_val),len(seq_data))


    dummy_targets = np.random.rand(num_samples,32)
    # model = MyModel()
    total_params = sum(p.numel() for p in chive.parameters())
    print(f"Number of parameters: {total_params}")
    # chive.summary()
    chive.train(x_train=[frnn_seq,clock_frnn, phrnn_seq,clock_phrnn,sylrnn_data,seq_timesteps], y_train=dummy_targets, batch_size=16, num_epochs=20)

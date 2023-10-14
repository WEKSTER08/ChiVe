import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_sequence, unpack_sequence
from cwrnn_torch import ClockworkRNNLayer
import random
print("HI")
class CHIVE(nn.Module):
    def __init__(self, latent_space_dim,input_size):
        super(CHIVE, self).__init__()
        self.latent_space_dim = latent_space_dim
        self.layers = nn.ModuleList()
        
        self.hidden_size = 32
        self.frnn_layer =  ClockworkRNNLayer(input_size[0], hidden_size= self.hidden_size)
        self.phrnn_layer =  ClockworkRNNLayer(input_size[1], hidden_size= self.hidden_size)
        self.sylrnn_layer =  ClockworkRNNLayer(input_size[2], hidden_size =self.hidden_size)
        # print(self.parameters)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        

    def forward(self, x):
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
        for t in range(len(frnn_inp[0])):
            # self.frnn_layer = ClockworkRNNLayer(input_size=frnn_seq[t].size(),hidden_size=self.hidden_size,clock_val=frnn_clock[t])
            h_frnn = self.frnn_layer(x=frnn_seq[t],h_prev = h_frnn,timestep= t, clock_val =frnn_clock[t])
            # self.layers.append(h_frnn)

            # self.phrnn_layer = ClockworkRNNLayer(input_size=phrnn_seq[t].size(),hidden_size=self.hidden_size,clock_val=phrnn_clock[t])
            h_phrnn = self.phrnn_layer(x=phrnn_seq[t],h_prev = h_phrnn,timestep= t, clock_val =phrnn_clock[t])
            # self.layers.append( h_phrnn)
        
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
                
                # print((h_frnn),(h_phrnn),syl_inp)

                syl_inp = syl_inp.to(torch.float32)

                result_tensor = torch.stack([h_frnn, h_phrnn, syl_inp], dim=0)
                # syl_inp = pad_sequence(syl_inp, batch_first=True, padding_value=0.0)

                # self.sylrnn_layer = ClockworkRNNLayer(input_size=sylrnn_seq[t].size(),hidden_size=self.hidden_size,clock_val=sylrnn_clock[t])
                h_sylrnn = self.sylrnn_layer(x= result_tensor,h_prev = h_sylrnn,timestep= freq_count, clock_val =syl_clock)
                # self.layers.append(h_sylrnn)

        return h_sylrnn
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
        seq_train = torch.tensor(seq_train,dtype=torch.float32).requires_grad_(True)
        # sylrnn_train = torch.tensor(sylrnn_train, dtype=torch.float32)
        y_train = torch.tensor(y_train,dtype=torch.float32).requires_grad_(True)

        dataset = torch.utils.data.TensorDataset(frnn_train,frnn_clock, phrnn_train,phrnn_clock, sylrnn_train,seq_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.loss_function = nn.MSELoss()
        for epoch in range(num_epochs):
            
            for data in train_loader:
                frnn_batch,frnn_clock, phrnn_batch,phrnn_clock, sylrnn_batch,seq_batch, y_batch = data
                self.optimizer.zero_grad()
                output = self.forward([(frnn_batch,frnn_clock), (phrnn_batch,phrnn_clock), sylrnn_batch,seq_batch])
                loss = self.loss_function(output, y_batch)
                loss.backward()
                self.optimizer.step()
            print("Epoch--",epoch+1,"| Loss --",loss.item())



if __name__ == "__main__":
    print("HI")
    
    # chive.summary()

    num_samples = 100
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
            seq_timesteps[i]=1
            syl_seq_count+=1
    # print(seq_timesteps)


    # print(frnn_seq)
    print(len(frnn_seq),len(phrnn_seq),len(sylrnn_data),(seq_timesteps[99]),len(clock_frnn),len(clock_phrnn),len(sylrnn_val),len(seq_data))


    dummy_targets = np.random.rand(num_samples, 1)
    # model = MyModel()
    total_params = sum(p.numel() for p in chive.parameters())
    print(f"Number of parameters: {total_params}")
    # chive.summary()
    chive.train(x_train=[frnn_seq,clock_frnn, phrnn_seq,clock_phrnn,sylrnn_data,seq_timesteps], y_train=dummy_targets, batch_size=16, num_epochs=50)

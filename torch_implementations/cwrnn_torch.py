import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pprint
import random
class ClockworkRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ClockworkRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.clock_val = None

        # Initialize parameters
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.U = nn.Parameter(torch.randn(hidden_size, input_size))

        
        # Register the parameters
        self.register_parameter('W', self.W)
        self.register_parameter('U', self.U)

    def forward(self, x, h_prev, timestep,clock_val):
        # considering the weight matrix to have len(hidden_size) number of modules
        mask = []
        for i in range(self.hidden_size):
            # masking the rest of the modules except for the clocked modules
            if i%2==0 or i%clock_val == 0:
                mask.append(1) 
            else : mask.append(0)
        t_mask = torch.tensor(mask)
        # Determine which neurons to update based on the clock intervals
        h_new = torch.tanh(F.linear(h_prev, self.W * t_mask) + F.linear(x, self.U))

        return h_new

class ClockworkRNN(nn.Module):
    def __init__(self, input_size, hidden_size,clock_val):
        super(ClockworkRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        selfclock_val =clock_val

        # Create Clockwork RNN layers
        self.layer = ClockworkRNNLayer(input_size, hidden_size, clock_val)

    def forward(self, x):
        # Initialize hidden state
        h = torch.zeros(self.hidden_size)
        # print(x)
        seq_input = x[0]
        clock_val = x[1]
        

        # Process each time step
        for t in range(len(x[0])):
            h = self.layer(x=seq_input[t],h_prev = h,timestep= t, clock_val =clock_val[t])

        return h

# Generate sinusoidal dummy data
def generate_sinusoidal_data(num_points, freq=1, amplitude=1):
    t = torch.arange(0, num_points, 1)
    x = amplitude * torch.sin(2 * torch.pi * freq * t / num_points)
    return x

# # Dummy data
# input_size = 1
# hidden_size = 5
# clockwork_intervals = [1, 2, 4, 8]  # Adjust as needed

# model = ClockworkRNN(input_size, hidden_size, clockwork_intervals)
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# # Training loop
# num_epochs = 5
# for epoch in range(num_epochs):
#     # Generate sinusoidal input and target data
#     input_sequence = generate_sinusoidal_data(50)
#     clock_vals = torch.tensor([random.uniform(1, 6) for _ in range(50)])
#     target_sequence = generate_sinusoidal_data(64)

#     # Reshape input for the model
#     input_sequence = input_sequence.unsqueeze(1).unsqueeze(1)  # Add batch and sequence length dimensions
#     clock_vals = clock_vals.unsqueeze(1).unsqueeze(1)
#     input_seq = [input_sequence,clock_vals]

#     # Forward pass
#     output = model(input_seq)
#     # print(target_sequence[:5])
#     # Compute loss
#     loss = criterion(output[0], target_sequence[0])

#     # Backward pass and optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # Test the trained model
# test_input = generate_sinusoidal_data(10)
# test_input = test_input.unsqueeze(1).unsqueeze(1)
# predicted_output = model(test_input)
# print("Test Input:", test_input.squeeze())
# print("Predicted Output:", predicted_output.squeeze().detach())

# # Plot the results
# plt.plot(target_sequence, label='Target')
# plt.plot(output.detach().numpy(), label='Predicted')
# plt.savefig('Test.png')

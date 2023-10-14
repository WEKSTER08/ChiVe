import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

class ClockworkRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, clock_val):
        super(ClockworkRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.clock_val = clock_val

        # Initialize parameters
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.U = nn.Parameter(torch.randn(hidden_size, input_size))

    def forward(self, x, h_prev, timestep, clock_val):
        # considering the weight matrix to have len(hidden_size) number of modules
        mask = []
        for i in range(self.hidden_size):
            # masking the rest of the modules except for the clocked modules
            if i % 2 == 0 or i % clock_val == 0:
                mask.append(1)
            else:
                mask.append(0)
        t_mask = torch.FloatTensor(mask)
        # Determine which neurons to update based on the clock intervals
        h_new = torch.tanh(F.linear(h_prev, self.W * t_mask) + F.linear(x, self.U))
        return h_new

class ClockworkRNN(nn.Module):
    def __init__(self, input_size, hidden_size, clock_val):
        super(ClockworkRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.clock_val = clock_val

        # Create Clockwork RNN layers
        self.layer = ClockworkRNNLayer(input_size, hidden_size, clock_val)

    def forward(self, x):
        # Initialize hidden state
        h = torch.zeros(self.hidden_size)

        seq_input = x[0]
        clock_val = x[1]

        # Process each time step
        for t in range(len(x[0])):

            # print(t)
            # break
            h = self.layer(x=seq_input[t], h_prev=h, timestep=t, clock_val=clock_val[t])

        return h
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

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    def summary(self):
        print("\nModel Summary:")
        print("=" * 50)
        print("LSTM Model:")
        print("-" * 30)
        total_params = 0
        for name, param in self.named_parameters():
            print(f"{name}: {param.size()}")
            total_params += param.numel()
        print("=" * 50)
        print(f"Total Trainable Parameters: {total_params}")

# Generate sinusoidal dummy data
def generate_sinusoidal_data(num_points, freq=1, amplitude=1):
    t = torch.arange(0, num_points, 1)
    x = amplitude * torch.sin(2 * torch.pi * freq * t / num_points)
    return x

# Dummy data
input_size = 1
hidden_size = 5
clockwork_intervals = [1, 2, 4, 8]  # Adjust as needed

# Clockwork RNN model
clockwork_model = ClockworkRNN(input_size, hidden_size, clockwork_intervals)
clockwork_criterion = nn.MSELoss()
clockwork_optimizer = optim.SGD(clockwork_model.parameters(), lr=0.01)

# LSTM model
lstm_model = LSTMModel(input_size, hidden_size)
lstm_criterion = nn.MSELoss()
lstm_optimizer = optim.SGD(lstm_model.parameters(), lr=0.01)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    if epoch > 500 : break
    # Generate sinusoidal input and target data
    input_sequence = generate_sinusoidal_data(50)
    clock_vals = torch.tensor([random.uniform(1, 6) for _ in range(50)])
    target_sequence = generate_sinusoidal_data(64)

    # Reshape input for the models
    input_sequence = input_sequence.unsqueeze(1).unsqueeze(1)
    clock_vals = clock_vals.unsqueeze(1).unsqueeze(1)
    input_seq = [input_sequence, clock_vals]

    # Clockwork RNN training
    clockwork_optimizer.zero_grad()
    clockwork_output = clockwork_model(input_seq)
    clockwork_loss = clockwork_criterion(clockwork_output[0], target_sequence[0])
    clockwork_loss.backward()
    clockwork_optimizer.step()

    # LSTM training
    lstm_optimizer.zero_grad()
    lstm_input = input_sequence.repeat(1, len(clockwork_intervals), 1)
    lstm_output = lstm_model(lstm_input)
    lstm_loss = lstm_criterion(lstm_output, target_sequence)
    lstm_loss.backward()
    lstm_optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Clockwork RNN Loss: {clockwork_loss.item():.4f}, LSTM Loss: {lstm_loss.item():.4f}')

# Test the trained models
test_input = generate_sinusoidal_data(10)
test_input = test_input.unsqueeze(1).unsqueeze(1)
clock_vals = torch.tensor([random.uniform(1, 6) for _ in range(10)])
clock_vals = clock_vals.unsqueeze(1).unsqueeze(1)

# Clockwork RNN
clockwork_predicted_output = clockwork_model([test_input, torch.tensor(clock_vals)])
print("Clockwork RNN Test Input:", test_input.squeeze())
print("Clockwork RNN Predicted Output:", clockwork_predicted_output.squeeze().detach())

# LSTM
lstm_input = test_input.repeat(1, len(clockwork_intervals), 1)
lstm_predicted_output = lstm_model(lstm_input)
print("LSTM Test Input:", test_input.squeeze())
print("LSTM Predicted Output:", lstm_predicted_output.squeeze().detach())

clockwork_model.summary()
lstm_model.summary()

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(target_sequence, label='Target')
plt.plot(clockwork_predicted_output.detach().numpy(), label='Clockwork RNN Predicted')
plt.plot(lstm_predicted_output.detach().numpy(), label='LSTM Predicted')
plt.legend()
plt.show()

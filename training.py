import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

# Kraków
file = open(file="Kraków_data.csv", mode="r")
data_kraków = file.read()
file.close()

data_kraków = data_kraków.split()[8:-365:]
print(len(data_kraków))
timestamps = [str(day.split(",")[0]) for day in data_kraków]
data_kraków = [[float(val) for val in (day.split(",")[1::])] for day in data_kraków]

# Rzeszów
file = open(file="Rzeszów_data.csv", mode="r")
data_rzeszów = file.read()
file.close()

data_rzeszów = data_rzeszów.split()[8:-365:]
data_rzeszów = [[float(val) for val in (day.split(",")[1::])] for day in data_rzeszów]

# Warszawa
file = open(file="Warszawa_data.csv", mode="r")
data_warszawa = file.read()
file.close()

data_warszawa = data_warszawa.split()[8:-365:]
data_warszawa = [[float(val) for val in (day.split(",")[1::])] for day in data_warszawa]

# Praga
file = open(file="Prague_data.csv", mode="r")
data_prague = file.read()
file.close()

data_prague = data_prague.split()[8:-365:]
data_prague = [[float(val) for val in (day.split(",")[1::])] for day in data_prague]

# Combine data
data = [data_kraków[i] + data_rzeszów[i] + data_warszawa[i] + data_prague[i] for i in range(len(data_kraków))]
print(data[0])

# neural network
days_n = 14
features_n = 5
cities_n = 4
input_n = days_n * features_n * cities_n
hidden_n = 512
output_n = features_n * cities_n

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(input_n, hidden_n)
        self.hidden_layer1 = nn.Linear(hidden_n, hidden_n)
        self.hidden_layer2 = nn.Linear(hidden_n, hidden_n)
        self.hidden_layer3 = nn.Linear(hidden_n, hidden_n)
        self.hidden_layer4 = nn.Linear(hidden_n, hidden_n)
        self.output_layer = nn.Linear(hidden_n, output_n)

    def forward(self, x):
        activation = nn.LeakyReLU()
        x = nn.Flatten(start_dim=0)(x)
        x = self.input_layer(x)
        x = activation(x)

        skip = x
        x = self.hidden_layer1(x)
        x = activation(x)
        x = self.hidden_layer2(x)
        x += skip
        x = activation(x)
        
        skip = x
        x = self.hidden_layer3(x)
        x = activation(x)
        x = self.hidden_layer4(x)
        x += skip
        x = activation(x)

        x = self.output_layer(x)
        return x
    

model = Net()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.00001, momentum=0.9)
criterion = nn.MSELoss()
t0 = time.time()

loss_sum = 0
rounds_n = 1000000
small_round_n = 5000
for i in range(rounds_n):
    day_id = torch.randint(0, len(data) - days_n, (1, ))
    inputs = torch.tensor(data[day_id:day_id + days_n], dtype=torch.float32)
    outputs = model.forward(inputs)
    target = torch.tensor([data[day_id + days_n]])

    optimizer.zero_grad()
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()
    loss_sum += loss

    if i % small_round_n == 0:
        print("Progres:", i / rounds_n, "Loss:", loss_sum / small_round_n)
        loss_sum = 0

print(time.time() - t0)
torch.save(model, "model.pt")

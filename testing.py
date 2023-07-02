import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import numpy

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

class Net(nn.Module):
    def __init__(self):
        hidden_n = 0
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
    

model = torch.load("model.pt")
days_n = 14

diffs = []
good = 0
for day in range(len(data) - days_n):
    day_id = day
    inputs = torch.tensor(data[day_id:day_id + days_n], dtype=torch.float32)
    outputs = model.forward(inputs)
    target = torch.tensor([data[day_id + days_n]])
    diff = torch.abs(target - outputs)
    if diff[0][2] < 3:
        good += 1
    diffs.append(float(diff[0][2]))
    
print(numpy.std(diffs))
print(numpy.quantile(diffs, q = [0.25, 0.5, 0.75, 0.9]))
print(good / (len(data) - days_n))

while 2137:
    day_id = torch.randint(0, len(data) - days_n, (1, ))
    inputs = torch.tensor(data[day_id:day_id + days_n], dtype=torch.float32)
    outputs = model.forward(inputs)
    print("prediction:", outputs)
    target = torch.tensor([data[day_id + days_n]])
    print("real:", target)
    diff = target - outputs
    print("difference:", diff[0])
    input()

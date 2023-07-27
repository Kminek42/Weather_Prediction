import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import learning_time_est as lte
import time
import weather_dataset as wd
import matplotlib.pyplot as plt

dev = torch.device("cpu")

if torch.cuda.is_available():
    dev = torch.device("cuda")

elif torch.backends.mps.is_available():
    dev = torch.device("mps")

print(dev)

train = True

if train:
    training_dataset = wd.Weather_Dataset(input_days_n=7, output_days_n=1, training=True, split=0.5, locations=[
        "./locations/Amsterdam_[NET].csv",
        "./locations/Barcelona_[SPA].csv",
        "./locations/Berlin_[GER].csv",
        "./locations/Bucharest_[ROM].csv",
        "./locations/Budapest_[HUN].csv",
        "./locations/Cologne_[GER].csv",
        "./locations/Hamburg_[GER].csv",
        "./locations/Kraków_[POL].csv",
    ])

    training_loader = DataLoader(
        dataset=training_dataset,
        batch_size=64,
        shuffle=True)

    locations_n = 8
    features = 3

    activation = nn.ReLU()
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(locations_n * features * training_dataset.input_hours, 2048),
        activation,
        nn.Linear(2048, 1024),
        activation,
        nn.Linear(1024, locations_n * features * training_dataset.output_hours),
        nn.Unflatten(dim=1, unflattened_size=(locations_n * features, training_dataset.output_hours))
    ).to(device=dev)

    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    epoch_n = 10
    t0 = time.time()
    loss_file = open("loss.txt", mode="w")
    loss_file.close()
    for epoch in range(1, epoch_n + 1):
        loss_sum = 0
        i = 0
        for data_in, target, mean, std in iter(training_loader):
            data_in = data_in.to(dev)
            target = target.to(dev)
            prediction = model.forward(data_in)

            optimizer.zero_grad()
            loss = criterion(target, prediction)
            loss.backward()
            optimizer.step()
            loss_sum += loss
            i += 1

        print("\nmean loss:", float(loss_sum / i))
        loss_file = open("loss.txt", mode="a")
        loss_file.write(str(float(loss_sum / i)) + "\n")
        loss_file.close()
        lte.show_time(start_timestamp=t0, progres=epoch/epoch_n)

    torch.save(obj=model.to("cpu"), f="model.pt")

else:
    test_dataset = wd.Weather_Dataset(input_days_n=7, output_days_n=1, training=False, split=0.5, locations=[
        "./locations/Amsterdam_[NET].csv",
        "./locations/Barcelona_[SPA].csv",
        "./locations/Berlin_[GER].csv",
        "./locations/Bucharest_[ROM].csv",
        "./locations/Budapest_[HUN].csv",
        "./locations/Cologne_[GER].csv",
        "./locations/Hamburg_[GER].csv",
        "./locations/Kraków_[POL].csv",
    ])

    training_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=True)

    model = torch.load(f="model.pt")

    print(model)

    for data_in, target, mean, std in iter(training_loader):
        prediction = model.forward(data_in)

        target[0][0] *= std[0]
        prediction[0][0] *= std[0]

        target[0][0] += mean[0]
        prediction[0][0] += mean[0]
        
        plt.plot(target[0][0].detach().numpy())
        plt.plot(prediction[0][0].detach().numpy())
        plt.show()

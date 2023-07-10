import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import location_data
import learning_time_est as lte
import time

timestamps, datasets = location_data.get_all_locations(filenames=
                                ["./locations/Amsterdam_[NET].csv",
                                 "./locations/Barcelona_[SPA].csv",
                                 "./locations/Berlin_[GER].csv",
                                 "./locations/Bucharest_[ROM].csv",
                                 "./locations/Budapest_[HUN].csv",
                                 "./locations/Cologne_[GER].csv",
                                 "./locations/Hamburg_[GER].csv",
                                 "./locations/Kraków_[POL].csv",
                                 "./locations/Madrid_[SPA].csv",
                                 "./locations/Marseille_[FRA].csv",
                                 "./locations/Milan_[ITA].csv",
                                 "./locations/Munich_[GER].csv",
                                 "./locations/Naples_[ITA].csv",
                                 "./locations/Paris_[FRA].csv",
                                 "./locations/Prague_[CZE].csv",
                                 "./locations/Rome_[ITA].csv",
                                 "./locations/Sofia_[BUL].csv",
                                 "./locations/Stockholm_[SWE].csv",
                                 "./locations/Turin_[ITA].csv",
                                 "./locations/Vienna_[AUS].csv",
                                 "./locations/Warsaw_[POL].csv",
                                 ])

class Weather_Dataset(Dataset):
    def __init__(self, data, days_n, training=True):
        super().__init__()
        l = (len(datasets))
        split_id = int(l * 0.7)
        self.hours = days_n * 24
        if training:
            self.samples = data[0:split_id]
        else:
            self.samples = data[split_id:]


    def __getitem__(self, index):
        input_days = torch.tensor(self.samples[index : index + self.hours])
        days_to_predict = torch.tensor(self.samples[index + self.hours : index + 2 * self.hours])
        return input_days, days_to_predict

    def __len__(self):
        return len(self.samples) - 2 * self.hours
    
training_dataset = Weather_Dataset(datasets, days_n=7)
training_loader = DataLoader(
    dataset=training_dataset,
    batch_size=2048,
    shuffle=True)

dev = torch.device("mps")
locations_n = 21
features = 3
hours = 7 * 24

activation = nn.Sigmoid()
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(locations_n * features * hours, 1024),
    activation,
    nn.Linear(1024, 1024),
    activation,
    nn.Linear(1024, 1024),
    activation,
    nn.Linear(1024, locations_n * features * hours),
    nn.Unflatten(dim=1, unflattened_size=(hours, locations_n * features))
).to(device=dev)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-2, momentum=0.9)

epoch_n = 10
print("checkpoint")
t0 = time.time()
loss_file = open("loss.txt", mode="w")
loss_file.close()
for epoch in range(1, epoch_n + 1):
    loss_sum = 0
    i = 0
    for data_in, target in iter(training_loader):
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


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import location_data
import learning_time_est as lte
import time

timestamps, datasets = location_data.get_all_locations(filenames=
                                ["./locations/Kraków_[PL].csv", 
                                 "./locations/Katowice_[PL].csv", 
                                 "./locations/Kielce_[PL].csv", 
                                 "./locations/Nowy_Sącz_[PL].csv"])

class Weather_Dataset(Dataset):
    def __init__(self, data, training=True):
        super().__init__()
        l = (len(datasets))
        split_id = int(l * 0.7)
        
        if training:
            self.samples = data[0:split_id]
        else:
            self.samples = data[split_id:]


    def __getitem__(self, index):
        input_days = torch.tensor(self.samples[index : index + 24])
        days_to_predict = torch.tensor(self.samples[index + 24 : index + 48])
        return input_days, days_to_predict

    def __len__(self):
        return len(self.samples) - 50
    

training_dataset = Weather_Dataset(datasets)
training_loader = DataLoader(
    dataset=training_dataset,
    batch_size=128,
    shuffle=True)

dev = torch.device("cpu")
model = nn.Sequential(
    nn.Linear(12, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 12)
).to(device=dev)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9)

epoch_n = 10

t0 = time.time()

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
    lte.show_time(start_timestamp=t0, progres=epoch/epoch_n)

torch.save(obj=model.to("cpu"), f="model.pt")

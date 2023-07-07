import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import location_data
import matplotlib.pyplot as plt
import numpy as np

timestamps, datasets = location_data.get_all_locations(filenames=
                                ["./locations/Kraków_[PL].csv", 
                                 "./locations/Warsaw_[Pl].csv", 
                                 "./locations/Berlin_[DE].csv", 
                                 "./locations/Budapest_[HU].csv", 
                                 "./locations/Lviv_[UKR].csv", 
                                 "./locations/Prague_[CZ].csv", 
                                 "./locations/Vilnius_[LT].csv", ])

class Weather_Dataset(Dataset):
    def __init__(self, data, training=True, days_n=2):
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
    
training_dataset = Weather_Dataset(datasets, training=False, days_n=7)
training_loader = DataLoader(
    dataset=training_dataset,
    batch_size=128,
    shuffle=False)

dev = torch.device("mps")
model = torch.load(f="model.pt").to(dev)
criterion = nn.MSELoss()

loss_sum = 0
diff = []
hour_to_test = 2 * 24
for data_in, target in iter(training_loader):
    data_in = data_in.to(dev)
    target = target.to(dev)
    prediction = model.forward(data_in)
    loss = criterion(target, prediction)
    diffs = target - prediction
    for d in diffs:
        diff.append(float(d[hour_to_test][0]))

plt.hist(diff, bins=64, density=True)
plt.title("+48 hour prediction")
plt.xlabel("Temperature difference [C]")
plt.show()

print(np.quantile(a=diff, q=[0.1, 0.25, 0.5, 0.75, 0.9]))
print(np.mean(diff))
print(np.std(diff))

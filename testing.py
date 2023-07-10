import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import location_data
import matplotlib.pyplot as plt
import numpy as np

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
    
training_dataset = Weather_Dataset(datasets, days_n=7, training=False)
training_loader = DataLoader(
    dataset=training_dataset,
    batch_size=128,
    shuffle=False)

dev = torch.device("mps")
model = torch.load(f="model.pt").to(dev)
criterion = nn.MSELoss()

loss_sum = 0
diff = []
hour_to_test = 0 * 24
i = 0
for data_in, target in iter(training_loader):
    data_in = data_in.to(dev)
    target = target.to(dev)
    prediction = model.forward(data_in)
    loss = criterion(target, prediction)
    diffs = target - prediction
    for d in diffs:
        diff.append(float(d[hour_to_test][0]))

    i += 1
    print(i)


plt.hist(diff, bins=64, density=True)
plt.title("+48 hour prediction")
plt.xlabel("Temperature difference [C]")
plt.show()

print(np.quantile(a=diff, q=[0.1, 0.25, 0.5, 0.75, 0.9]))
print(np.mean(diff))
print(np.std(diff))

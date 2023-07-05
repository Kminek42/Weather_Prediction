import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import location_data
import matplotlib.pyplot as plt

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
    

training_dataset = Weather_Dataset(datasets, training=False)
training_loader = DataLoader(
    dataset=training_dataset,
    batch_size=1,
    shuffle=False)

model = torch.load(f="model.pt")
criterion = nn.MSELoss()

loss_sum = 0
i = 0
diff = []
i = 0
for data_in, target in iter(training_loader):
    prediction = model.forward(data_in)
    loss = criterion(target, prediction)
    diff.append(float(target[0][23][0] - prediction[0][23][0]))
    i += 1
    print(i)

plt.hist(diff, bins=40)
plt.show()

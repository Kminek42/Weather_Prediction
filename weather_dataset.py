import torch
import location_data
import matplotlib.pyplot as plt

class Weather_Dataset(torch.utils.data.Dataset):
    def __init__(self, *, input_days_n, output_days_n, split, training, locations):
        super().__init__()
        timestamps, data = location_data.get_all_locations(filenames=locations)
        l = (len(data))
        split_id = int(l * split)
        self.input_hours = input_days_n * 24
        self.output_hours = output_days_n * 24
        if training:
            self.samples = data[0:split_id]
            self.timestamps = timestamps[0:split_id]
        else:
            self.samples = data[split_id:]
            self.timestamps = timestamps[split_id:]


    def __getitem__(self, index):
        # timestamps -----------------------------------------------------------
        timestamps_in = self.timestamps[index : index + self.input_hours]
        timestamps_out = self.timestamps[index + self.input_hours : index + self.input_hours + self.output_hours]

        # input data -----------------------------------------------------------
        input_days = torch.tensor(self.samples[index : index + self.input_hours])
        input_days = torch.t(input_days)

        # output data -----------------------------------------------------------
        output_days = torch.tensor(self.samples[index + self.input_hours : index + self.input_hours + self.output_hours])
        output_days = torch.t(output_days)
    
        # norm temperatures
        mean = 0
        std = 0
        for row in range(0, len(input_days), 3):
            mean += torch.mean(input_days[row])
            std += torch.std(input_days[row])

        mean /= (len(input_days) // 3)
        std /= (len(input_days) // 3)

        for row in range(0, len(input_days), 3):
            input_days[row] = (input_days[row] - mean) / std
            output_days[row] = (output_days[row] - mean) / std
        

        return input_days, output_days, mean, std

    def __len__(self):
        return len(self.samples) - self.input_hours - self.output_hours

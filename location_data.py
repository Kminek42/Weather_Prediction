import numpy as np

def get_location_data(*, filename):
    file = open(file=filename, mode="r")
    data = file.read().split()[6:]
    file.close()
    timestamps = [sample.split(sep=",")[0] for sample in data]
    samples = [[float(num) for num in sample.split(sep=",")[1:]] for sample in data]
    samples = [[sample[0], sample[1] / 100, sample[2] / 1000] for sample in samples]
    return timestamps, samples

def get_all_locations(*, filenames):
    t, data = get_location_data(filename=filenames[0])
    final_data = [[] for i in data]
    for filename in filenames:
        t, data = get_location_data(filename=filename)
        final_data = [final_data[i] + data[i] for i in range(len(final_data))]

    return t, final_data

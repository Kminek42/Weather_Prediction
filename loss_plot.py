import matplotlib.pyplot as plt

file = open("loss.txt")
text = file.read().split("\n")
file.close
text = [float(num) for num in text if len(num)]
print(text)
plt.plot(text)
plt.ylim(bottom=0)
plt.show()

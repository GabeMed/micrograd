from neural_network.network import NN
from matplotlib import pyplot as plt
from numpy import random
from sklearn.datasets import make_moons, make_blobs

random.seed(42)

X, y = make_moons(n_samples=100, noise=0.1)

y = y * 2 - 1  # Working with 1 and -1

model = NN(2, [16, 16, 1])  # 2-layer neural network

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap="jet")
plt.show()

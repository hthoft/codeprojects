from scipy.stats import qmc
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def closest_three_nodes(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argpartition(dist_2, 4)[:4]

sampler = qmc.Halton(d=2, scramble=True)
sample = sampler.random(n=10)

x_val = []
y_val = []

for i, x_values in enumerate(sample):
    x_val.append(sample[i][0])

for i, y_values in enumerate(sample):
    y_val.append(sample[i][1])

for i, point in enumerate(sample):
    k_nearest = closest_three_nodes([point[0],point[1]], sample)
    for j, neighbour in enumerate(k_nearest):
        x_values = [point[0], sample[neighbour][0]]
        y_values = [point[1], sample[neighbour][1]]
        plt.plot(x_values, y_values, 'g', linestyle="-")


plt.scatter(x_val, y_val, marker='o')

## Find the coordinate closest to 0, for us to begin with:
first_coordinate = sample[closest_node([0,0], sample)]
plt.scatter(first_coordinate[0], first_coordinate[1], marker='X', s=200)

last_coordinate = sample[closest_node([1,1], sample)]
plt.scatter(last_coordinate[0], last_coordinate[1], marker='X', s=200)

plt.show()

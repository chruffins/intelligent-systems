from typing import Any, Iterable
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time

PIXELDATA = "data/MNISTnumImages5000_balanced.txt"
LABELDATA = "data/MNISTnumLabels5000_balanced.txt"

class Image:
    def __init__(self, data, label: int) -> None:
        self.data = data
        self.label = label
        self.y = label_to_arr(label)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def label_to_arr(label: int):
    arr = np.zeros(10)
    arr[label] = 1
    return arr

def get_error_fraction(results: list[tuple[int, int]]) -> float:
    return 1 - (sum(map(lambda x: x[0] == x[1] and 1 or 0, results)) / len(results))

class Layer:
    def __init__(self, neurons: int, inputs: int, activation_funcs: list = None) -> None:
        # xavier initialization?
        stdev = np.sqrt(6 / (inputs + neurons))
        self.neurons: np.ndarray = np.random.rand(neurons, inputs) * stdev - stdev # 150 neurons x 725 weights
        self.input_size = inputs
        self.last_delta = None

        if activation_funcs is None:
            self.activation_funcs = [sigmoid for _ in range(neurons)]

    def process(self, data, raw: bool = False) -> np.ndarray:
        # assert len(data) == self.input_size
        # no wait its 150 neurons x 784 weights TIMES 784 inputs x 1 output -> 150 x 1 output
        result: np.ndarray = np.matmul(self.neurons, data)
        #print(result)
        if raw:
            return result
        else:
            return sigmoid(result)
    
    def adjust(self, delta: np.ndarray, momentum_alpha: float = 0):
        self.neurons += delta
        if momentum_alpha and self.last_delta is not None:
            self.neurons += (momentum_alpha * self.last_delta)
        self.last_delta = delta

def get_ij(shape: tuple[int, int], index: int) -> np.ndarray:
    return np.array((index // shape[0], index % shape[0]))

def compute_ij(shape: tuple[int, int]) -> np.ndarray:
    listresult = list()
    for i in range(shape[0] * shape[1]):
        listresult.append((i // shape[0], i % shape[0]))
    return np.array(listresult)

def get_euclidean_distance(arr1: np.ndarray, arr2: np.ndarray) -> float:
    return np.sum(np.power((arr1 - arr2), 2))

def get_neighborhood(i_star: np.ndarray, theta: float, tau: float):
    def neighborhood_func(i: np.ndarray, t: float) -> float:
        def thetaf(t):
            return theta * np.exp(-t/tau)
        return np.exp(-(np.dot(i - i_star, i - i_star))
                        / (2 * np.power(thetaf(t), 2)))
    return neighborhood_func

class SOFM(Layer):
    def __init__(self, shape: tuple[int, int], inputs: int, eta: float, activation_funcs: list = None) -> None:
        self.shape = shape
        self.length = shape[0]
        self.height = shape[1]
        self.eta = eta
        self.t = 0
        self.coords = compute_ij(shape)
        self.neighborhood_cache: dict[int, np.ndarray] = dict()
        super().__init__(self.length * self.height, inputs, activation_funcs)

        self.neurons: np.ndarray = np.random.rand(self.length * self.height, inputs)# 150 neurons x 725 weights

    def process(self, data, raw: bool = False) -> np.ndarray:
        presented = self.neurons - data[1:]
        min_index, min_result = 0, 9001 # it's over 9000
        for i, row in enumerate(presented):
            result = np.sum(np.power(row, 2))
            if result < min_result:
                min_index = i
                min_result = result

        if raw:
            return self.coords[min_index]
        else:
            result = np.zeros(self.length * self.height)
            result[min_index] = 1
            return result

    def train(self, data):
        # data will be a 784 vector

        presented = self.neurons - data[1:]
        min_index, min_result = 0, 9001 # it's over 9000
        for i, row in enumerate(presented):
            result = np.sum(np.power(row, 2))
            if result < min_result:
                min_index = i
                min_result = result
                
        winner = self.coords[min_index]
        neighbor_func = get_neighborhood(winner, 3, 250)

        delta_weights = self.eta * -presented * np.array([neighbor_func(i, self.t) for i in self.coords]).reshape(-1, 1)
        # for i, row in enumerate(presented):
            # r_i = get_ij(self.neurons.shape, i)
            # learning = neighbor_func(r_i, self.t)
            # delta_weights[i, ] = self.eta * learning * row
        
        self.neurons += delta_weights

def problem_two(training_set: list[Image], testing_set: list[Image]):
    test_results = np.zeros((10, 12, 12))

    if not os.path.exists("sofmtest.txt") or not os.path.exists("sofmdata.npy"):
        sofm = SOFM((12, 12), 784, eta=0.075)

        start = time.time()
        for epoch in range(250):
            sofm.t = epoch
            for td in training_set:
                sofm.train(td.data)
            
            """
            if epoch % 5 == 6:
                fig2, axes = plt.subplots(12, 12, figsize=(12, 12))
                for i in range(12):
                    for j in range(12):
                        ax = axes[i, j]
                        k = (i * 12) + j

                        ax.imshow(sofm.neurons[k, ].reshape((28, 28), order='F'), cmap='gray')
                plt.show()
            """
            print(f"finished epoch {epoch}")
        end = time.time()

        print(f"completed in {end - start}s")

        np.save("sofmdata", sofm.neurons)
        fp = open("sofmtest.txt", 'w')
        for tp in testing_set:
            result = sofm.process(tp.data, raw=True)
            test_results[tp.label, result[0], result[1]] += 1
            fp.write(f"{tp.label}\t{result[0]}\t{result[1]}\n")
        fp.close()
    else:
        sofm = SOFM((12, 12), 784, eta=0.075)
        sofm.neurons = np.load("sofmdata.npy")

        for tp in testing_set:
            result = sofm.process(tp.data, raw=True)
            test_results[tp.label, result[0], result[1]] += 1

    test_results /= 100
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(2):
        for j in range(5):
            ax = axes[i, j]
            k = (i * 5) + j

            ax.set_title(f"Class {k} Heatmap")
            ax.imshow(test_results[k, ], cmap='gray')
    
    fig2, axes = plt.subplots(12, 12, figsize=(12, 12))
    for i in range(12):
        for j in range(12):
            ax = axes[i, j]
            k = (i * 12) + j

            ax.imshow(sofm.neurons[k, ].reshape((28, 28), order='F'), cmap='gray')

    plt.show()

def read_data() -> list[Image]:
    images: list[Image] = list()

    pixelfp = open(PIXELDATA)
    labelfp = open(LABELDATA)

    while True:
        pixelline = pixelfp.readline()
        labelline = labelfp.readline()

        if not pixelline or not labelline:
            break

        labelline = int(labelline.rstrip('\n'))

        pixelline = [1] + list(map(float, pixelline.rstrip('\n').split('\t')))
        pixeldata = np.array(pixelline, dtype=np.float64)

        images.append(Image(pixeldata, labelline))
    
    pixelfp.close()
    labelfp.close()

    return images

if __name__ == "__main__":
    random.seed(7)
    np.random.seed(2401)

    images = read_data()
    training_set: list[Image] = list()
    testing_set: list[Image] = list()
    for i in range(10):
        training_set += images[(i*500)+100:(i*500)+500]
        testing_set += images[(i*500):(i*500)+100]
    random.shuffle(training_set)

    problem_two(training_set, testing_set)
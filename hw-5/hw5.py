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

# input layer does not need to be represented...
class FeedForwardNetwork:
    def __init__(self, eta: float, layers: list[Layer]) -> None:
        self.eta = eta
        self.layers: list[Layer] = layers
    
    def forward_pass(self, data) -> np.ndarray:
        current_data: np.ndarray = data

        for layer in self.layers:
            current_data = layer.process(current_data)

        return current_data
    
    # train function basically
    def backpropagate(self, data: np.ndarray, y: np.ndarray, label: int, 
                      results_container: list[tuple[int, int]] = list(), # by reference
                      momentum_alpha: float = 0):
        current_data: np.ndarray = data

        hidden_raw_outputs: list[np.ndarray] = list()
        hidden_outputs: list[np.ndarray] = [data]

        raw_output: np.ndarray
        for layer in self.layers:
            current_data = layer.process(current_data, raw=True)
            hidden_raw_outputs.append(current_data)
            current_data = sigmoid(current_data)
            hidden_outputs.append(current_data)

        raw_output = hidden_raw_outputs[-1]

        if results_container is not None:
            results_container.append((label, np.argmax(current_data)))

        # set up the equations for dJdw for output layer
        # current_data is equivalent to yhat right now
        # combine dJdy and dyds
        dJds: np.ndarray = (y - current_data) * sigmoid_derivative(raw_output)
        dsdw: np.ndarray = hidden_outputs[-2]

        dJdw: np.ndarray = np.outer(dJds, dsdw)

        # change in weights
        self.layers[-1].adjust(self.eta * dJdw, momentum_alpha)

        # dJds is reused. this prepares it to become dJds for next layer
        dJds = self.layers[-1].neurons.transpose() @ dJds
        # dJdw_h: np.ndarray
        for s, layer in zip(reversed(hidden_raw_outputs[:-1]), self.layers[:-1]):
            # still need to setup dJds.
            # in presentation, it is currently \sum_i w_{ij} \delta^q_i

            # now, we can reuse it for the next layer as necessary
            # although now in the presentation this now denotes delta of hidden layer
            dJds = dJds * sigmoid_derivative(s)

            layer.adjust(self.eta * np.outer(dJds, data), momentum_alpha)
            
            # prepare for next layer...
            dJds = layer.neurons.transpose() @ dJds

        return

# gradient descent is -\eta dJ/dw_{ij}
# with momentum: that + \alpha * old delta
def j2(y: np.ndarray, yhat: np.ndarray):
    return np.sum((y - yhat) ** 2) / 2

class C1FFNetwork(FeedForwardNetwork):
    def __init__(self, eta: float, layers: list[Layer]) -> None:
        super().__init__(eta, layers)

    # only trains the last layer
    def backpropagate(self, data: np.ndarray, y: np.ndarray, label: int, 
                      results_container: list[tuple[int, int]] = list(), # by reference
                      momentum_alpha: float = 0):
        current_data: np.ndarray = data

        hidden_raw_outputs: list[np.ndarray] = list()
        hidden_outputs: list[np.ndarray] = [data]

        raw_output: np.ndarray
        for layer in self.layers:
            current_data = layer.process(current_data, raw=True)
            hidden_raw_outputs.append(current_data)
            current_data = sigmoid(current_data)
            hidden_outputs.append(current_data)

        raw_output = hidden_raw_outputs[-1]

        if results_container is not None:
            results_container.append((label, np.argmax(current_data)))

        # set up the equations for dJdw for output layer
        # current_data is equivalent to yhat right now
        # combine dJdy and dyds
        dJds: np.ndarray = (y - current_data) * sigmoid_derivative(raw_output)
        dsdw: np.ndarray = hidden_outputs[-2]

        dJdw: np.ndarray = np.outer(dJds, dsdw)

        # change in weights
        self.layers[-1].adjust(self.eta * dJdw, momentum_alpha)
        return
    
    # for an entire epoch
    def train(self, data: list[Image], momentum_alpha: float = 0.5) -> tuple[int, int]:
        results: list[tuple[int, int]] = list()

        for image in data:
            self.backpropagate(image.data, image.y, image.label, results, momentum_alpha)

        return results
    
    def test(self, data: list[Image]) -> list[tuple[int, int]]:
        results = list()

        for image in data:
            yhat_label = np.argmax(self.forward_pass(image.data))
            results.append((image.label, yhat_label))
        
        return results

class C2FFNetwork(FeedForwardNetwork):
    def __init__(self, eta: float, layers: list[Layer]) -> None:
        super().__init__(eta, layers)
    
    # for an entire epoch
    def train(self, data: list[Image], momentum_alpha: float = 0.5) -> tuple[int, int]:
        results: list[tuple[int, int]] = list()

        for image in data:
            self.backpropagate(image.data, image.y, image.label, results, momentum_alpha)

        return results
    
    def test(self, data: list[Image]) -> list[tuple[int, int]]:
        results = list()

        for image in data:
            yhat_label = np.argmax(self.forward_pass(image.data))
            results.append((image.label, yhat_label))
        
        return results

class P3FFNetwork(C1FFNetwork):
    # only train the last layer
    def backpropagate(self, data: np.ndarray, y: np.ndarray, label: int, 
                      results_container: list[tuple[int, int]] = list(), # by reference
                      momentum_alpha: float = 0):
        current_data: np.ndarray = data

        
        hidden_output: np.ndarray
        raw_output: np.ndarray

        current_data = self.layers[0].process(current_data, raw=False)
        hidden_output = current_data
        current_data = self.layers[1].process(current_data, raw=True)
        raw_output = current_data
        current_data = sigmoid(current_data)

        if results_container is not None:
            results_container.append((label, np.argmax(current_data)))

        # set up the equations for dJdw for output layer
        # current_data is equivalent to yhat right now
        # combine dJdy and dyds
        dJds: np.ndarray = (y - current_data) * sigmoid_derivative(raw_output)
        dsdw: np.ndarray = hidden_output

        dJdw: np.ndarray = np.outer(dJds, dsdw)

        # change in weights
        self.layers[-1].adjust(self.eta * dJdw, momentum_alpha)
        return

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

def case_one(training_set: list[Image], testing_set: list[Image]):
    # set the input-to-hidden layer weights with the ones from the autoencoder
    # and set the hidden-to-output weights randomly
    c1_layers = [Layer(125, 784 + 1), Layer(10, 125)]
    c1_layers[0].neurons = np.load("autoencoderl1.npy")
    training_fp = open("c1training.txt", 'w')

    c1_network = C1FFNetwork(eta=0.015, layers=c1_layers)
    i = 0

    # calculate error fractions of training and testing sets before training
    training_results = get_error_fraction(c1_network.test(training_set))
    testing_results = get_error_fraction(c1_network.test(testing_set))
    print(f"epoch {i}: {training_results * 100:.3f}%, {testing_results * 100:.3f}%")
    training_fp.write(f"{i}\t{training_results:.5f}\t{testing_results:.5f}\n")

    # calculate error fractions of training and testing every 10 epochs
    for i in range(1, 191):
        training_results = c1_network.train(random.choices(training_set, k=1200))

        if i % 10 == 0:
            training_results = get_error_fraction(training_results)
            testing_results = get_error_fraction(c1_network.test(testing_set))
            print(f"epoch {i}: {training_results * 100:.3f}%, {testing_results * 100:.3f}%")
            training_fp.write(f"{i}\t{training_results:.5f}\t{testing_results:.5f}\n")

    training_fp.close()
    # get outputs of all images in training and test sets
    train_confusion_matrix = np.zeros((10, 10))
    test_confusion_matrix = np.zeros((10, 10))

    results = c1_network.test(training_set)
    for y, yhat in results:
        train_confusion_matrix[y][yhat] += 1

    results = c1_network.test(testing_set)
    for y, yhat in results:
        test_confusion_matrix[y][yhat] += 1
    
    print(train_confusion_matrix)
    print(test_confusion_matrix)

def case_two(training_set: list[Image], testing_set: list[Image]):
    # set the input-to-hidden layer weights with the ones from the autoencoder
    # and set the hidden-to-output weights randomly
    c2_layers = [Layer(125, 784 + 1), Layer(10, 125)]
    c2_layers[0].neurons = np.load("autoencoderl1.npy")
    training_fp = open("c2training.txt", 'w')

    c2_network = C2FFNetwork(eta=0.015, layers=c2_layers)
    i = 0

    # calculate error fractions of training and testing sets before training
    training_results = get_error_fraction(c2_network.test(training_set))
    testing_results = get_error_fraction(c2_network.test(testing_set))
    print(f"epoch {i}: {training_results * 100:.3f}%, {testing_results * 100:.3f}%")
    training_fp.write(f"{i}\t{training_results:.5f}\t{testing_results:.5f}\n")

    # calculate error fractions of training and testing every 10 epochs
    for i in range(1, 191):
        training_results = c2_network.train(random.choices(training_set, k=1200))

        if i % 10 == 0:
            training_results = get_error_fraction(training_results)
            testing_results = get_error_fraction(c2_network.test(testing_set))
            print(f"epoch {i}: {training_results * 100:.3f}%, {testing_results * 100:.3f}%")
            training_fp.write(f"{i}\t{training_results:.5f}\t{testing_results:.5f}\n")

    training_fp.close()
    # get outputs of all images in training and test sets
    train_confusion_matrix = np.zeros((10, 10))
    test_confusion_matrix = np.zeros((10, 10))

    results = c2_network.test(training_set)
    for y, yhat in results:
        train_confusion_matrix[y][yhat] += 1

    results = c2_network.test(testing_set)
    for y, yhat in results:
        test_confusion_matrix[y][yhat] += 1
    
    print(train_confusion_matrix)
    print(test_confusion_matrix)

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

def problem_three(training_set: list[Image], testing_set: list[Image]):
    # set the input-to-hidden layer weights with the ones from the autoencoder
    # and set the hidden-to-output weights randomly
    c1_layers = [SOFM((12, 12), 784, eta=0.075), Layer(10, 144)]
    c1_layers[0].neurons = np.load("sofmdata.npy")
    training_fp = open("p3training.txt", 'w')

    p3_network = P3FFNetwork(eta=0.015, layers=c1_layers)
    i = 0

    # calculate error fractions of training and testing sets before training
    training_results = get_error_fraction(p3_network.test(training_set))
    testing_results = get_error_fraction(p3_network.test(testing_set))
    print(f"epoch {i}: {training_results * 100:.3f}%, {testing_results * 100:.3f}%")
    training_fp.write(f"{i}\t{training_results:.5f}\t{testing_results:.5f}\n")

    # calculate error fractions of training and testing every 10 epochs
    for i in range(1, 191):
        training_results = p3_network.train(random.choices(training_set, k=1400))

        if i % 10 == 0:
            training_results = get_error_fraction(training_results)
            testing_results = get_error_fraction(p3_network.test(testing_set))
            print(f"epoch {i}: {training_results * 100:.3f}%, {testing_results * 100:.3f}%")
            training_fp.write(f"{i}\t{training_results:.5f}\t{testing_results:.5f}\n")

    training_fp.close()
    # get outputs of all images in training and test sets
    train_confusion_matrix = np.zeros((10, 10))
    test_confusion_matrix = np.zeros((10, 10))

    results = p3_network.test(training_set)
    for y, yhat in results:
        train_confusion_matrix[y][yhat] += 1

    results = p3_network.test(testing_set)
    for y, yhat in results:
        test_confusion_matrix[y][yhat] += 1
    
    print(train_confusion_matrix)
    print(test_confusion_matrix)

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

    # case_one(training_set, testing_set)
    # case_two(training_set, testing_set)
    # problem_two(training_set, testing_set)
    problem_three(training_set, testing_set)
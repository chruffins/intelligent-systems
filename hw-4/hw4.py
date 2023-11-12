from typing import Any, Iterable
import numpy as np
import random
import matplotlib.pyplot as plt

PIXELDATA = "data/MNISTnumImages5000_balanced.txt"
LABELDATA = "data/MNISTnumLabels5000_balanced.txt"

class Neuron:
    def __init__(self, eta: float) -> None:
        self.eta = eta

    def train(self):
        raise NotImplementedError

class Image:
    def __init__(self, data, label: int) -> None:
        self.data = data
        self.label = label
        self.y = label_to_arr(label)

class HW4Neuron:
    def __init__(self, initial_weights, activation_func = lambda s: s > 0 and 1 or 0) -> None:
        self.weights = initial_weights
        self.activation_func = activation_func
    
    def process(self, data):
        result = np.dot(data, self.weights)
        return self.activation_func(result)
    
    def raw_process(self, data):
        return np.dot(data, self.weights)

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
        self.neurons = np.random.rand(neurons, inputs) * stdev - stdev # 150 neurons x 725 weights
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

class P1FFNetwork(FeedForwardNetwork):
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

class P2FFNetwork(FeedForwardNetwork):
    def __init__(self, eta: float, layers: list[Layer]) -> None:
        super().__init__(eta, layers)
    
    # for an entire epoch
    def train(self, data: list[Image], momentum_alpha: float = 0.5) -> None:
        for image in data:
            self.backpropagate(image.data, image.data, image.label, momentum_alpha=momentum_alpha)
    
    def test(self, data: list[Image]) -> float:
        loss: float = 0

        for image in data:
            loss += j2(image.data, self.forward_pass(image.data)) # x and yhat
        
        return loss
    
    def digit_test(self, data: list[Image]) -> list[float]:
        losses: np.ndarray = np.zeros((10))

        for image in data:
            losses[image.label] += j2(image.data, self.forward_pass(image.data)) # x and yhat
        
        return list(losses)
            
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
    
def get_test_statistics(results: list[tuple[int, int]]) -> dict:
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for result, truth in results:
        if truth:
            if result:
                tp += 1
            else:
                fn += 1
        elif not truth: # actually negative
            if result:
                fp += 1
            else:
                tn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    error_fraction = (fp + fn) / len(list(results))
    return dict(precision=precision, recall=recall, f1=f1, error_fraction=error_fraction)

def hw4p1(training_set: list[Image], testing_set: list[Image]):
    training_fp = open("p1training.txt", 'w')

    p1_layers = [Layer(125, 784 + 1), Layer(10, 125)]
    p1_network = P1FFNetwork(eta=0.015, layers=p1_layers)

    i = 0

    # calculate error fractions of training and testing sets before training
    training_results = get_error_fraction(p1_network.test(training_set))
    testing_results = get_error_fraction(p1_network.test(testing_set))
    print(f"epoch {i}: {training_results * 100:.3f}%, {testing_results * 100:.3f}%")
    training_fp.write(f"{i}\t{training_results:.5f}\t{testing_results:.5f}\n")

    # calculate error fractions of training and testing every 10 epochs
    for i in range(1, 191):
        training_results = p1_network.train(random.choices(training_set, k=1200))

        if i % 10 == 0:
            training_results = get_error_fraction(training_results)
            testing_results = get_error_fraction(p1_network.test(testing_set))
            print(f"epoch {i}: {training_results * 100:.3f}%, {testing_results * 100:.3f}%")
            training_fp.write(f"{i}\t{training_results:.5f}\t{testing_results:.5f}\n")

    # get outputs of all images in training and test sets
    train_confusion_matrix = np.zeros((10, 10))
    test_confusion_matrix = np.zeros((10, 10))

    results = p1_network.test(training_set)
    for y, yhat in results:
        train_confusion_matrix[y][yhat] += 1

    results = p1_network.test(testing_set)
    for y, yhat in results:
        test_confusion_matrix[y][yhat] += 1
    
    print(train_confusion_matrix)
    print(test_confusion_matrix)

    random.seed(77)
    random_hidden_neurons = random.sample(range(125), 20)

    plt.figure()
    fig, axes = plt.subplots(4, 5, figsize=(10, 13))
    for i in range(4):
        for j in range(5):
            ax = axes[i, j]
            k = (i * 5) + j
            neuron_row = random_hidden_neurons[k]
            neuron_data = np.array(p1_network.layers[0].neurons[neuron_row, 1:])

            ax.set_title(f"Hidden Neuron {neuron_row}")
            ax.imshow(neuron_data.reshape((28,28), order='F'), cmap='gray')
    
    fig.tight_layout()
    plt.show()

def hw4p2(training_set: list[Image], testing_set: list[Image]):
    training_fp = open("p2training.txt", 'w')

    p2_layers = [Layer(125, 784 + 1), Layer(784 + 1, 125)]
    p2_network = P2FFNetwork(eta=0.03, layers=p2_layers)

    i = 0

    # get loss 
    training_results = p2_network.test(training_set) / len(training_set)
    testing_results = p2_network.test(testing_set) / len(testing_set)
    print(f"epoch {i}: {training_results:.2f}, {testing_results:.2f}")
    training_fp.write(f"{i}\t{training_results:.2f}\t{testing_results:.2f}\n")

    # calculate error fractions of training and testing every 10 epochs
    for i in range(1, 191):
        p2_network.train(training_set)
        training_results = p2_network.test(training_set) / len(training_set)

        if i % 10 == 0:
            testing_results = p2_network.test(testing_set) / len(testing_set)
            print(f"epoch {i}: {training_results:.2f}, {testing_results:.2f}")
            #training_fp.write(f"{i}\t{training_results:.2f}\t{testing_results:.2f}\n")
        if training_results < 5.90:
            print(f"epoch {i}: {training_results:.2f}")
        if training_results < 1.01:
            print(f"ending early at epoch {i}")
            break
    
    training_fp.write(f"{training_results * 4000}\t{testing_results * 1000}\n")
    digit_losses = p2_network.digit_test(training_set)
    training_fp.write("\t".join(map(str, digit_losses))+"\n")
    digit_losses = p2_network.digit_test(testing_set)
    training_fp.write("\t".join(map(str, digit_losses))+"\n")

    training_fp.close()

    random.seed(77)
    random_hidden_neurons = random.sample(range(125), 20)

    plt.figure()
    fig, axes = plt.subplots(4, 5, figsize=(10, 13))
    for i in range(4):
        for j in range(5):
            ax = axes[i, j]
            k = (i * 5) + j
            neuron_row = random_hidden_neurons[k]
            neuron_data = np.array(p2_network.layers[0].neurons[neuron_row, 1:])

            ax.set_title(f"Hidden Neuron {neuron_row}")
            ax.imshow(neuron_data.reshape((28,28), order='F'), cmap='gray')
    
    fig.tight_layout()

    plt.figure()
    random_data = random.sample(testing_set, 8)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        ax = axes[0, i]
        ay = axes[1, i]

        dx = random_data[i].data[1:]
        dy = p2_network.forward_pass(random_data[i].data)[1:]

        ax.set_title(f"Actual Image {i}")
        ax.imshow(dx.reshape((28,28), order='F'), cmap='gray')

        ay.set_title(f"Reconstructed Image {i}")
        ay.imshow(dy.reshape((28,28), order='F'), cmap='gray')

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # need to make results "deterministic" lmao
    random.seed(7)
    np.random.seed(7)

    images = read_data()
    training_set: list[Image] = list()
    testing_set: list[Image] = list()
    for i in range(10):
        training_set += images[(i*500)+100:(i*500)+500]
        testing_set += images[(i*500):(i*500)+100]
    random.shuffle(training_set)

    hw4p1(training_set, testing_set)
    # hw4p2(training_set=training_set, testing_set=testing_set)
    """
    binary_train = list()
    binary_test = list()
    for i in range(500):
        a = 1 #random.randint(0, 1)
        b = 1 #random.randint(0, 1)
        r = a & b

        if i < 400:
            binary_train.append((np.array((1, a, b)), r))
        else:
            binary_test.append((np.array((1, a, b)), r))

    binary_layers = [Layer(1, 3)]
    binary_ff = FeedForwardNetwork(0.01, binary_layers)
    for i in range(40):
        for j, v in enumerate(binary_train):
            if j == 0 and i % 5 == 0:
                print(f"epoch {i}: {binary_ff.layers[0].neurons}")
            binary_ff.backpropagate(v[0], v[1])
    print(binary_ff.layers[0].neurons)
    for test in binary_test[:5]:
        print(test, binary_ff.forward_pass(test[0]))
    """
    #for i in range(300):
    #    p1_network.train(random.choices(training_set, k=1000))

    #p1_network.test(testing_set)
    #print(p1_network.forward_pass(images[0].data))
    #print(images[0].label)
    #print(p1_network.backpropagate(images[0].data, images[0].label))
    #print(l.process(np.zeros((784, 1)), raw=True))

    #plt.imshow(np.delete(perceptron.weights, 0).reshape((28,28), order='F'), cmap='rainbow', interpolation='nearest')


    

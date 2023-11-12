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

class HW3Perceptron(Neuron):
    def __init__(self, eta: float, activation_func = lambda s: s > 0 and 1 or 0) -> None:
        super().__init__(eta)
        self.weights = np.random.rand(785) / 2
        self.initial_weights = np.array(self.weights)
        self.is_trained = False
        self.activation_func = activation_func

    def process(self, image: Image) -> int:
        result = np.dot(image.data, self.weights)
        return self.activation_func(result)
    
    def train(self, images: list[Image]):
        results = []

        for image in images:
            yhat = self.activation_func(np.dot(image.data, self.weights))
            y = image.label > 0 and 1 or 0

            # updating the weights
            self.weights += (self.eta * (y - yhat)) * image.data

            # adding results of training
            results.append((yhat, image.label))
        
        return results
    
    def test(self, images: list[Image]):
        results = list(zip(map(self.process, images), [x.label for x in images]))
        return results

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

def create_training_and_test_sets(data: list[Image]) -> tuple[list[Image], list[Image]]:
    zeroes = data[:500]
    random.shuffle(zeroes)
    ones = data[500:1000]
    random.shuffle(ones)

    training_set = zeroes[:400] + ones[:400]

    test_set = zeroes[400:] + ones[400:]

    return training_set, test_set

def create_p1_datasets(data: list[Image]) -> tuple[list[Image], list[Image], list[Image]]:
    zeroes = data[:500]
    random.shuffle(zeroes)
    ones = data[500:1000]
    random.shuffle(ones)

    training_set = zeroes[:400] + ones[:400]

    test_set = zeroes[400:] + ones[400:]
    challenge_set = []
    for i in range(2, 10):
        challenge_set += data[i*500:(i*500)+100]

    return training_set, test_set, challenge_set

def create_p2_datasets(data: list[Image]) -> tuple[list[Image], list[Image], list[Image]]:
    zeroes = data[:500]
    random.shuffle(zeroes)
    nines = data[4500:5000]
    random.shuffle(nines)

    training_set = zeroes[:400] + nines[:400]

    test_set = zeroes[400:] + nines[400:]
    challenge_set = []
    for i in range(1, 9):
        challenge_set += data[i*500:(i*500)+100]

    return training_set, test_set, challenge_set
    
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

def get_tpr_and_fpr(results: list[tuple[int, int]]) -> tuple[float, float]:
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
    
    return tp / (tp + fn), fp / (fp + tn)

def run_perceptron(training: list[Image], test: list[Image], challenge: list[Image], perceptron: HW3Perceptron):
    # 1 and 2 are done in the main function...
    
    # 3. present images in test set to get outputs and then get statistics
    results = get_test_statistics(list(zip(map(perceptron.process, test), [x.label for x in test])))
    n = 0

    print(f"epoch {n}: {results}")
    epoch_statistics = list([results])
    # 4. train the perceptron through multiple passes through the entire training set
    while n < 15:
        n += 1
        raw = perceptron.train(training)
        results = get_test_statistics(raw)

        epoch_statistics.append(results)
        print(f"epoch {n}: {results}")

    # 5. plot the training error fraction versus epoch
    epoch_error_fractions = list(map(lambda x: x['error_fraction'], epoch_statistics))
    # 6. after training, calculate the test statistics on the test set
    trained_results = perceptron.test(test)
    trained_stats = get_test_statistics(trained_results)
    untrained_stats = epoch_statistics[0] # this is the before training stats

    # 7. take trained bias weight and choose a range of 20 values around it
    old_w0 = perceptron.weights[0]
    bias_set = np.linspace(0.8*old_w0, 1.2*old_w0, 20)

    precisions7 = list()
    recalls7 = list()
    f1scores7 = list()
    error_fractions7 = list()

    rocs7 = list()

    labels7 = ["" for _ in range(10)] + ["Trained $w_0$"] + ["" for _ in range(10)]

    for bias in bias_set:
        # get test statistics for each bias.
        perceptron.weights[0] = bias
        bias_test_results = perceptron.test(test)
        
        rocs7.append(get_tpr_and_fpr(bias_test_results))

        test_stats = get_test_statistics(bias_test_results)

        precisions7.append(test_stats['precision'])
        recalls7.append(test_stats['recall'])
        f1scores7.append(test_stats['f1'])
        error_fractions7.append(test_stats['error_fraction'])

    precisions7.insert(10, trained_stats['precision'])
    recalls7.insert(10, trained_stats['recall'])
    f1scores7.insert(10, trained_stats['f1'])
    error_fractions7.insert(10, trained_stats['error_fraction'])
    rocs7.insert(10, get_tpr_and_fpr(trained_results))

    error_fractions7 = list(map(lambda x: 1 - x, error_fractions7))

    # reset the bias weight
    perceptron.weights[0] = old_w0

    # 9. present inputs from the challenge set and see which points are classified as which
    challenge_results_table = np.zeros((2, 8))
    challenge_results = perceptron.test(challenge)
    for yhat, y in challenge_results:
        challenge_results_table[yhat, y - 2] += 1

    # do plotting now
    plot_error_vs_epoch(epoch_error_fractions)
    plot_untrained_vs_trained_stats(untrained_stats, trained_stats)
    plot_bias_change_results(precisions7, recalls7, f1scores7, error_fractions7, labels7)
    plot_roc_points(rocs7, labels7)
    plot_heatmaps(perceptron)

    print(challenge_results_table)

def plot_error_vs_epoch(epoch_error_fractions: list[float]):
    # 5. plots training error fraction
    plt.figure()
    plt.plot(epoch_error_fractions, color='r')
    plt.scatter(range(len(epoch_error_fractions)), epoch_error_fractions, color='r')
    plt.xlabel("Epoch")
    plt.ylabel("Error Fraction")
    plt.title("Error Fraction through Perceptron Training Epochs")

def plot_untrained_vs_trained_stats(untrained_stats: dict, trained_stats: dict):
    # 6. plot error fraction, precision, recall, f1 of untrained vs trained perceptron on test set
    plt.figure()
    plt.bar(range(4), untrained_stats.values(), width=0.35, label="Untrained")
    plt.bar([i + 0.35 for i in range(4)], trained_stats.values(), width=0.35, label="Trained")
    plt.legend()
    plt.xticks([i + 0.35 / 2 for i in range(4)], trained_stats.keys())
    plt.title("Validation Statistics of Untrained vs. Trained Perceptron")

def plot_bias_change_results(precisions7: list[float], recalls7: list[float], f1scores7: list[float], error_fractions7: list[float], labels7: list[str], 
                             ylim: tuple[float, float] = (0.9, 1.05)):
    bw = 0.2
    x = np.arange(21)

    plt.figure(figsize=(12,6))
    plt.bar(x - 1.5 * bw, precisions7, width=bw, label='Precision')
    plt.bar(x - 0.5 * bw, recalls7, width=bw, label='Recall')
    plt.bar(x + 0.5 * bw, f1scores7, width=bw, label='F1')
    plt.bar(x + 1.5 * bw, error_fractions7, width=bw, label='1 - Error Fraction')
    plt.xticks(x, labels7)
    plt.xlabel("Test Statistics")
    plt.ylim(ylim)
    plt.title("Comparison of Different Bias Weights on Perceptron Performance")
    plt.legend(loc='lower right')

def plot_roc_points(rocs7: list[tuple[float, float]], labels7: list[str]):
    # 7. plot the roc curves...
    roc_x, roc_y = zip(*rocs7)

    plt.figure(4)
    plt.scatter(roc_y, roc_x)
    for i, (x, y) in zip(labels7, rocs7):
        plt.text(y, x, i, fontsize=10, ha='left')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.title("ROC Curve")

def plot_heatmaps(perceptron: HW3Perceptron):
    # 8. plot untrained and trained weights side by side as heatmaps
    fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    trained_map = ax1.imshow(np.delete(perceptron.weights, 0).reshape((28,28), order='F'), cmap='rainbow')
    ax1.set_title('Trained Perceptron Weight Heatmap')
    untrained_map = ax2.imshow(np.delete(perceptron.initial_weights, 0).reshape((28,28), order='F'), cmap='rainbow')
    ax2.set_title('Untrained Perceptron Weight Heatmap')

    fig5.tight_layout()

if __name__ == "__main__":
    # need to make results "deterministic" lmao
    random.seed(7)
    np.random.seed(7)

    perceptron1 = HW3Perceptron(0.001)
    perceptron2 = HW3Perceptron(0.001)

    # problem 1
    run_perceptron(*create_p1_datasets(read_data()), perceptron1)
    plt.show()
    # problem 2
    run_perceptron(*create_p2_datasets(read_data()), perceptron2)
    plt.show()

    #plt.imshow(np.delete(perceptron.weights, 0).reshape((28,28), order='F'), cmap='rainbow', interpolation='nearest')


    

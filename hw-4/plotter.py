import matplotlib.pyplot as plt
import numpy as np

DO_P1 = False
DO_P2 = True

if __name__ == "__main__":
    if DO_P1:
        p1data = list(map(str.split, open("p1training.txt").readlines()))
        p1epochs = [int(x[0]) for x in p1data]
        p1trainerr = [float(x[1]) for x in p1data]
        p1testerr = [float(x[2]) for x in p1data]

        plt.plot(p1epochs, p1trainerr, label='Training', color='blue')
        plt.plot(p1epochs, p1testerr, label='Test', color='red')
        plt.xlabel("Epoch")
        plt.xlim(0, 191)
        plt.xticks(np.arange(0, 190, step=20))
        plt.ylabel("Error Fraction")
        plt.title("Error Fraction as a Function of Training Epoch")
        plt.legend()
        plt.show()

    if DO_P2:
        p2data = list(map(str.split, open("p2training.txt").readlines()))
        p2finallosses = [float(x) for x in p2data[1]]
        p2finallosses[0] /= 4000
        p2finallosses[1] /= 1000
        p2traindigitlosses = [float(x) / 400 for x in p2data[2]]
        print(p2data[3])
        p2testdigitlosses = [float(x) / 100 for x in p2data[3]]
        p2categories = ['Training', 'Test']
        p2digits = [str(x) for x in range(10)]

        bar_width = 0.35
        bar_positions1 = np.arange(10)
        bar_positions2 = bar_positions1 + bar_width

        plt.bar(p2categories, p2finallosses, color=['green', 'red'])
        plt.title("Average Loss for Datasets")

        plt.figure()
        plt.bar(bar_positions1, p2traindigitlosses, width=bar_width, label='Training', color='green')
        plt.bar(bar_positions2, p2testdigitlosses, width=bar_width, label='Test', color='red')
        plt.ylabel('Average Loss')
        plt.xticks(bar_positions1 + bar_width / 2, p2digits)
        plt.title('Average Loss for Digits for Datasets')
        plt.legend()

        plt.show()
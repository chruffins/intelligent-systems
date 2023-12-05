import matplotlib.pyplot as plt
import numpy as np

DO_C1 = False
DO_P3 = True

if __name__ == "__main__":
    if DO_C1:
        c1data = list(map(str.split, open("c1training.txt").readlines()))
        c1epochs = [int(x[0]) for x in c1data]
        c1trainerr = [float(x[1]) for x in c1data]
        c1testerr = [float(x[2]) for x in c1data]

        c2data = list(map(str.split, open("c2training.txt").readlines()))
        c2epochs = [int(x[0]) for x in c2data]
        c2trainerr = [float(x[1]) for x in c2data]
        c2testerr = [float(x[2]) for x in c2data]

        plt.plot(c1epochs, c1trainerr, label='Training I', color='green')
        plt.plot(c1epochs, c1testerr, label='Test I', color='yellow')
        plt.plot(c2epochs, c2trainerr, label='Training II', color='blue')
        plt.plot(c2epochs, c2testerr, label='Test II', color='red')
        plt.xlabel("Epoch")
        plt.xlim(0, 191)
        plt.xticks(np.arange(0, 190, step=20))
        plt.yticks(np.linspace(0, 1, 11))
        plt.ylabel("Error Fraction")
        plt.title("Error Fraction as a Function of Training Epoch")
        plt.legend()
        plt.show()

    if DO_P3:
        c1data = list(map(str.split, open("p3training.txt").readlines()))
        c1epochs = [int(x[0]) for x in c1data]
        c1trainerr = [float(x[1]) for x in c1data]
        c1testerr = [float(x[2]) for x in c1data]

        plt.plot(c1epochs, c1trainerr, label='Training I', color='green')
        plt.plot(c1epochs, c1testerr, label='Test I', color='yellow')
        plt.xlabel("Epoch")
        plt.xlim(0, 191)
        plt.xticks(np.arange(0, 190, step=20))
        plt.yticks(np.linspace(0, 1, 11))
        plt.ylabel("Error Fraction")
        plt.title("Error Fraction as a Function of Training Epoch")
        plt.legend()
        plt.show()
import time
from RBF import RBF
import matplotlib.pyplot as plt
import numpy as np
import preProcessing

def main():
    xlabel = np.linspace(0, 1, 1000)
    ylabel = 0.5 + 0.4 * np.sin(2 * np.pi * xlabel)
    learning_rates = [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02]
    num_of_bases = [2, 4, 7, 11, 16,2, 4, 7, 11, 16]
    epochs = 100
    x = preProcessing.data
    y = preProcessing.desired


    for i, (learning_rate, clusters) in enumerate(zip(learning_rates, num_of_bases)):
        start_time = time.time()
        # initialize the network
        rbf = RBF(clusters, epochs, learning_rate, True)
        # training
        rbf.train(x, y)
        # estimation
        h_proximated = rbf.approximation(xlabel)

        end_time = time.time()

        print(f"total time taken with learning rate {learning_rate} and nun of clusters {clusters} with SSE {rbf.get_sse(x, y).item()}: ", end_time - start_time)
        
        # plot results
        plt.figure(figsize=[14, 8])
        plt.plot(xlabel, ylabel, color='black', label='original function')
        plt.plot(xlabel, h_proximated , color='blue', label='estimated h(x)')
        plt.scatter(x, y, color='red', label='h(x)')
        plt.title('RBF Network')
        plt.text(0.9, 0.20, f'#epochs = {epochs}')
        plt.text(0.9, 0.15, 'different variances')
        plt.text(0.9, 0.1, f'k = {clusters}')
        plt.text(0.9, 0.05, f'learning rate = {learning_rate}')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


    for  i, (learning_rate, clusters) in enumerate(zip(learning_rates, num_of_bases)):
        start_time = time.time()
        # initialize the network
        rbf = RBF(clusters, epochs, learning_rate, False)
        # training
        rbf.train(x, y)
        # estimation
        h_proximated = rbf.approximation(xlabel)

        end_time = time.time()

        print(f"total time taken with learning rate {learning_rate} and nun of clusters {clusters} with SSE {rbf.get_sse(x, y).item()}: ", end_time - start_time)
        
        # plot results
        plt.figure(figsize=[14, 8])
        plt.plot(xlabel, ylabel, color='black', label='original function')
        plt.plot(xlabel, h_proximated , color='blue', label='estimated h(x)')
        plt.scatter(x, y, color='red', label='h(x)')
        plt.title('RBF Network')
        plt.text(0.9, 0.20, f'#epochs = {epochs}')
        plt.text(0.9, 0.15, 'same variance')
        plt.text(0.9, 0.1, f'k = {clusters}')
        plt.text(0.9, 0.05, f'learning rate = {learning_rate}')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

if __name__ == '__main__':
    main()
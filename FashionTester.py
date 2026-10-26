from ANN import ANN
from utils import mnist_reader
import numpy as np
import cv2

class FashionTester:

    def __init__(self):
        print ("Loading testing data.")
        self.x_test, self.y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
        print(f"Loaded {self.y_test.shape} examples.")
        print ("Loading ANN from file")
        self.myANN = ANN.fromFile("testANN.dat")
        print(f"Loaded ANN: {self.myANN.layer_sizes=}")




if __name__ == "__main__":
    ft = FashionTester()


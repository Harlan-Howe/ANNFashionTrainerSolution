import math
import random

from ANN import ANN
from ActivationFunctionFile import ActivationType
from utils import mnist_reader
import numpy as np
import cv2

class FashionTester:

    def __init__(self):
        print("Loading training data.")
        self.x_train, self.y_train = mnist_reader.load_mnist('data/fashion', kind='train')
        self.expected_one_hot_train = np.eye(10)[self.y_train]
        print(f"Loaded {self.x_train.shape} and {self.y_train.shape} examples.")
        print ("Loading testing data.")
        self.x_test, self.y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
        print(f"Loaded {self.x_test.shape} and {self.y_test.shape} examples.")
        # print ("Loading ANN from file")
        # self.myANN = ANN.fromFile("fashion 49.dat")
        self.myANN = ANN(layer_sizes=[784, 150, 100, 80, 10], activation_ids=[ActivationType.IDENTITY,
                                                                              ActivationType.SIGMOID,
                                                                     ActivationType.SIGMOID,
                                                                     ActivationType.SIGMOID,
                                                                     ActivationType.SIGMOID])
        print(f"Loaded ANN: {self.myANN.layer_sizes=}")


    def train_one_epoch(self, alpha = 0.0025):
        indices = [j for j in range(len(self.x_train))]
        for i in range(2*len(self.x_train)):
            a = random.randint(0,len(indices)-1)
            b = random.randint(0,len(indices)-1)
            temp = indices[a]
            indices[a] = indices[b]
            indices[b] = temp

        for i in indices:
            ft.myANN.predict(self.x_train[i])
            ft.myANN.backpropagate(ft.expected_one_hot_train[i], alpha=alpha)
            if i>0 and i% 10000 == 0:
                print("*",end="")
        print(f"Trained one epoch at {alpha=}")

if __name__ == "__main__":
    CLASS_NAMES = ("T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot")
    ft = FashionTester()

    NUM_EPOCHS = 200
    scores = [0] * NUM_EPOCHS

    for j in range(NUM_EPOCHS):
        print(f"{j}\t{'-'*40}")
        NUM_TO_CHECK = 1000
        for count in range(NUM_TO_CHECK):
            i = random.randint(0, len(ft.y_test) - 1)
            float_image = ft.x_test[i] / 256.0
            output2 = ft.myANN.predict(float_image)
            # print(output2)
            if np.argmax(output2) == ft.y_test[i]:
                scores[j] += 1
        print(f"{j}\t% correct: {100 * scores[j] / NUM_TO_CHECK:3.2f}")
        ft.train_one_epoch(alpha=0.005*0.99**j)



    score = 0
    print("Pred\tActual")
    NUM_TO_CHECK = 100
    for count in range(NUM_TO_CHECK):
        i = random.randint(0,len(ft.y_test)-1)
        float_image = ft.x_test[i] / 256.0
        output2 = ft.myANN.predict(float_image)
        # print(output2)
        if np.argmax(output2) == ft.y_test[i]:
            x_or_check = "\033[32m√\033[0m"
            score+=1
        else:
            x_or_check = "\033[31mX\033[0m"
        print(f"{np.argmax(output2)}\t{ft.y_test[i]}\t{x_or_check}")
    print(f"% correct: {100 * score / NUM_TO_CHECK:3.2f}")

    screen = np.zeros((800, 1000),dtype=np.uint8)
    for i in range(45):
        j = random.randint(0,len(ft.y_test)-1)
        screen[40*(i%15):40*(i%15)+28, (i//15)*300+2:(i//15)*300+30] = ft.x_test[j].reshape((28,28))
        cv2.putText(screen, text= CLASS_NAMES[np.argmax(ft.myANN.predict(ft.x_test[j]))],fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1.0, org=(50+(i//15)*300, 40*(i%15)+28), color=255, thickness=1)
    cv2.imshow("output", screen)
    cv2.waitKey(1000)
    print(scores)

    response = input("Save file? Y/N")
    if response.lower() == "y":
        ft.myANN.save(f"fashion {score}.dat")

    cv2.destroyAllWindows()







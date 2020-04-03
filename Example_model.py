import numpy as np
import math
from DNN_utils import get_error
from Deep_Neural_Network import DeepNeuralNetwork
from data_parser import*



X, Y  = loadData("inputs.txt","outputs.txt")

# hperparameters
layers_dims = [X.shape[0], 50,40,30, 30 , 3]  # 5-layer model
learning_rate = 0.0075
iterations = 10000


# initializing the model with desired architecture
dnn = DeepNeuralNetwork(layers_dims,"tanh", "linear")

#setting up learning rate decay
dnn.train_with_decay("step_decay",epochs_step= 400, decay_with = 0.0005, till_epoch = 1500) 

# setting up the loss function used for optimization
dnn.set_loss_func("MSE")

# setting up regularizarion
# dnn.train_with_regularization(dropout = False , L2_reg = False, lambd = 10, keep_prop = 0.8)

# training the model
parameters = dnn.train_model(X, Y, iterations, learning_rate, mini_batch_size= 500 , epochs= 3000,
                         params_initialization= "he", optimizer= "adam")


dnn.plotCost()
dnn.export_parameters_to("params.txt")


# making predictions with :
#       Train data:
Y_hat = dnn.predict(X)
print("train mean error: " + str(get_error(Y,Y_hat)))

#       Test data:
X_test, Y_test  = loadData("test_in.txt","test_out.txt")
Y_test_hat = dnn.predict(X_test)
print("test mean error: " + str(get_error(Y_test,Y_test_hat)))


#testing the elements individually from the consol
import random as rnd
while True:
    i = rnd.randint(0,2)
    index = int(input("what element do you want to test (index) ?"))
    print("trained_element: true %f  vs pred %f" %(math.degrees(Y[i][index]),math.degrees(Y_hat[i][index])))
    print("test_element: true %f  vs pred %f" %(math.degrees(Y_test[i][index]),math.degrees(Y_test_hat[i][index])))






# todo:
#  * add batch norm
#  * add dropout norm
#  * add sklearn.preprocessing.MinMaxScaler to scale inputs 






import numpy as np
from DNN_utils import *
import matplotlib.pyplot as plt
import time

class DeepNeuralNetwork:

    layers_dims = []
    hidden_activation  = ""
    output_activation  = ""
    lear_rate = 0
    costs = []
    global params
    loss_func = "MSE"

    # regularization parameters
    dropout = False
    L2_reg = False
    lambd = 0
    keep_prob = 1


    # initalizing learning rate decay parameters:
    epochs_step = 10
    decay_with = 0.00001
    decay_rate = 1
    learning_rate_decay = False
    decayer = ""





    def __init__(self, layers_dims, hidden_activation = "tanh", output_activation = "linear"):
        self.layers_dims = layers_dims
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation


    # learning rate decay functions
    def train_with_decay(self, decayer, epochs_step = None, decay_with = None, till_epoch = None ,  decay_rate = None):
        self.decayer = decayer
        self.learning_rate_decay = True
        self.epochs_step = epochs_step
        self.decay_with = decay_with
        self.decay_rate = decay_rate
        self.till_epoch = till_epoch


    def train_model(self, X, Y, iterations, learning_rate, mini_batch_size, epochs=1000, params_initialization="random",
                    beta=0.9, beta1=0.9, beta2=0.999
                    , optimizer="gd"):

        
        start_time = time.monotonic()
        self.lear_rate = learning_rate
        lr_0 = learning_rate
        global v, s
        self.costs = []


        # initializing network parameters W and b
        parameters = init_parameters_deep(self.layers_dims, params_initialization)

        # initializing optimizers
        if optimizer == "adam":
            v, s = initialize_adam(parameters)
        elif optimizer == "momentum":
            v = initialize_momentum_velocity(parameters)

        # spliting the input dataset into minibatches
        mini_batches = random_mini_batches(X, Y, mini_batch_size)
        num_minibatches = len(mini_batches)
        print("number of mini_batches is: " + str(num_minibatches))



        # training the model
        for i in range(epochs):


            # applying learnig_rate_decay
            if self.learning_rate_decay == True : 
                
                if self.decayer == "step_decay":
                    learning_rate = step_decay(i,learning_rate, self.epochs_step,self.decay_with, till_epoch = self.till_epoch)

                elif self.decayer == "time_based_decay":
                    learning_rate = time_based_decay(i,lr_0, self.decay_rate)
        
                elif self.decayer == "exp_decay":
                    learning_rate = exp_decay(i,lr_0, self.decay_rate)


            self.lear_rate = learning_rate


            # training
            cost_total = 0
            
            for mini_batch in mini_batches:
                (mini_batch_X, mini_batch_Y) = mini_batch

                AL, caches = L_forward_propagation(mini_batch_X, parameters, self.hidden_activation, self.output_activation)
                
                if self.L2_reg == True :
                    cost = compute_cost_with_reg(self.loss_func, mini_batch_Y, AL, parameters , self.lambd )
                
                else:    
                    cost = compute_cost(self.loss_func,mini_batch_Y, AL)


                cost_total += cost

                # checking for L2_regularization request:
                if self.L2_reg == True:
                    grads = L_model_backward(AL, mini_batch_Y, caches, self.loss_func, self.hidden_activation, self.output_activation,
                    L2_reg = self.L2_reg ,parameters = parameters, m = mini_batch_Y.shape[1], lambd = self.lambd)

                else:
                    grads = L_model_backward(AL, mini_batch_Y, caches, self.loss_func, self.hidden_activation, self.output_activation)


                if optimizer == "adam":

                    parameters, v, s = update_params_adam(parameters, grads, v, s, (i + 1), learning_rate, beta1, beta2 )

                elif optimizer == "momentum":

                    parameters, v = update_params_momentum(parameters, grads, v, beta, learning_rate)
                else:
                    parameters = update_params_gd(parameters, grads, learning_rate)

            cost_avg = cost_total / num_minibatches
            




            # Print the cost every 100 epoch
            if i % 100 == 0:
                print("Cost after epoch %i: %f  " % (i, cost_avg))
            if i % 100 == 0:
                self.costs.append(cost_avg)




        # getting the total time taken from training the model
        end_time = time.monotonic()
        delta = (end_time - start_time) / 60.
        seconds = (delta - int(delta)) * 60.
        print("The training took: " + str(int(delta)) + " Minutes and " + str(int(seconds)) + " seconds.")

        self.params = parameters

        return parameters



    # function for testing the model on the current network
    def predict(self,X):
        Y_hat, _ = L_forward_propagation(X, self.params, self.hidden_activation, self.output_activation) 
        return Y_hat



    # function for plotting the cost graph
    def plotCost(self):
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(self.lear_rate))
        plt.show()


    # function for exporting the model parameters to a textfile
    def export_parameters_to(self, filename):
        print("Saving parameters......")

        with open((filename), 'w') as outputStream:
            outputStream.write(self.params)
        
        print("Done Saving parameters !")

    #setter for setting the cost/loss
    def set_loss_func(self, loss_func):
        self.loss_func = loss_func


    def train_with_regularization(self,dropout = False , L2_reg = False, lambd = 0, keep_prop = 1):
        self.L2_reg = L2_reg
        self.dropout = dropout
        self.lambd = lambd
        self.keep_prop = keep_prop



    
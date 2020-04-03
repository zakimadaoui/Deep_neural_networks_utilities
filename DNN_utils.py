import numpy as np
import math
import matplotlib.pyplot as plt


# ========================================== Popular activation functions ==========================================
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    cache = z
    return s,cache

def tanh(z):
    t = np.tanh(z)
    cache = z
    return t, cache

def relu(z):
    r = np.maximum(0, z)
    assert (r.shape == z.shape)
    return r, z


# ========================================== derivatives of activation functions========================================

def sigmoid_prime(z):
    s = 1/(1+np.exp(-z))
    sp = s*(1-s)
    return sp

def tanh_prime(z):
    tp = 1 - np.tanh(z)**2
    return tp

def relu_prime(z):
    rp = np.greater(z, 0).astype(float)  # returns 1 if greater than 0 OR 0 if less or equal to 0
    rp = np.reshape(rp, z.shape)

    assert (z.shape == rp.shape)
    return rp

#================================================= Popular loss functions ==============================================
"""to compute the cost just call compute_cost(your_loss_function())"""

def L1loss_MAE(Y,Y_hat):
    return np.sum(np.abs(Y - Y_hat))


def L2loss_MSE(Y,Y_hat):
    return ((Y - Y_hat)**2)

def binary_cross_entropy(Y,Y_hat):

    if np.min(Y) <= 0  or np.min(Y_hat) <= 0 :
        raise ValueError("Negative element in log function , Solution: try to map your outputs to values that are grater than zero!")
    else:
        return np.sum(-(Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat)))
    
        


def cross_entropy(Y,Y_hat):
    
    if np.min(Y) <= 0  or np.min(Y_hat) <= 0 :
        raise ValueError("Negative element or zero in log function , Solution: try to map your outputs to values that are grater than zero!")
    else:
        return np.sum(-(Y*np.log(Y_hat)))
        



def hurber_loss(Y,Y_hat,delta):
    loss = np.where(np.abs(Y - Y_hat) < delta, 0.5 * ((Y - Y_hat) ** 2),
                    delta * np.abs(Y - Y_hat) - 0.5 * (delta ** 2))
    return np.sum(loss)


def log_Cosh_loss(Y,Y_hat):
    loss = np.log(np.cosh(Y - Y_hat))
    return np.sum(loss)


def getLossDerivative(Y,AL,cost_func_name):
    global dAl

    if cost_func_name == "binary_cross_entropy":
        dAl = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # derivative of cost with respect to AL

    elif cost_func_name == "cross_entropy":
        dAl = - np.divide(Y, AL)

    elif cost_func_name == "MSE":
        dAl = 2*(AL - Y)

    return dAl


def compute_cost(loss_func, Y_true, Y_pred):
    
    if loss_func == "cross_entropy":
        loss = cross_entropy(Y_true, Y_pred)

    elif loss_func == "binary_cross_entropy":
        loss = binary_cross_entropy(Y_true, Y_pred)

    elif loss_func == "MSE":
        loss = L2loss_MSE(Y_true, Y_pred)

    # elif loss_func == "MAE":   #backprop not emplimented yet
    #     loss = L1loss_MAE(Y_true, Y_hat)
    
    return np.array(loss).mean()


def compute_cost_with_reg(loss_func, Y_true, Y_pred, parameters = None, lambd = None ):
    
    if loss_func == "cross_entropy":
        loss = cross_entropy(Y_true, Y_pred)

    elif loss_func == "binary_cross_entropy":
        loss = binary_cross_entropy(Y_true, Y_pred)

    elif loss_func == "MSE":
        loss = L2loss_MSE(Y_true, Y_pred)

    # elif loss_func == "MAE":   #backprop not emplimented yet
    #     loss = L1loss_MAE(Y_true, Y_hat)


    squared_weights= 0
    m = Y_true.shape[1]
    L = len(parameters) //2
    for l in range(1,L):
    
        squared_weights+= np.sum(np.square(parameters["W"+ str(l)]))

    
    return np.array(loss).mean() +  (lambd * squared_weights / m)

#==================================================== Learning rate decay ===============================================

def step_decay(epoch, current_learning_rate, epochs_step, decay_with, till_epoch):

    lr = current_learning_rate

    if till_epoch != None and till_epoch == epoch :
        lr = current_learning_rate
    else:
        if (epoch % epochs_step) == 0:
            lr -= decay_with
        #avoiding the case where learning_rate is less or eqal to 0
        if lr < 0. or lr == 0. : 
            lr = 0.00000001

    return lr




def time_based_decay(epoch, init_learning_rate, decay_rate):
    lr = init_learning_rate / (1 + decay_rate* epoch)
    return lr


def exp_decay(epoch, init_learning_rate, decay_rate = 0.1):
    lr = init_learning_rate * math.exp(-decay_rate*epoch)
    return lr



#================================================= Parameters initialization ===========================================
def init_parameters_deep(layers_dims, type = "random"):

    L = len(layers_dims)
    parameters = {}
    optimization = []

    if type == "random":
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])*0.01
            parameters["b" + str(l)] = np.zeros(shape = (layers_dims[l], 1))
            assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
            assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

            
    elif type == "xavier":
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])*np.sqrt(1/layers_dims[l-1])
            parameters["b" + str(l)] = np.zeros(shape = (layers_dims[l], 1))
            assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
            assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))
            
            
    elif type == "he":
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])*np.sqrt(2/layers_dims[l-1])
            parameters["b" + str(l)] = np.zeros(shape = (layers_dims[l], 1))
            assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
            assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))


    return  parameters



def initialize_momentum_velocity(parameters):

    L = len(parameters) // 2
    v = {}

    # Initialize velocity
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(
            shape=(parameters['W' + str(l + 1)].shape[0], parameters['W' + str(l + 1)].shape[1]))
        v["db" + str(l + 1)] = np.zeros(
            shape=(parameters['b' + str(l + 1)].shape[0], parameters['b' + str(l + 1)].shape[1]))

    return v


def initialize_adam(parameters):

    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):

        v["dW" + str(l + 1)] = np.zeros(
            shape=(parameters['W' + str(l + 1)].shape[0], parameters['W' + str(l + 1)].shape[1]))
        v["db" + str(l + 1)] = np.zeros(
            shape=(parameters['b' + str(l + 1)].shape[0], parameters['b' + str(l + 1)].shape[1]))
        s["dW" + str(l + 1)] = np.zeros(
            shape=(parameters['b' + str(l + 1)].shape[0], parameters['b' + str(l + 1)].shape[1]))
        s["db" + str(l + 1)] = np.zeros(
            shape=(parameters['b' + str(l + 1)].shape[0], parameters['b' + str(l + 1)].shape[1]))


    return v, s

#======================================= One layer Forward propagation ===================================================
# helper function for linear_activation_forward()
def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    assert (Z.shape == (W.shape[0], A_prev.shape[1]))
    cache = (A_prev, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    * Here we are doing forward propagation on one layer on (layer 'l') then we cache A, Z , A_prev (or the layers inputs),w and b.
    * in the model we are going to repeat this for all layers and keep the cache in a dictionary so that we use it for backprop.
    """

    global A, activation_cache


    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "tanh":
        A, activation_cache = tanh(Z)
    elif activation == "linear":
        A, activation_cache = Z,Z

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    # return the cache as a tuple
    cache = (linear_cache, activation_cache)

    return A, cache  # 'A' here will become A_prev (input) for the next layer when we feed the network

# forward prop thought all layers and return y_hat with caches
def L_forward_propagation(X, parameters ,hidden_activation,output_activation):
    caches = []
    A_prev = X
    L = len(parameters) // 2

    for l in range(1, L):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A_prev, cache = linear_activation_forward(A_prev, W, b, hidden_activation)
        caches.append(cache)

    # finding y hat or aka AL


    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL, cache = linear_activation_forward(A_prev, W, b, output_activation)
    caches.append(cache)

    assert (AL.shape[1] ==  X.shape[1])

    return AL, caches
#======================================== One layer Backward propagation ===============================================
# helper function for linear_activation_backward()
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (dZ @ A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = W.T @ dZ

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, caches, activation):
    """
    *Here we take the cache saved from forward prop and use it to calculate gradients for the current layer
    """

    linear_cache, activation_cache = caches
    z = activation_cache

    if activation == "sigmoid":
        dZ = dA * sigmoid_prime(z)

    if activation == "relu":
        dZ = dA * relu_prime(z)

    if activation == "tanh":
        dZ = dA * tanh_prime(z)

    if activation == "linear":
        dZ = dA

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db



#backprop through all layers
def L_model_backward(AL, Y, caches,cost_func_name, hidden_activation, output_activation, L2_reg = False, parameters = None, m = 1 , lambd = 0 ):

    grads = {}
    L = len(caches)  # nbr of layers
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    dAL = getLossDerivative(Y,AL,cost_func_name)

    # checking for L2_regularization request for layer L
    if L2_reg == True : 
        L2_L = ((lambd * parameters["W" + str(L)]) / m )
    else:
        L2_L = 0

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]  # get the last  element
    dA_prev, dW, db = linear_activation_backward(dAL, current_cache, output_activation)
    grads["dA" + str(L - 1)] = dA_prev
    grads["dW" + str(L)] = dW + L2_L 
    grads["db" + str(L)] = db

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):

        # checking for L2_regularization request for layer L
        if L2_reg == True : 
            L2_l = ((lambd * parameters["W" + str(l+1)]) / m )
        else:
            L2_l = 0

        # continueing backprop
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    activation = hidden_activation)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp + L2_l
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_params_gd(parameters,grads,learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)] 
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def update_params_momentum(parameters, grads, v, beta, learning_rate):

    L = len(parameters) // 2

    # Momentum update for each parameter
    for l in range(L):
        # compute velocities
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
        # update parameters
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]


    return parameters, v


def update_params_adam(parameters, grads, v, s, t, learning_rate,
                                beta1=0.9, beta2=0.999, epsilon=1e-8  ):

    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    # Perform Adam update on all parameters
    for l in range(L):

        # Moving average of the gradients
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

        # Compute bias-corrected first moment estimate
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta1 ** t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta1 ** t)

        # Moving average of the squared gradients
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * grads["dW" + str(l + 1)] ** 2
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * grads["db" + str(l + 1)] ** 2


        # Compute bias-corrected second raw moment estimate
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - beta2 ** t)
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - beta2 ** t)


        # Update the parameters combining momentum and RMS_prop

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (
                    v_corrected["dW" + str(l + 1)] / (np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon)) 
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (
                    v_corrected["db" + str(l + 1)] / (np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon))


    return parameters, v, s

# ========================================== mini-batches generator ====================================================

def random_mini_batches(X, Y, mini_batch_size = 64):

    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    assert(Y.shape[1] == X.shape[1])
    assert(shuffled_Y.shape[1] == shuffled_X.shape[1])

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitioning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, (num_complete_minibatches) * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, (num_complete_minibatches) * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# ==================================================================================================================

def get_error(Y, Y_hat):
    error = 100* (np.abs(Y - Y_hat))/Y
    return np.mean(error)

def get_accuracy(Y, Y_hat):
    error = get_error(Y , Y_hat)
    return 100 - error






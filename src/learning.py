"""
This script contains the machine learning algorithms (with the exception of the dimensionless SINDy algorithm)
"""

import numpy as np
import numpy.random as rng
from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
import pdb

## TODO:
# - add plotting class in vizualization.py

#############################################################
#############################################################

class KRidgeReg:
    """
    Kernel Ridge Regression fits dimensionless inputs and outputs using the Buckingham Pi constraint
    """
    def __init__(self, inputs, outputs, dim_matrix, 
                    test_size= 0.15, 
                    num_nondim=1,
                    use_test_set=True,
                    normalize=False,
                    l1_reg=1e-3, 
                    alpha=1e-4, 
                    kernel='rbf', 
                    gamma=10):

        ## !! All inputs are logged - scaling doesn't work otherwise
        self.inputs = np.log(inputs) 
        # if len(np.where(self.inputs==float('inf'))) > 0:
        #     raise Exception('data has infinities')
        self.outputs = outputs
        self.krr = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)

        ## To normalize need to define log(inputs) and work with them
        self.inputs_train, self.inputs_test, self.outputs_train, self.outputs_test = \
                train_test_split(inputs, outputs, test_size=test_size, shuffle=True)

        self.dim_matrix = dim_matrix
        self.l1_reg = l1_reg
        self.num_nondim = num_nondim
        self.use_test_set = use_test_set
        self.scaler = StandardScaler()

    def single_run(self):
        res = self.optimize(True)
        return self.reshape_x(res.x)

    def multi_run(self, ntrials=10):
        min_loss = 1e8
        x = None
        for i in range(ntrials):
            try:
                res = self.optimize(False)
                if self.use_test_set:
                    final_loss = 1 - self.krr.score(np.exp(self.inputs_test @ self.reshape_x(res.x)), self.outputs_test)
                else:
                    final_loss = res.fun

                if final_loss < min_loss:
                    x = self.reshape_x(res.x)
                    min_loss = final_loss 
                    print(min_loss)

            # If optimization fails, try again
            except ValueError:
                pass
        if x is None:
            raise Exception('did not find optimal solution')

        # might want to return loss too
        return x

    def optimize(self, display=True):
        x0 = rng.randn(self.dim_matrix.shape[1], self.num_nondim)
        res = minimize(lambda x: self.loss(x, l1_reg=self.l1_reg),
                    x0, constraints=[{'type':'eq', 'fun': self.constr}], options={'disp': display})
        return res

    def loss(self, x, l1_reg=0):
        ## L2 might help (?)
        # Consider other metrics - score vs. rmse 
        theta_train = np.exp(self.inputs_train @ self.reshape_x(x))
        self.krr.fit(theta_train, self.outputs_train)
        if self.use_test_set:
            theta_test = np.exp(self.inputs_test @ self.reshape_x(x))
            return 1 - self.krr.score(theta_test, self.outputs_test) + l1_reg*np.linalg.norm(x, ord=1)
        else:
            return 1 - self.krr.score(theta_train, self.outputs_train) + l1_reg*np.linalg.norm(x, ord=1)

    def constr(self, x):
        return (self.dim_matrix @ self.reshape_x(x)).flatten()

    def reshape_x(self, x):
        # There might be a neater way of doing this
        if len(self.inputs.shape)>1:
            return np.reshape(x, [self.inputs.shape[1], self.num_nondim])
        else:
            return x

    def normalize(self):
        self.inputs_train = self.scaler.fit_transform(self.inputs_train)
        self.inputs_test = self.scaler.transform(self.inputs_test)
        
        

#####################################################################################
#####################################################################################
        
class SVMClassifier:
    def __init__(self, inputs, outputs, dim_matrix, 
                    test_size= 0.15, 
                    num_nondim=1,
                    use_test_set=True,
                    normalize=False,
                    l1_reg=1e-3, 
                    alpha=1e-4, 
                    kernel='rbf', 
                    gamma=10):

        ## !! All inputs are logged - scaling doesn't work otherwise
        self.inputs = np.log(inputs) 
        # if len(np.where(self.inputs==float('inf'))) > 0:
        #     raise Exception('data has infinities')
        self.outputs = outputs
        self.classifier = SVC(C=alpha, kernel=kernel, gamma=gamma)

        ## To normalize need to define log(inputs) and work with them
        self.inputs_train, self.inputs_test, self.outputs_train, self.outputs_test = \
                train_test_split(inputs, outputs, test_size=test_size, shuffle=True)

        self.l1_reg = l1_reg
        self.num_nondim = num_nondim
        self.use_test_set = use_test_set

    def loss(self, x, l1_reg=0):
        ## L2 might help (?)
        # Consider other metrics - score vs. rmse 
        theta_train = np.exp(self.inputs_train @ self.reshape_x(x))
        self.classifier.fit(theta_train, self.outputs_train)
        if self.use_test_set:
            theta_test = np.exp(self.inputs_test @ self.reshape_x(x))
            return 1 - self.classifier.score(theta_test, self.outputs_test) + l1_reg*np.linalg.norm(x, ord=1)
        else:
            return 1 - self.classifier.score(theta_train, self.outputs_train) + l1_reg*np.linalg.norm(x, ord=1)

    def reshape_x(self, x):
        # There might be a neater way of doing this
        if len(self.inputs.shape)>1:
            return np.reshape(x, [self.inputs.shape[1], self.num_nondim])
        else:
            return x

#####################################################################################
#####################################################################################

class BuckyNet:
    """
    BuckiNet is a deep learning architecture that fits inputs and outputs using the Buckingham Pi constraint
    """
    def __init__(self, inputs, outputs, dim_matrix, 
                num_nondim= 1,  
                num_layers = 2,
                num_neurons = 40, # Can pass a list of neurons instead
                activation = 'elu',
                initializer = 'he_normal',
                nepoch = 500,
                patience = 20,
                test_size = 0.15,
                nullspace_loss = None,
                l1_reg= 0.0001,
                l2_reg = 0.000001,
                adamlr = 0.001,
                verbose = 0,
                normalize = False):

        self.inputs_train, self.inputs_test, self.outputs_train, self.outputs_test = \
            train_test_split(np.log(inputs), outputs, test_size=test_size, shuffle=False)
        self.inputs_train, self.inputs_dev, self.outputs_train, self.outputs_dev = \
                train_test_split(self.inputs_train, self.outputs_train, test_size=test_size, shuffle=False) 
        self.dim_matrix = dim_matrix
        self.num_nondim = num_nondim 
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.activation = activation
        self.initializer = initializer
        self.nepoch = nepoch
        self.patience = patience
        self.nullspace_loss = nullspace_loss
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.adamlr = adamlr
        self.normalize = normalize
        self.verbose = verbose
        self.model = self.build_model()
        self.history = None

    def build_model(self):
        input_layer = keras.layers.Input(shape=[self.inputs_train.shape[1]], name='inputs')
        pi_layer = keras.layers.Dense(self.num_nondim, activation='exponential', kernel_initializer=self.initializer, 
                kernel_regularizer=keras.regularizers.l1_l2(self.l1_reg, self.l2_reg), use_bias=False, name='pi_numbers')(input_layer)
        layer = pi_layer 
        for i in range(self.num_layers):
            layer = keras.layers.Dense(self.num_neurons, activation=self.activation, kernel_initializer=self.initializer, 
                    kernel_regularizer=keras.regularizers.l2(self.l2_reg), 
                    bias_regularizer=keras.regularizers.l2(self.l2_reg), name='layer_'+str(i+1))(layer)
        
        if len(self.outputs_train.shape) > 1:
            outlayers = self.outputs_train.shape[1]
        else:
            outlayers = 1
        output_layer = keras.layers.Dense(outlayers, activation='linear', name='outputs', 
                            kernel_initializer=self.initializer,
                            kernel_regularizer=keras.regularizers.l2(self.l2_reg), 
                            bias_regularizer=keras.regularizers.l2(self.l2_reg))(layer)
        

        model = keras.Model(inputs=input_layer, outputs=[output_layer])
        optimizer = keras.optimizers.Adam(lr=self.adamlr)
        model.compile(loss='mse', optimizer=optimizer, metrics=[[keras.metrics.RootMeanSquaredError()]])

        if self.nullspace_loss is not None:
            P = tf.constant(self.dim_matrix, dtype=tf.float32)
            model.add_loss(lambda: self.nullspace_loss * tf.reduce_mean(tf.square( tf.matmul(P, model.layers[1].kernel))) )

        return model

    def single_run(self):
        eval_loss = self.loss()
        weights = self.model.layers[1].get_weights()[0]
        return weights

    def loss(self): 
        self.history = self.model.fit(self.inputs_train, self.outputs_train, 
                                validation_data=(self.inputs_dev, self.outputs_dev), 
                                epochs=self.nepoch, verbose=self.verbose, callbacks=[keras.callbacks.EarlyStopping(patience=self.patience)])
        lossval = self.model.evaluate(self.inputs_test, self.outputs_test)
        return lossval


#####################################################################################
#####################################################################################



class NeuralNet:
    """
    This class simply fits inputs to outputs with a neural network, using the .loss method
    It is used in the brute force method that pre-generates all dimensionless numbers
    """
    def __init__(self, inputs, outputs, dim_matrix, 
                num_nondim= 1,  
                num_layers = 2,
                num_neurons = 40, # Can pass a list of neurons instead
                activation = 'elu',
                initializer = 'he_normal',
                nepoch = 500,
                patience = 20,
                test_size = 0.15,
                normalize = False,
                l1_reg= 1e-3,
                l2_reg = 0.0001,
                adamlr = 0.001):

        self.inputs_train, self.inputs_test, self.outputs_train, self.outputs_test = \
            train_test_split(inputs, outputs, test_size=test_size, shuffle=True)
        self.inputs_train, self.inputs_dev, self.outputs_train, self.outputs_dev = \
                train_test_split(self.inputs_train, self.outputs_train, test_size=test_size, shuffle=True) 

        self.dim_matrix = dim_matrix
        self.num_nondim = num_nondim 
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.activation = activation
        self.initializer = initializer
        self.nepoch = nepoch
        self.patience = patience
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.adamlr = adamlr
        self.model = self.build_model()

    def build_model(self):
        input_layer = keras.layers.Input(shape=[self.num_nondim], name='inputs')
        layer = input_layer
        for i in range(self.num_layers):
            layer = keras.layers.Dense(self.num_neurons, activation=self.activation, kernel_initializer=self.initializer, 
                    kernel_regularizer=keras.regularizers.l2(self.l2_reg), 
                    bias_regularizer=keras.regularizers.l2(self.l2_reg), name='layer_'+str(i+1))(layer)

        output_layer = keras.layers.Dense(self.outputs.shape[1], activation='linear', name='outputs', 
                            kernel_initializer=self.initializer,
                            kernel_regularizer=keras.regularizers.l2(self.l2_reg), 
                            bias_regularizer=keras.regularizers.l2(self.l2_reg))(layer)

        model = keras.Model(inputs=input_layer, outputs=[output_layer])
        optimizer = keras.optimizers.Adam(lr=self.adamlr)
        model.compile(loss='mse', optimizer=optimizer, metrics=[[keras.metrics.RootMeanSquaredError()]])
        return model

    def loss(self, x):
        # Fix input shape x
        history = self.model.fit(np.exp(np.log(self.inputs_train) @ x), self.outputs_train, 
                                validation_data=(np.exp(np.log(self.inputs_dev) @ x), self.outputs_dev), 
                epochs=self.nepoch, verbose=0, callbacks=[keras.callbacks.EarlyStopping(patience=self.patience)])
        lossval = self.model.evaluate(np.exp(np.log(self.inputs_test) @ x), self.outputs_test)
        return lossval


#############################################################
#############################################################

class KRidgeReg_struct:
# Same as KRidgeReg but uses DataStruct - BREAKS
## TODO:
# Re-check code below - getting worse results when using DataStruct in KRdigeReg
# Originally replaces all codes above using the DataStruct class (below), but it's giving strange results so I switched back to old version

    def __init__(self, inputs, outputs, dim_matrix, 
                    test_size= 0.15, 
                    num_nondim=1,
                    use_test_set=True,
                    normalize=False,
                    l1_reg=1e-3, 
                    alpha=1e-4, 
                    kernel='rbf', 
                    gamma=10):

        self.krr = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)
        # It might be better to pass in the data object as input to the class.
        self.data = DataStruct(inputs, outputs, log_inputs=True, normalize=False, test_size=test_size, dev_size=None, shuffle=True)
        self.dim_matrix = dim_matrix
        self.l1_reg = l1_reg
        self.num_nondim = num_nondim
        self.use_test_set = use_test_set

    def single_run(self):
        res = self.optimize(True)
        return self.reshape_x(res.x)

    def multi_run(self, ntrials=10):
        min_loss = 1e8
        x = None
        for i in range(ntrials):
            try:
                res = self.optimize(False)
                if self.use_test_set:
                    final_loss = 1 - self.krr.score(np.exp(self.data.inputs_test @ self.reshape_x(res.x)), self.data.outputs_test)
                else:
                    final_loss = res.fun

                if final_loss < min_loss:
                    x = self.reshape_x(res.x)
                    min_loss = final_loss 
                    print(min_loss)

            # If optimization fails, try again
            except ValueError:
                pass
        if x is None:
            raise Exception('did not find optimal solution')

        # might want to return loss too
        return x

    def optimize(self, display=True):
        x0 = rng.randn(self.dim_matrix.shape[1], self.num_nondim)
        res = minimize(lambda x: self.loss(x, l1_reg=self.l1_reg),
                    x0, constraints=[{'type':'eq', 'fun': self.constr}], options={'disp': display})
        return res

    def loss(self, x, l1_reg=0):
        ## L2 might help (?)
        # Consider other metrics - score vs. rmse 
        theta_train = np.exp(self.data.inputs_train @ self.reshape_x(x))
        self.krr.fit(theta_train, self.data.outputs_train)
        return 1 - self.krr.score(theta_train, self.data.outputs_train) + l1_reg*np.linalg.norm(x, ord=1)

    def constr(self, x):
        return (self.dim_matrix @ self.reshape_x(x)).flatten()

    def reshape_x(self, x):
        # There might be a neater way of doing this
        if len(self.data.inputs.shape)>1:
            return np.reshape(x, [self.data.inputs.shape[1], self.num_nondim])
        else:
            return x


###############################################3

class DataStruct:
# This class puts data in object where it can be split, scaled, log()ed and transformed etc. 
# TODO: including it in code causes some problems:
    def __init__(self, inputs, outputs, log_inputs=True, normalize=False, test_size=0.15, dev_size=0.15, shuffle=True):
        self.inputs = inputs
        if log_inputs:
            self.inputs = np.log(inputs)
            # Check if array has infinities or nans
        self.outputs = outputs
        self.scaler = StandardScaler()
        if normalize:
            self.scaler.fit_transform(self.inputs)
            # To transform back use self.scaler.inverse_transform(x)
            # See https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

        self.inputs_test = None
        self.outputs_test = None
        self.inputs_dev = None
        self.outputs_dev = None
        self.inputs_train = None
        self.outputs_train = None

        if test_size is not None:
            self.inputs_train, self.inputs_test, self.outputs_train, self.outputs_test = \
                train_test_split(inputs, outputs, test_size=test_size, shuffle=shuffle)
            if dev_size is not None:
            # Makes sure test_size is not none 
                self.inputs_train, self.inputs_dev, self.outputs_train, self.outputs_dev = \
                        train_test_split(self.inputs_train, self.outputs_train, test_size=dev_size, shuffle=shuffle) 


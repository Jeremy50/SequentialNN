#How to implement a Sequential NN of Dense layers with Adam opt and MSE loss in python using numpy
#WARNING: DO NOT ASSIGN SAME OPTIMIZER TO MULTIPLE WEIGHTS INCLUDING BIAS AND WEIGHTS IN SAME LAYER

import numpy as np

batch_size = 32
input_size = 3
output_size = 2

class Adam:
    
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, e=1e-8):
        
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.m = 0
        self.v = 0
    
    def updated_weights(self, weights, grads):
        
        self.m = self.b1 * self.m + (1 - self.b1) * grads
        self.v = self.b2 * self.v + (1 - self.b2) * grads ** 2
        m = self.m / (1 - self.b1)
        v = self.v / (1 - self.b2)
        return weights - self.lr * m / np.sqrt(v + self.e)

class Dense:
    
    def __init__(self, input_size, output_size, weight_opt=None, bias_opt=None):
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)
        
        self.dy_dw = None
        self.dy_db = 1
        self.x = None
        
        self.w_opt = weight_opt if weight_opt != None else Adam(0.001)
        self.b_opt = bias_opt if bias_opt != None else Adam(0.005)
    
    def __call__(self, x):
        
        self.x = x
        self.dy_dw = np.expand_dims(np.sum(x, 0) / x.shape[0], 1)
        return x @ self.w + self.b

    def update_weights(self, partial):
        
        self.w = self.w_opt.updated_weights(self.w, np.ones_like(self.w) * partial * self.dy_dw)
        self.b = self.b_opt.updated_weights(self.b, np.ones_like(self.b) * partial * self.dy_db)
        return self.get_partial(partial)
    
    def get_partial(self, partial):
        
        return np.sum(partial * self.w, 1) / self.w.shape[1]
    
class MSE:
    
    def __init__(self):
        
        self.y = None
        self.dl_dy = None
    
    def __call__(self, y, y_true):
        
        self.y = y
        self.dl_dy = 2 * np.sum(y - y_true, 0) / y.shape[0]
        return (y_true - y) ** 2

    def get_partial(self):
        
        return self.dl_dy

dense_true = Dense(input_size, output_size)

dense1 = Dense(input_size, 16)
dense2 = Dense(16, 8)
dense3 = Dense(8, output_size)

mse = MSE()

for i in range(10000):
    
    x = np.random.randn(batch_size, input_size)
    y_true = dense_true(x)
    y = dense3(dense2(dense1(x)))
    
    l = mse(y, y_true)
    if i % 100 == 0:
        print("Average Loss:", np.sum(l) / np.size(l))
    
    dense1.update_weights(dense2.update_weights(dense3.update_weights(mse.get_partial())))
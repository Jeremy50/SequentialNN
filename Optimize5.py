#How to implement a Sequential NN of Dense layers with Adam opt and MSE loss in python using numpy
#WARNING: DO NOT ASSIGN SAME OPTIMIZER TO MULTIPLE WEIGHTS INCLUDING BIAS AND WEIGHTS IN SAME LAYER

import numpy as np

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

class Layer:
    
    prev_layer = None
    next_layer = None
    layers = 0
    name = ""
    
    def __init__(self, name=None):
    
        Layer.layers += 1
        self.name = name if name is not None else f"Layer{Layer.layers}"
    
    def add_to(self, prev_layer):
        
        self.prev_layer = prev_layer
        prev_layer.next_layer = self
        return self

    def first(self):
        
        if self.prev_layer is None: return self
        return self.prev_layer.first()

    def print_summary(self): self.first()._print_summary()
    
    def _print_summary(self):
        
        print(self)
        if self.next_layer is not None:
            self.next_layer._print_summary()

    def __str__(self): return self.name

class Dense(Layer):
    
    def __init__(self, input_size, output_size, weight_opt=None, bias_opt=None, name=None):
        
        super().__init__(name)
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)
        
        self.dy_dw = None
        self.dy_db = 1
        self.x = None
        
        self.w_opt = weight_opt if weight_opt != None else Adam(0.001)
        self.b_opt = bias_opt if bias_opt != None else Adam(0.005)

    def update_weights(self, partial):
        
        self.w = self.w_opt.updated_weights(self.w, np.ones_like(self.w) * partial * self.dy_dw)
        self.b = self.b_opt.updated_weights(self.b, np.ones_like(self.b) * partial * self.dy_db)
        if self.prev_layer is not None: self.prev_layer.update_weights(np.sum(partial * self.w, 1) / self.w.shape[1])
    
    def __call__(self, x): return self.first()._call(x)
    def __str__(self): return f"{self.name}[Dense]: {self.input_size} -> {self.output_size}"
        
    def _call(self, x):
        
        self.x = x
        self.dy_dw = np.expand_dims(np.sum(x, 0) / x.shape[0], 1)
        out = x @ self.w + self.b
        if self.next_layer is None or isinstance(self.next_layer, MSE): return out
        return self.next_layer._call(out)
    
class MSE(Layer):
    
    def __init__(self, name=None):
        
        super().__init__(name)
        
        self.y = None
        self.dl_dy = None
    
    def __call__(self, y, y_true):
        
        self.y = y
        self.dl_dy = 2 * np.sum(y - y_true, 0) / y.shape[0]
        return (y_true - y) ** 2

    def update_weights(self):
        
        self.prev_layer.update_weights(self.dl_dy)

    def __str__(self): return f"{self.name}[MSE Loss]"

if __name__ == "__main__":
    
    #Params
    batch_size = 32
    input_size = 3
    output_size = 2

    model = Dense(input_size, 16)
    model = Dense(16, 8).add_to(model)
    model = Dense(8, output_size).add_to(model)
    mse = MSE().add_to(model)
    model.print_summary()
    print()

    model_true = Dense(input_size, output_size)
    model_true.print_summary()
    print()

    for i in range(10000):
        
        x = np.random.randn(batch_size, input_size)
        y_true = model_true(x)
        y = model(x)
        
        l = mse(y, y_true)
        if i % 100 == 0:
            print("Average Loss:", np.sum(l) / np.size(l))
        
        mse.update_weights()
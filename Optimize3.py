import numpy as np

batch_size = 32
input_size = 3
output_size = 2

class Dense:
    
    def __init__(self, input_size, output_size):
        
        self.input_size = input_size
        self.output_size = output_size
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)
        self.dy_dw = None
        self.dy_db = 1
        self.x = None
        self.w_lr = 0.0001
        self.b_lr = 0.0005
    
    def __call__(self, x):
        
        self.x = x
        self.dy_dw = np.expand_dims(np.sum(x, 0) / x.shape[0], 1)
        return x @ self.w + self.b

    def update_weights(self, partial):
        
        self.w -= self.w_lr * np.ones_like(self.w) * partial * self.dy_dw
        self.b -= self.b_lr * np.ones_like(self.b) * partial * self.dy_db
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
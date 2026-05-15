import numpy as np

batch_size = 32
input_size = 3
output_size = 2

w_true = np.random.randn(input_size, output_size)
b_true = np.random.randn(output_size)

w = np.random.randn(*w_true.shape)
b = np.random.randn(*b_true.shape)

for i in range(1000):

    x = np.random.randn(batch_size, input_size)
    y_true = x @ w_true + b_true
    y = x @ w + b
    
    l = (y_true - y) ** 2
    if i % 100 == 0:
        print("Average Loss:", np.sum(l) / np.size(l))
    
    dl_dy = 2 * np.sum(y - y_true, 0)
    dy_dw = np.expand_dims(np.sum(x, 0), 1)
    
    w_grads = np.ones_like(w) * dl_dy * dy_dw
    b_grads = np.ones_like(b) * dl_dy
    
    w -= 0.0001 * w_grads
    b -= 0.0005 * b_grads

print(w_true)
print(w)

print(b_true)
print(b)
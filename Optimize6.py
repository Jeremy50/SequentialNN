from Optimize5 import Adam, Dense, MSE
import numpy as np

epochs = 10000
batch_size = 32
input_size = 3
output_size = 3

model = Dense(input_size, 16)
model = Dense(16, output_size).add_to(model)
mse = MSE().add_to(model)
model.print_summary()

def f(x):
    linear_comb = np.expand_dims(np.sum(x * np.array([1, -2, 3]), 1), 1)
    average = np.expand_dims(np.sum(x / input_size, 1), 1)
    poly = np.expand_dims(5 - x[:, 1] + 3 * x[:, 2], 1)
    y_true = np.concatenate([linear_comb, average, poly], 1)
    return y_true

print()
print("TRIAL TIME!!!")
x = np.array([[int(input("Enter a number: ")) for i in range(input_size)]])
print("Input:", x[0])
print("Pred:", model(x)[0])
print("True:", f(x)[0])
print()

for _ in range(epochs):
    
    x = np.random.randn(batch_size, input_size)
    y_true = f(x)
    y = model(x)
    
    l = mse(y, y_true)
    if _ % 100 == 0:
        print("Average Loss:", np.sum(l) / np.size(l))
    
    mse.update_weights()

print()
print("TRIAL TIME!!!")
while True:
    x = np.array([[int(input("Enter a number: ")) for i in range(input_size)]])
    print("Input:", x[0])
    print("Pred:", model(x)[0])
    print("True:", f(x)[0])
    print()
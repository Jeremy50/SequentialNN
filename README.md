A simple implementation of a sequential neural network API using Numpy.
Implemented following objects: Dense layer, MSE loss, ADAM optimizer.
Multiple layers can be chained to a model using Layer().add_to(model)
The dense layer was efficiently implemented by exploiting the underlying symmetry of derivatives.
A sample model was generated and used to approximate a linear combination, average, and polynomial.
Optimize1-4 show the process of development.
Optimize5 holds the main classes.
Optimize6 contains the testing/demo program.

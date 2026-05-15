# Overview
A simple implementation of a sequential neural network API using Numpy.

Implemented following objects: Dense layer, MSE loss, ADAM optimizer.

## Breakdown

Optimize1-4 are iterations towards Optimize5.

Optimize5 contains the main classes.

Optimize6 contains the testing/demo program.

Demo.mov is a run of Optimize6.py

Additional notes(pdf and png) are provided towards the developement of this program

## Extra Info

Multiple layers can be chained to a model using Layer().add_to(model)

The dense layer was efficiently implemented by exploiting the underlying symmetry of derivatives.

A sample model was generated and used to approximate a linear combination, average, and polynomial.

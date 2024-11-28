# Basic-GPR-study-for-myself

## Prior

For flexibility, mean values are set to zero. 

The implemented kernel is radial basis function (RBF).

You can see how the covariance matrix based on the kernel is constructed.

Three samples are generated based on the mean values and covariance.

As you can see, the three samples follow the gaussian distribution.


## Posterior

Given data set and new input data, you can predict the distribution of unknown output.

Although it is very simple example, the key point of gaussian process is that it generalizes

the distribution of output through kernel.

## NLL optimization

It is necessary to optimize hyperparameters of kernel.

In this example, L-BFGS is used because it is hard to get appropriate hessian right away.

L-BFGS makes it possible to optimize the parameter by approximating hessian matrix.
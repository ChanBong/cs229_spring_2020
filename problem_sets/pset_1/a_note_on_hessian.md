# Role of Hessian

1. Several nonlinear optimization algorithms for neural networks are based on second order derivative
2. Basis for fast procedure for retraining with small change of training data
3. Identifying least significant weights via inverse of Hessian
4. In Baysian NN
    - Bayesian neural network
    - Central role in Laplace approximation
    - Hessian inverse is used to determine the predictive distribution for a trained network
    - Hessian eigenvalues determine the values of hyperparameters
    - Hessian determinant is used to evaluate the model evidence

## Evaluating the Hessian Matrix

- Full Hessian matrix can be difficult to compute in practice
- Important consideration is efficiency
    - With W parameters (weights and biases) matrix has dimension W x W
    - Efficient methods have O(W2)

## Evaluation Methods

1. Diagonal Approximation
    - In many cases inverse of the hessian is needed
    - If Hessian is approximated by a diagonal matrix (i.e., offdiagonal elements are zero), its inverse is trivially computed
    - Complexity is O(W) rather than O(W^2) for full Hessian

2. Outer product approximation
    - Neural networks commonly use sum-of-squared error functions
    - Can write Hessian matrix in the form
    - Where
    - Elements can be found in O(W2) steps

3. Inverse Hessian
    - Use outer product approximation to obtain computationally efficient procedure for approximating inverse of Hessian

4. Finite Differences
    - Using backprop, complexity is reduced from O(W3) to O(W2)

5. Exact Evaluation of the Hessian
    - Using an extension of backprop
    - Complexity is O(W2)

6. Fast Multiplication by the Hessian
    - Application of the Hessian involve multiplication by the Hessian
    - The vector vTH has only W elements
    - Instead of computing H as an intermediate step, find efficient method to compute product

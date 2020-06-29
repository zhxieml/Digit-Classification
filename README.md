# Digit-Classification
Individual project of CS385 (Machine Learning), SJTU

## Objective

We are supposed to:

- Have deep understanding of basic machine-learning techniques.
- Learn how to design an experiment to prove or verify your ideas.
  - I.e. learn how to ensure the scientific rigor and how to obtain convincing results, when designing experiments.
- Propose new hypotheses or our own understanding of some machine learning methods, and try to verify our opinions through experiments.
  - "Whether the hypotheses themselves are correct or not" is not the most important. Instead, focus on how we prove them correct or incorrect.

## Methods

Several machine learning methods are implemented by myself and conducted on MNIST dataset:

- Logistic regression
- Logistic regression with Ridge loss
- Logistic regression with Lasso loss ([IRLS method](https://ai.stanford.edu/~ang/papers/aaai06-l1logisticregression.pdf))
- Kernel-based logistic regression with Lasso loss
- LDA
- Neural networks
- Gaussian mixture model
- Grad-CAM

## Some Notes

- In stochastic gradient descent, [feature scaling](https://en.wikipedia.org/wiki/Feature_scaling) can sometimes improve the convergence speed of the algorithm ([Why?](https://www.zhihu.com/question/37129350))
- [Efficient L1 Regularized Logistic Regression](https://ai.stanford.edu/~ang/papers/aaai06-l1logisticregression.pdf)
- [Regularization Paths for Generalized Linear Models via Coordinate Descent](http://statweb.stanford.edu/~jhf/ftp/glmnet.pdf)
- [Quadratic Approximation for Logistic Loss](https://myweb.uiowa.edu/pbreheny/7600/s16/notes/4-20.pdf)
- [Soft Thresholding](http://www.scutmath.com/coordiante_descent_for_lasso.html)
- [IRWLS (iteratively re-weighted least squares)](http://hua-zhou.github.io/teaching/biostatm280-2017spring/slides/18-newton/newton.html)
- Considering the stability of numerical solutions, in practice we usually apply singular value decomposition on `S_w` to obtain its inverse. 
- [LDA threshold](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
- [DCGAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
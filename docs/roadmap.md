# Roadmap

A personal roadmap to implement **everything from basic math to transformers/LLMs**.
I would update this checklist with time......

---

## 0. Absolute Basics (Math & Core Utilities)

- [ ] Implement vector & matrix classes
- [ ] Implement matrix–vector and matrix–matrix multiplication
- [ ] Implement transpose, dot product, outer product
- [ ] Implement norms: L1, L2, max norm

- [ ] Implement Gaussian elimination + back-substitution (solve Ax = b)
- [ ] Implement LU decomposition
- [ ] Implement eigenvalue/eigenvector computation (power iteration)
- [ ] Implement simple SVD for small matrices (e.g., via power iteration)

- [ ] Implement numerical derivative (finite differences)
- [ ] Implement numerical gradient for multivariate functions

- [ ] Implement random number generators:
	- [ ] Uniform [0,1)
	- [ ] Gaussian (Box–Muller)
	- [ ] Bernoulli(p)
	- [ ] Sampling from a discrete distribution

---
<!--
## 1. Probability & Statistics Foundations

- [ ] Implement descriptive statistics:
  - [ ] Mean
  - [ ] Variance and standard deviation (sample vs population)
  - [ ] Covariance and correlation

- [ ] Implement probability distributions (with `pdf/pmf`, `logpdf`, `sample`):
  - [ ] Bernoulli
  - [ ] Binomial
  - [ ] Gaussian (univariate)
  - [ ] Gaussian (multivariate)
  - [ ] Exponential
  - [ ] Poisson
  - [ ] Categorical
  - [ ] Multinomial

- [ ] Implement MLE estimators:
  - [ ] Gaussian mean and variance
  - [ ] Bernoulli / Binomial parameter

- [ ] Implement confidence intervals for the mean (normal approximation)

- [ ] Implement Bayesian conjugate updates:
  - [ ] Beta–Bernoulli
  - [ ] Dirichlet–Multinomial
  - [ ] Gaussian prior + Gaussian likelihood (posterior mean/variance)

---

## 2. Optimization From Scratch

- [ ] Implement gradient descent
- [ ] Implement stochastic gradient descent (SGD)
- [ ] Implement mini-batch SGD
- [ ] Implement Momentum
- [ ] Implement Nesterov accelerated gradient
- [ ] Implement RMSProp
- [ ] Implement Adam

- [ ] Implement Newton’s method (with numerical Hessian)
- [ ] Implement basic quasi-Newton (BFGS or L-BFGS) for small problems

- [ ] Implement learning rate schedules:
  - [ ] Step decay
  - [ ] Exponential decay
  - [ ] Cosine schedule

- [ ] Implement early stopping

---

## 3. Core ML Infrastructure

- [ ] Implement dataset utilities:
  - [ ] Train/validation/test split
  - [ ] K-fold cross-validation splitter
  - [ ] Batch iterator / DataLoader

- [ ] Implement preprocessing:
  - [ ] Standardization (z-score)
  - [ ] Min-max scaling
  - [ ] One-hot encoding
  - [ ] Missing value imputation (mean/median/mode)

- [ ] Implement loss functions:
  - [ ] MSE
  - [ ] MAE
  - [ ] Huber loss
  - [ ] Binary cross-entropy
  - [ ] Multiclass cross-entropy
  - [ ] Hinge loss
  - [ ] L1 regularization
  - [ ] L2 regularization
  - [ ] Elastic Net regularization

- [ ] Implement evaluation metrics:
  - [ ] Accuracy
  - [ ] Precision
  - [ ] Recall
  - [ ] F1-score
  - [ ] Confusion matrix
  - [ ] ROC curve points
  - [ ] AUC
  - [ ] RMSE
  - [ ] R² score

- [ ] Define a base `Model` interface (`fit`, `predict`, `save`, `load`)

---

## 4. Classical Supervised Learning

### 4.1 Linear Models

- [ ] Linear Regression:
  - [ ] Closed-form (normal equations)
  - [ ] Gradient descent version
  - [ ] Ridge regression (L2)
  - [ ] Lasso regression (L1, via coordinate descent)
  - [ ] Elastic Net regression

- [ ] Logistic Regression:
  - [ ] Binary logistic regression (cross-entropy)
  - [ ] Gradient descent + regularization
  - [ ] Multiclass softmax regression

- [ ] Perceptron:
  - [ ] Online update rule
  - [ ] Experiment on linearly separable data

- [ ] LDA / QDA:
  - [ ] Linear Discriminant Analysis
  - [ ] Quadratic Discriminant Analysis

---

### 4.2 Non-parametric Methods

- [ ] k-Nearest Neighbors (k-NN):
  - [ ] Classification
  - [ ] Regression
  - [ ] Distance metrics: Euclidean, Manhattan, cosine
  - [ ] Optional: KD-tree / ball tree

- [ ] Naive Bayes:
  - [ ] Gaussian Naive Bayes
  - [ ] Multinomial Naive Bayes
  - [ ] Bernoulli Naive Bayes

---

### 4.3 Margin-Based & Kernel Methods

- [ ] Support Vector Machines (SVM):
  - [ ] Linear SVM (hinge loss + L2 via SGD)
  - [ ] SMO implementation (hard/soft margin)
  - [ ] Kernel SVM:
    - [ ] RBF kernel
    - [ ] Polynomial kernel

---

### 4.4 Trees and Ensembles

- [ ] Decision Trees (CART):
  - [ ] Classification trees (Gini, entropy)
  - [ ] Regression trees (MSE)
  - [ ] Pre-pruning (max depth, min samples)
  - [ ] Optional post-pruning

- [ ] Random Forests:
  - [ ] Bootstrapped sampling
  - [ ] Random feature subsets
  - [ ] Aggregation (voting/averaging)

- [ ] Gradient Boosted Trees:
  - [ ] Basic GBDT for regression (squared loss)
  - [ ] Basic GBDT for classification (logistic loss)

- [ ] Bagging & AdaBoost:
  - [ ] Bagging wrapper
  - [ ] AdaBoost with decision stumps

---

## 5. Unsupervised Learning

### 5.1 Clustering

- [ ] k-means:
  - [ ] Basic algorithm (Lloyd’s)
  - [ ] k-means++ initialization
  - [ ] Elbow method
  - [ ] Silhouette score

- [ ] k-medoids (PAM algorithm)

- [ ] Gaussian Mixture Models (GMM):
  - [ ] EM algorithm (E-step, M-step)
  - [ ] Log-likelihood tracking

- [ ] Hierarchical Clustering:
  - [ ] Agglomerative clustering
  - [ ] Single-link, complete-link, average-link
  - [ ] Dendrogram construction (at least in Python)

- [ ] DBSCAN:
  - [ ] Core, border, noise identification
  - [ ] BFS/DFS cluster expansion

---

### 5.2 Dimensionality Reduction

- [ ] Principal Component Analysis (PCA):
  - [ ] Via covariance + eigen-decomposition
  - [ ] Via SVD
  - [ ] Explained variance curves

- [ ] Kernel PCA (RBF, polynomial)

- [ ] Independent Component Analysis (ICA):
  - [ ] FastICA algorithm

- [ ] t-SNE (basic implementation):
  - [ ] High-dimensional similarities
  - [ ] Low-dimensional similarities
  - [ ] Gradient descent on KL divergence

- [ ] Matrix Factorization:
  - [ ] SVD-based recommender
  - [ ] Non-negative matrix factorization (NMF)

---

## 6. Probabilistic ML & Graphical Models

- [ ] Bayesian Linear Regression:
  - [ ] Gaussian prior on weights
  - [ ] Posterior computation
  - [ ] Predictive distribution

- [ ] Hidden Markov Models (HMMs):
  - [ ] Forward algorithm
  - [ ] Backward algorithm
  - [ ] Viterbi decoding
  - [ ] Baum–Welch (EM) training

- [ ] EM Algorithm Examples:
  - [ ] Mixture of Gaussians (GMM)
  - [ ] Probabilistic PCA

- [ ] MCMC:
  - [ ] Metropolis–Hastings sampler
  - [ ] Gibbs sampling for simple models

- [ ] Variational Inference (basic):
  - [ ] Derive and implement ELBO for a simple model
  - [ ] Mean-field VI with coordinate ascent

- [ ] Latent Dirichlet Allocation (LDA):
  - [ ] Collapsed Gibbs sampler for topic modeling

---

## 7. Deep Learning Foundations

### 7.1 Autograd & Tensor

- [ ] Tensor class (N-dimensional):
  - [ ] Basic ops: +, -, *, /, matmul
  - [ ] Sum, mean, transpose, reshape, slicing

- [ ] Reverse-mode Autograd Engine:
  - [ ] Computational graph construction
  - [ ] Backward methods for each operation
  - [ ] Gradient accumulation

- [ ] Module / Layer System:
  - [ ] `Module` base class with `parameters()` and `zero_grad()`
  - [ ] `Linear` layer
  - [ ] Activation layers:
    - [ ] ReLU
    - [ ] LeakyReLU
    - [ ] Sigmoid
    - [ ] Tanh
    - [ ] Softmax
  - [ ] Dropout layer
  - [ ] BatchNorm
  - [ ] LayerNorm

- [ ] Training loop:
  - [ ] Mini-batch training
  - [ ] Use earlier optimizers (SGD, Adam, etc.)

---

### 7.2 Basic Neural Networks

- [ ] MLP for regression
- [ ] MLP for binary classification
- [ ] MLP for multiclass classification

- [ ] Regularization techniques:
  - [ ] L2 weight decay
  - [ ] Dropout
  - [ ] Early stopping

---

## 8. Convolutional & Sequence Models

### 8.1 CNNs

- [ ] Conv2D layer:
  - [ ] Manual convolution (loops)
  - [ ] Stride, padding, dilation
  - [ ] Backprop for convolution

- [ ] Pooling layers:
  - [ ] Max pooling
  - [ ] Average pooling

- [ ] Architectures:
  - [ ] LeNet-5
  - [ ] Small VGG-style network
  - [ ] Small ResNet with residual blocks

- [ ] Data augmentation:
  - [ ] Random crop
  - [ ] Horizontal flip
  - [ ] Random rotation
  - [ ] Channel-wise normalization

---

### 8.2 RNNs

- [ ] Vanilla RNN cell
- [ ] Backprop Through Time (BPTT)

- [ ] LSTM:
  - [ ] LSTM cell
  - [ ] Sequence modeling experiments

- [ ] GRU:
  - [ ] GRU cell
  - [ ] Sequence modeling experiments

- [ ] Seq2Seq:
  - [ ] Encoder–decoder with RNN/LSTM
  - [ ] Greedy decoding
  - [ ] Toy tasks (sequence reversal, addition, simple translation)

---

## 9. Transformers & LLM Building Blocks

- [ ] Scaled dot-product attention
- [ ] Multi-head attention
- [ ] Masked self-attention (for autoregressive models)

- [ ] Positional encodings:
  - [ ] Sinusoidal positional encodings

- [ ] Transformer encoder block:
  - [ ] Self-attention
  - [ ] Feedforward network
  - [ ] Residual connections
  - [ ] LayerNorm

- [ ] Transformer decoder block:
  - [ ] Masked self-attention
  - [ ] Cross-attention
  - [ ] Feedforward + residual + LayerNorm

- [ ] Mini-transformer models:
  - [ ] Encoder-only transformer (classification / embeddings)
  - [ ] Decoder-only (GPT-style) language model:
    - [ ] Character-level language model
    - [ ] Train on a small text corpus

- [ ] Tokenization:
  - [ ] Basic whitespace + punctuation tokenizer
  - [ ] Byte Pair Encoding (BPE) or WordPiece
  - [ ] Vocab building + encode/decode

---

## 10. Autoencoders, VAEs, GANs

- [ ] Autoencoders:
  - [ ] Fully-connected AE on MNIST
  - [ ] Denoising autoencoder

- [ ] Variational Autoencoder (VAE):
  - [ ] Reparameterization trick
  - [ ] KL divergence term in loss

- [ ] GANs:
  - [ ] Vanilla GAN on 2D toy data
  - [ ] DCGAN on simple image dataset (e.g., MNIST)

---

## 11. Graph & Geometric Deep Learning (Optional)

- [ ] Graph representation:
  - [ ] Adjacency matrix
  - [ ] Adjacency list
  - [ ] Mini-batch handling

- [ ] Graph Convolutional Network (GCN):
  - [ ] Implement Â H W formulation
  - [ ] Node classification

- [ ] Graph classification via pooling/readout

---

## 12. NLP-Specific Models

- [ ] Classical NLP:
  - [ ] Bag-of-words representation
  - [ ] TF–IDF vectorizer
  - [ ] N-gram language models with smoothing

- [ ] Word embeddings:
  - [ ] word2vec Skip-gram with negative sampling
  - [ ] CBOW
  - [ ] Optional: GloVe-style embeddings

- [ ] Text classification pipelines:
  - [ ] TF–IDF + logistic regression
  - [ ] RNN-based classifier
  - [ ] CNN-based classifier
  - [ ] Transformer encoder classifier

---

## 13. Reinforcement Learning

- [ ] MDP utilities (states, actions, rewards, transitions)

- [ ] Tabular RL:
  - [ ] Policy evaluation
  - [ ] Policy improvement
  - [ ] Value iteration
  - [ ] Policy iteration

- [ ] Model-free RL:
  - [ ] Monte Carlo control
  - [ ] TD(0), TD(λ)
  - [ ] SARSA
  - [ ] Q-learning

- [ ] Deep RL:
  - [ ] DQN (experience replay, target network)
  - [ ] Policy gradient (REINFORCE)
  - [ ] Basic Actor–Critic

---

## 14. Engineering & Framework-Level Stuff

- [ ] Model serialization:
  - [ ] Save/load weights to binary format
  - [ ] Load configs from JSON/YAML

- [ ] Experiment tracking:
  - [ ] Logging training metrics
  - [ ] Plotting loss and accuracy curves

- [ ] C++ performance:
  - [ ] Template-based numeric code
  - [ ] Simple BLAS-like routines
  - [ ] Optional: Multithreaded matrix operations (OpenMP or similar)

-->


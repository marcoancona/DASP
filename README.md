# Deep Approximate Shapley Propagation

This repository contains the Keras implementation for Deep Approximate Shapley Propagation (DASP).

The problem of explaining the behavior of deep neural networks has gained a lot of attention over the last years. 
*Shapley values* as a unique way of assigning relevance scores such that certain desirable properties are satisfied.
Unfortunately, the exact evaluation of Shapley values is prohibitively expensive, exponential in the number of input features. DASP is a polynomial-time approximation of Shapley values designed for deep neural networks. It relies on uncertainty propagation using [Lightweight Probabilistic Deep Networks (LPDN)](https://arxiv.org/abs/1805.11327) to approximate Shapley values. This libraries relies on a [Keras implementation of such probabilistic framework](https://github.com/marcoancona/LPDN).

## How to use
Please see the [example folder](https://github.com/marcoancona/DASP/tree/master/examples) for practical usage example and results.

## License
MIT

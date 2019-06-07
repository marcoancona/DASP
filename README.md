# Deep Approximate Shapley Propagation

This repository contains the Keras implementation for [Deep Approximate Shapley Propagation (DASP)](https://arxiv.org/abs/1903.10992), "Explaining Deep Neural Networks with a Polynomial Time Algorithm for Shapley Values Approximation", ICML 2019.

The problem of explaining the behavior of deep neural networks has gained a lot of attention over the last years. 
*Shapley values* as a unique way of assigning relevance scores such that certain desirable properties are satisfied.
Unfortunately, the exact evaluation of Shapley values is prohibitively expensive, exponential in the number of input features. DASP is a polynomial-time approximation of Shapley values designed for deep neural networks. It relies on uncertainty propagation using [Lightweight Probabilistic Deep Networks (LPDN)](https://arxiv.org/abs/1805.11327) to approximate Shapley values. This libraries relies on a [Keras implementation of such probabilistic framework](https://github.com/marcoancona/LPDN).

## How to use
Please see the [example folder](https://github.com/marcoancona/DASP/tree/master/examples) for practical usage example and results.

## Citation
```
@InProceedings{ancona19a,
  title = 	 {Explaining Deep Neural Networks with a Polynomial Time Algorithm for Shapley Value Approximation},
  author = 	 {Ancona, Marco and Oztireli, Cengiz and Gross, Markus},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {272--281},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Long Beach, California, USA},
  month = 	 {09--15 Jun},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v97/ancona19a/ancona19a.pdf},
  url = 	 {http://proceedings.mlr.press/v97/ancona19a.html},
  abstract = 	 {The problem of explaining the behavior of deep neural networks has recently gained a lot of attention. While several attribution methods have been proposed, most come without strong theoretical foundations, which raises questions about their reliability. On the other hand, the literature on cooperative game theory suggests Shapley values as a unique way of assigning relevance scores such that certain desirable properties are satisfied. Unfortunately, the exact evaluation of Shapley values is prohibitively expensive, exponential in the number of input features. In this work, by leveraging recent results on uncertainty propagation, we propose a novel, polynomial-time approximation of Shapley values in deep neural networks. We show that our method produces significantly better approximations of Shapley values than existing state-of-the-art attribution methods.}
}
```

## License
MIT

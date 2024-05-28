# Utility maximisation of NN hedging strategies

As a model the standard single-asset Black scholes model is used

$$dS_t = S_t(\mu dt + \sigma dW_t)$$

over a period of 1 year, initial value $S_0 = 100$ with a return of $\mu = 8\%$ and a volatility of $\sigma = 20\%$.
 
The objective it to optimise the expected utility of the terminal wealth 
$$\mathbb{E}[U(V_T (H))]$$
after $T = 1$ year where $U : \mathbb{R} \rightarrow \mathbb{R}$ is a utility function. We will assume that the initial wealth is

$$V_0 = 100$$
 
The key is to implement a new loss function $L$ such that

$$
L(V_T(H), 0) = −U(V_T(H))
$$

via a custom loss function. Since the correct values for $V_T(H)$ are unknown they cannot appear in the output of the loss function, hence, we simply compare to zero.
 
We are going to analyse the performance of the algorithm for the following hedges:
1. $U(x) = \log(x)$, log-utility.
2. $U(x) = x^p$ for $p = 0.5,$ power utility.
3. $U(x) = \frac{1−e^{−\alpha x}}{\alpha}$ for $\alpha = 0.005$, exponential utility.


## Initial setup

### Black Scholes (BS) Model

In our model, we used P-dynamics with a positive drift. It shall be noted that the drift parameter has a big influence on the hedging behavior of the model. If the drift was negative, the neural network (NN) would learn to (short-)sell the asset immediately, recognizing it as a risk (because of the volatility) with negative return.

### Hedge NN

We decided to restrict the hedge NN to return only hedge ratios between 0 and 1 by using tangens hyperbolicus (`tanh`) as the activation function for the output (`sigmoid` or other functions could work as well). This can have an economic interpretation: we do not want our model to short-sell or to buy more than 100% of the asset. Despite this constraint, the results remain meaningful and interpretable.

## Utility functions

We can see from the plots below that all three utility functions are concave, which implies risk aversion. This means that the marginal utility decreases as wealth increases, and that the negative impact of losses on the utility is larger than the positive impact of gains of the same magnitude.

Since the functions are all concave, we expect similar hedging behavior, despite their different degrees of concavity - function (1) is the most concave and function (3) the least concave.

# Utility functions

```{python}
def U1(x):
    return tf.math.log(x)

def U2(x):
    return tf.math.sqrt(x)

def U3(x):
    return tf.math.divide(tf.math.subtract(tf.constant(1, dtype=tf.float32), tf.math.exp(tf.math.multiply(tf.constant(-0.005, dtype=tf.float32), x))), tf.constant(0.005, dtype=tf.float32))
```

## Construction of price paths using BS-model

## Neural networks

Since we only consider a single asset (and no derivative written on it), we define the payoff function simply as the final value of the asset price path.

### Loss functions for the networks

```{python}
# Define loss functions
def L1(y_true, y_predict):
    return tf.reduce_mean(-U1(y_predict - y_true))


def L2(y_true, y_predict):
    return tf.reduce_mean(-U2(y_predict - y_true))


def L3(y_true, y_predict):
    return tf.reduce_mean(-U3(y_predict - y_true))
```

### Network for initial wealth

### Network for hedging position

### Network for wealth

### Model training

### Model testing

## Results

### Terminal Wealth

Our results indicate that the wealth distribution of our 3 NNs are (nearly) the same, because they used (almost) the same hedging strategy!

```
Setup cost (NN): 100.0

Test data analysis:
Standard deviation (NN L1 Loss): 21.55
Mean sample error (NN L1 Loss): 108.41


Standard deviation (NN L2 Loss): 21.55
Mean sample error (NN L2 Loss): 108.41


Standard deviation (NN L3 Loss): 21.55
Mean sample error (NN L3 Loss): 108.41
```

### Hedging Strategy

We can see that the hedging strategy seems to be the same for all utility functions, namely buy and hold (the hedge ratio is 1 at any point in time). This makes sense: an (average) return of $\mu=8$\% appears to be enough to convince all three risk-averse investors that it is worthwhile to buy and hold the asset.

As mentioned in the beginning, the $\mu$ is the important parameter here: it needs to be large enough so that it compensates the risk stemming from $\sigma$ (of course, the size of $\sigma$ is also important in this sense). It should also be noted that it is possible to find a $\mu$ such that the most risk averse NN (1) would not invest into the asset and the least averse NN (3) would.



# %% [markdown]

# %% [markdown]
# As a model, the standard single-asset Black scholes model is used
# $$
# dS_t = S_t(\mu dt + \sigma dW_t)
# $$
# over a period of 1 year, initial value $S_0 = 100$ with a return of $\mu = 8\%$ and a volatility of $\sigma = 20\%$.
# 
# The objective it to optimise the expected utility of the terminal wealth 
# $$\mathbb{E}[U(V_T (H))]$$
# after $T = 1$ year where $U : \mathbb{R} \rightarrow \mathbb{R}$ is a utility function. We will assume that the initial wealth is
# $$
# V_0 = 100
# $$
# 
# The key is to implement a new loss function $L$ such that
# $$
# L(V_T(H), 0) = −U(V_T(H))
# $$
# via a custom loss function. Since the correct values for $V_T(H)$ are unknown they cannot appear in the output of the loss function and we simply compare to zero.
# 
# The performance of the algorithm for the following hedges is analysided:
# 1. $U(x) = \log(x)$, log-utility.
# 2. $U(x) = x^p$ for $p = 0.5,$ power utility.
# 3. $U(x) = \frac{1−e^{−\alpha x}}{\alpha}$ for $\alpha = 0.005$, exponential utility.

# %% [markdown]
# ## Initial setup

# %%
# Importing necessary libraries
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Subtract, Multiply, Add, Dot, Concatenate
from keras.models import Model
import matplotlib.pyplot as plt

np.random.seed(10538)

# %%
# Setting up parameters
Q_dynamics = False  # Flag for Q dynamics

if Q_dynamics:
    mu = 0  # Drift parameter for Q dynamics
else:
    mu = 0.08  # Drift parameter for P dynamics

S0 = 100  # Initial price of the underlying asset
N = 260  # Number of time steps
sigma = 0.2  # Volatility
T = 1  # Time to maturity
V0 = 100  # Initial wealth
activator_hidden = "tanh"  # Activation function for neural networks
activator_output = "tanh"   # Activation function for the output layer
R = 5  # Number of shown trajectories (not used for training/testing)
m = 1  # Dimension of price
prec = 2 # Precision for printing

# %%
# Train/Test setup
Ktrain = 50000  # Size of training data
Ktest = 10000  # Size of test data
epochs = 20  # Number of training epochs

# Network structure
learnV0 = False  # Learn setup wealth. But we already know V_0 = 100!
d = 2  # Number of hidden layers in strategy
n = 200  # Number of nodes in hidden layers

# Define time points
TimePoints = np.linspace(0, T, N+1)

# Time converter function
def TimeConv(t):
    """Converts time to maturity."""
    return T - t

# %% [markdown]
# ## Utility functions

# %%
# Utility functions
def U1(x):
    return tf.math.log(x)

def U2(x):
    return tf.math.sqrt(x)

def U3(x):
    return tf.math.divide(tf.math.subtract(tf.constant(1, dtype=tf.float32), tf.math.exp(tf.math.multiply(tf.constant(-0.005, dtype=tf.float32), x))), tf.constant(0.005, dtype=tf.float32))

# %%
# Generate sample data
x_values = np.linspace(0.1, 200, 500)

# Plot utility functions
plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.plot(x_values, U1(x_values))
plt.title('Utility Function 1')
plt.xlabel('Wealth')
plt.ylabel('Utility')

plt.subplot(132)
plt.plot(x_values, U2(x_values))
plt.title('Utility Function 2')
plt.xlabel('Wealth')
plt.ylabel('Utility')

plt.subplot(133)
plt.plot(x_values, U3(x_values))
plt.title('Utility Function 3')
plt.xlabel('Wealth')
plt.ylabel('Utility')

plt.tight_layout()
plt.savefig('out/fig1.png')
plt.show()

# %% [markdown]
# ## Construction of price paths using BS-model

# %%
# Define price model function
def path(S0, mu, sigma, Timepoints, R):
    """Generates sample paths from the price model."""
    N = len(Timepoints) - 1
    X = np.zeros((N+1, R)) + np.log(S0)
    mu_log = mu - sigma**2/2
    for j in range(N):
        dt = Timepoints[j+1] - Timepoints[j]
        dZ = np.random.normal(0, 1, R)
        increment = mu_log * dt + sigma * dZ * np.sqrt(dt)
        X[j+1, :] = X[j, :] + increment
    return np.exp(X)

# %%
# Generate sample paths from the model
S = path(S0, mu, sigma, TimePoints, R)

# Plot sample paths
plt.figure(figsize=(12, 6))
for i in range(R):
    plt.plot(TimePoints, S[:, i])
plt.title(str(R) + " sample paths from our model")
plt.xlabel("Time")
plt.ylabel("Price")
plt.savefig('out/fig2.png')
plt.show()

# %%
# Plot utility of sample paths
plt.figure(figsize=(12, 6))

plt.subplot(131)
for i in range(R):
    plt.plot(TimePoints, U1(S[:, i]))
plt.title("Utility Function 1 for " + str(R) + " sample paths")
plt.xlabel("Time")
plt.ylabel("Utility")

plt.subplot(132)
for i in range(R):
    plt.plot(TimePoints, U2(S[:, i]))
plt.title("Utility Function 2 for " + str(R) + " sample paths")
plt.xlabel("Time")
plt.ylabel("Utility")

plt.subplot(133)
for i in range(R):
    plt.plot(TimePoints, U3(S[:, i]))
plt.title("Utility Function 3 for " + str(R) + " sample paths")
plt.xlabel("Time")
plt.ylabel("Utility")


plt.tight_layout()
plt.savefig('out/fig3.png')
plt.show()

# %% [markdown]
# ## Neural networks

# %%
# Define payoff function
def f(S):
    return S[N, :]

# %% [markdown]
# ### Loss functions for the networks

# %%
# Define loss functions
def L1(y_true, y_predict):
    return tf.reduce_mean(-U1(y_predict - y_true))


def L2(y_true, y_predict):
    return tf.reduce_mean(-U2(y_predict - y_true))


def L3(y_true, y_predict):
    return tf.reduce_mean(-U3(y_predict - y_true))

# %% [markdown]
# ### Network for initial wealth

# %%
# Define network for initial wealth
d_V0 = 0  # Number of hidden layers

price0_L1 = Input(shape=(m,))
V0_L1 = price0_L1
for i in range(d_V0):
    V0_L1 = Dense(1, activation=activator_hidden)(V0_L1)
V0_L1 = Dense(1, activation='linear', trainable=learnV0)(V0_L1)
pi_L1 = Model(inputs=price0_L1, outputs=V0_L1)

price0_L2 = Input(shape=(m,))
V0_L2 = price0_L2
for i in range(d_V0):
    V0_L2 = Dense(1, activation=activator_hidden)(V0_L2)
V0_L2 = Dense(1, activation='linear', trainable=learnV0)(V0_L2)
pi_L2 = Model(inputs=price0_L2, outputs=V0_L2)

price0_L3 = Input(shape=(m,))
V0_L3 = price0_L3
for i in range(d_V0):
    V0_L3 = Dense(1, activation=activator_hidden)(V0_L3)
V0_L3 = Dense(1, activation='linear', trainable=learnV0)(V0_L3)
pi_L3 = Model(inputs=price0_L3, outputs=V0_L3)

# %% [markdown]
# ### Network for hedging position

# %%
# Define network for hedging position
timeprice_L1 = Input(shape=(1+m,))
output_L1 = timeprice_L1
for i in range(d):
    output_L1 = Dense(n, activation=activator_hidden)(output_L1)
output_L1 = Dense(m, activation=activator_output)(output_L1)
hedge_L1 = Model(inputs=timeprice_L1, outputs=output_L1)

timeprice_L2 = Input(shape=(1+m,))
output_L2 = timeprice_L2
for i in range(d):
    output_L2 = Dense(n, activation=activator_hidden)(output_L2)
output_L2 = Dense(m, activation=activator_output)(output_L2)
hedge_L2 = Model(inputs=timeprice_L2, outputs=output_L2)

timeprice_L3 = Input(shape=(1+m,))
output_L3 = timeprice_L3
for i in range(d):
    output_L3 = Dense(n, activation=activator_hidden)(output_L3)
output_L3 = Dense(m, activation=activator_output)(output_L3)
hedge_L3 = Model(inputs=timeprice_L3, outputs=output_L3)

# %% [markdown]
# ### Network for wealth

# %%
# Define network for wealth
time_L1 = Input(shape=(1,))
price_L1 = Input(shape=(m,))
inputs_L1 = [time_L1, price_L1]
wealth_L1 = pi_L1(price_L1)
for j in range(N):
    timenew_L1 = Input(shape=(1,))
    pricenew_L1 = Input(shape=(m,))
    inputs_L1 = inputs_L1 + [timenew_L1, pricenew_L1]
    priceshift_L1 = Subtract()([pricenew_L1, price_L1])
    strategy_L1 = hedge_L1(Concatenate()([time_L1, price_L1]))
    if (m == 1):
        gains_L1 = Multiply()([strategy_L1, priceshift_L1])  
    if (m > 1):
        # For multi-asset setting
        gains_L1 = Dot()([strategy_L1, priceshift_L1])
    wealth_L1 = Add()([wealth_L1, gains_L1])
    price_L1 = pricenew_L1
    time_L1 = timenew_L1

time_L2 = Input(shape=(1,))
price_L2 = Input(shape=(m,))
inputs_L2 = [time_L2, price_L2]
wealth_L2 = pi_L2(price_L2)
for j in range(N):
    timenew_L2 = Input(shape=(1,))
    pricenew_L2 = Input(shape=(m,))
    inputs_L2 = inputs_L2 + [timenew_L2, pricenew_L2]
    priceshift_L2 = Subtract()([pricenew_L2, price_L2])
    strategy_L2 = hedge_L2(Concatenate()([time_L2, price_L2]))
    if (m == 1):
        gains_L2 = Multiply()([strategy_L2, priceshift_L2])  
    if (m > 1):
        # For multi-asset setting
        gains_L2 = Dot()([strategy_L2, priceshift_L2])
    wealth_L2 = Add()([wealth_L2, gains_L2])
    price_L2 = pricenew_L2
    time_L2 = timenew_L2

time_L3= Input(shape=(1,))
price_L3 = Input(shape=(m,))
inputs_L3= [time_L3, price_L3]
wealth_L3 = pi_L3(price_L3)
for j in range(N):
    timenew_L3 = Input(shape=(1,))
    pricenew_L3 = Input(shape=(m,))
    inputs_L3 = inputs_L3 + [timenew_L3, pricenew_L3]
    priceshift_L3 = Subtract()([pricenew_L3, price_L3])
    strategy_L3 = hedge_L3(Concatenate()([time_L3, price_L3]))
    if (m == 1):
        gains_L3 = Multiply()([strategy_L3, priceshift_L3])  
    if (m > 1):
        # For multi-asset setting
        gains_L3 = Dot()([strategy_L3, priceshift_L3])
    wealth_L3 = Add()([wealth_L3, gains_L3])
    price_L3 = pricenew_L3
    time_L3 = timenew_L3

# %%
# Compile networks
model_wealth_L1 = Model(inputs=inputs_L1, outputs=wealth_L1)
model_wealth_L1.compile(optimizer='adam', loss=L1)

model_wealth_L2 = Model(inputs=inputs_L2, outputs=wealth_L2)
model_wealth_L2.compile(optimizer='adam', loss=L2)

model_wealth_L3 = Model(inputs=inputs_L3, outputs=wealth_L3)
model_wealth_L3.compile(optimizer='adam', loss=L3)

# %% [markdown]
# ### Model training

# %%
# Train the models
trainpathes = path(S0, mu, sigma, TimePoints, Ktrain)
xtrain = []
for i in range(N+1):
    TimeToMaturity = np.repeat(TimeConv(TimePoints[i]), Ktrain)
    xtrain = xtrain + [TimeToMaturity, trainpathes[i, :]]
ytrain = np.zeros(shape=(Ktrain))

V0_train = 100
weights_new = [np.array([[0]]), np.array([V0_train])]

pi_L1.set_weights(weights_new)
pi_L2.set_weights(weights_new)
pi_L3.set_weights(weights_new)

print("\nPre-setting initial wealth for NN-hedge to:", V0_train)
print("\nStart of training ...")
maxL = 0
stepsize = 5
steps = int(np.round(epochs/stepsize))

for i in range(steps):
    model_wealth_L1.fit(x=xtrain, y=ytrain, epochs=stepsize*(i+1), initial_epoch=stepsize*i, verbose=0, batch_size=100*(1+i))
    evalu_L1 = model_wealth_L1.evaluate(xtrain, ytrain, verbose=False, batch_size=1000)
    print("\nEpoch:", (i+1)*stepsize, "/", epochs)
    print("Loss:", evalu_L1)
    if evalu_L1 < maxL:
        break
print("Training completed for L1.")

for i in range(steps):
    model_wealth_L2.fit(x=xtrain, y=ytrain, epochs=stepsize*(i+1), initial_epoch=stepsize*i, verbose=0, batch_size=100*(1+i))
    evalu_L2 = model_wealth_L2.evaluate(xtrain, ytrain, verbose=False, batch_size=1000)
    print("\nEpoch:", (i+1)*stepsize, "/", epochs)
    print("Loss:", evalu_L2)
    if evalu_L2 < maxL:
        break
print("Training completed for L2.")

for i in range(steps):
    model_wealth_L3.fit(x=xtrain, y=ytrain, epochs=stepsize*(i+1), initial_epoch=stepsize*i, verbose=0, batch_size=100*(1+i))
    evalu_L3 = model_wealth_L3.evaluate(xtrain, ytrain, verbose=False, batch_size=1000)
    print("\nEpoch:", (i+1)*stepsize, "/", epochs)
    print("Loss:", evalu_L3)
    if evalu_L3 < maxL:
        break
print("Training completed for L3.")

# %% [markdown]
# ### Model testing

# %%
# Testing the models
testpathes = path(S0, mu, sigma, TimePoints, Ktest)
xtest = []
for i in range(N+1):
    TimeToMaturity = np.repeat(TimeConv(TimePoints[i]), Ktest)
    xtest = xtest + [TimeToMaturity, testpathes[i, :]]
ytest = np.zeros(shape=(Ktest))

V0test_L1 = pi_L1.predict(testpathes[0, :], verbose=0)[:, 0]
V0diff_L1 = np.mean(V0test_L1)

V0test_L2 = pi_L2.predict(testpathes[0, :], verbose=0)[:, 0]
V0diff_L2 = np.mean(V0test_L2)

V0test_L3 = pi_L3.predict(testpathes[0, :], verbose=0)[:, 0]
V0diff_L3 = np.mean(V0test_L3)

model_wealth_L1.evaluate(x=xtest, y=ytest, verbose=0)
difftest_L1 =  model_wealth_L1.predict(xtest, verbose=0)[:, 0] - ytest

model_wealth_L2.evaluate(x=xtest, y=ytest, verbose=0)
difftest_L2 =  model_wealth_L2.predict(xtest, verbose=0)[:, 0] - ytest

model_wealth_L3.evaluate(x=xtest, y=ytest, verbose=0)
difftest_L3 =  model_wealth_L3.predict(xtest, verbose=0)[:, 0] - ytest

# %% [markdown]
# ## Results

# %%
# Visualizing results
plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.hist(difftest_L1, bins=50)
plt.title('Hisgoram of NN wealth with L1 loss')
plt.xlabel('Wealth')
plt.ylabel('Frequency')

plt.subplot(132)
plt.hist(difftest_L2, bins=50)
plt.title('Hisgoram of NN wealth with L2 loss')
plt.xlabel('Wealth')
plt.ylabel('Frequency')

plt.subplot(133)
plt.hist(difftest_L3, bins=50)
plt.title('Hisgoram of NN wealth with L3 loss')
plt.xlabel('Wealth')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('out/fig4.png')
plt.show()

# %%
# Analyzing results
print("Setup cost (NN):", round(V0test_L1[0], prec))

print("\nTest data analysis:")

print("Standard deviation (NN L1 Loss):", round(np.std(difftest_L1), prec))
print("Mean sample error (NN L1 Loss):", round(np.mean(difftest_L1), prec))
print("\n")

print("Standard deviation (NN L2 Loss):", round(np.std(difftest_L2), prec))
print("Mean sample error (NN L2 Loss):", round(np.mean(difftest_L2), prec))
print("\n")

print("Standard deviation (NN L3 Loss):", round(np.std(difftest_L3), prec))
print("Mean sample error (NN L3 Loss):", round(np.mean(difftest_L3), prec))

# %%
def Comparehedge(t=0, showBS=True, sameArea=False, loss='L1'):
    for i in range(N):
        if TimePoints[i] <= t:
            k = i
    t = TimePoints[k]
    Svals = testpathes[k,]
    if sameArea:
        a = np.min(testpathes)
        b = np.max(testpathes)
        Svals = np.linspace(a, b, Ktest)
    timeprice = np.concatenate((np.reshape(np.repeat(TimeConv(TimePoints[k]), Ktest), (Ktest, 1)), np.reshape(Svals, (Ktest, 1))), axis=1)
    if loss == 'L2':
        h_NN = hedge_L2.predict(timeprice, verbose=0)[:,0]
    elif loss == 'L3':
        h_NN = hedge_L3.predict(timeprice, verbose=0)[:,0]
    else : 
        h_NN = hedge_L1.predict(timeprice, verbose=0)[:,0]
    
    return Svals, h_NN, round(T - t, prec)  # Return values for plotting

# %%
# Define time points for plotting
tshow = np.array(T) * np.array((0.1, 0.25, 0.5, 0.75, 0.9))

# Create subplots
num_plots = len(tshow)
cols = 3  # Number of columns in the grid
rows = num_plots // cols + (1 if num_plots % cols > 0 else 0)  # Calculate number of rows

# %%
# Set figure size based on the number of rows
plt.figure(figsize=(15, 5 * rows))  

# Plot each result in a separate subplot
for i, t in enumerate(tshow, start=1):
    plt.subplot(rows, cols, i)
    Svals, h_NN, content_str = Comparehedge(t, True, False, 'L1')
    plt.plot(Svals, h_NN, 'o')
    plt.title("NN hedging position time to maturity: " + str(content_str))
    plt.xlabel("Stock price")
    plt.ylabel("Hedging position")

plt.suptitle('NN hedging with L1 loss')
plt.tight_layout()
plt.savefig('out/fig5.png')
plt.show()

# %%
# Set figure size based on the number of rows
plt.figure(figsize=(15, 5 * rows))  

# Plot each result in a separate subplot
for i, t in enumerate(tshow, start=1):
    plt.subplot(rows, cols, i)
    Svals, h_NN, content_str = Comparehedge(t, True, False, 'L2')
    plt.plot(Svals, h_NN, 'o')
    plt.title("NN hedging position time to maturity: " + str(content_str))
    plt.xlabel("Stock price")
    plt.ylabel("Hedging position")

plt.suptitle('NN hedging with L2 loss')
plt.tight_layout()
plt.savefig('out/fig6.png')
plt.show()

# %%
# Set figure size based on the number of rows
plt.figure(figsize=(15, 5 * rows))  

# Plot each result in a separate subplot
for i, t in enumerate(tshow, start=1):
    plt.subplot(rows, cols, i)
    Svals, h_NN, content_str = Comparehedge(t, True, False, 'L3')
    plt.plot(Svals, h_NN, 'o')
    plt.title("NN hedging position time to maturity: " + str(content_str))
    plt.xlabel("Stock price")
    plt.ylabel("Hedging position")

plt.suptitle('NN hedging with L3 loss')
plt.tight_layout()
plt.savefig('out/fig7.png')
plt.show()


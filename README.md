# neural-networks-spam-and-classification
import numpy as np

# ---------------------------------------------------
# FEEDFORWARD NETWORK (1 hidden layer, ReLU + Sigmoid)
# ---------------------------------------------------
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def feedforward(X):
    # weights for feedforward network
    W1 = np.array([0.6, -0.1, 0.5])
    b1 = 0.2

    W2 = np.array([0.3, 0.4, -0.6])
    b2 = -0.1

    # ---- Hidden neuron 1 ----
    z1 = np.dot(X, W1) + b1
    h1_out = relu(z1)

    # ---- Hidden neuron 2 ----
    z2 = np.dot(X, W2) + b2
    h2_out = sigmoid(z2)

    return h1_out, h2_out


# ---------------------------------------------------
# ORIGINAL INPUT + ORIGINAL h1, h2 from your example
# ---------------------------------------------------
X = np.array([1, 0, 1])

# Your original network weights
h1 = np.array([0.5, -0.2, 0.3])
h2 = np.array([0.4, 0.1, -0.5])

# --------------------------
# STEP 1: Neuron outputs
# --------------------------
h1_out = np.dot(X, h1)
h2_out = np.dot(X, h2)
h = np.array([h1_out, h2_out])   # [0.8, -0.1]

print("\n=== ORIGINAL HIDDEN OUTPUTS ===")
print("h1 =", h1_out)
print("h2 =", h2_out)


# ---------------------------------------------------
# PART A → FEEDFORWARD NETWORK
# ---------------------------------------------------
ff_h1, ff_h2 = feedforward(X)

print("\n=== FEEDFORWARD OUTPUTS ===")
print("FeedForward h1 =", ff_h1)
print("FeedForward h2 =", ff_h2)


# ---------------------------------------------------
# PART B → SPAN / ATTENTION
# ---------------------------------------------------
scores = np.array([1.2, 0.3])
exp_scores = np.exp(scores)
weights = exp_scores / np.sum(exp_scores)

span_output = np.sum(h * weights)

print("\n=== ATTENTION (SPAN) ===")
print("Attention Weights =", weights)
print("SPAN Output =", span_output)


# ---------------------------------------------------
# PART C → CLASSIFICATION
# ---------------------------------------------------
W = np.array([
    [1.0, -1.0],   # class 0
    [-0.5, 0.8]    # class 1
])

scores_class = np.dot(W, h)
pred_class = np.argmax(scores_class)

print("\n=== CLASSIFICATION ===")
print("Class Scores =", scores_class)
print("Predicted Class =", pred_class)

import numpy as np
from sig import sig, sig_prime
from tanh import tanh, tanh_prime
from mse import mse, mse_prime

# Inicialização
def init_params():
    W1 = np.random.randn(3)
    W2 = np.random.randn(3)
    W3 = np.random.randn(3)
    weights = np.array([W1, W2, W3], dtype=object)
    b = 1
    return weights, b

# Forward pass
def forward(weights, b, X):
    Z1 = weights[0].dot(X) + b
    A1 = tanh(Z1)

    Z2 = weights[1].dot(X) + b
    A2 = tanh(Z2)

    H = np.array([b, A1, A2])  # entrada para saída
    Z3 = weights[2].dot(H)
    A3 = sig(Z3)

    cache = {
        'X': X,  'Z1': Z1, 'A1': A1,
        'Z2': Z2, 'A2': A2,
        'H': H, 'Z3': Z3, 'A3': A3
    }

    return A3, cache

# Backpropagation
def backprop(weights, cache, yt):
    dC_da3 = mse_prime(yt, cache['A3'])
    da3_dZ3 = sig_prime(cache['Z3'])
    dZ3_dw3 = cache['H']
    dC_dw3 = dC_da3 * da3_dZ3 * dZ3_dw3

    dZ3_da1 = weights[2][1]
    da1_dZ1 = tanh_prime(cache['Z1'])
    dZ1_dw1 = cache['X']
    dC_dw1 = dC_da3 * da3_dZ3 * dZ3_da1 * da1_dZ1 * dZ1_dw1

    dZ3_da2 = weights[2][2]
    da2_dZ2 = tanh_prime(cache['Z2'])
    dZ2_dw2 = cache['X']
    dC_dw2 = dC_da3 * da3_dZ3 * dZ3_da2 * da2_dZ2 * dZ2_dw2

    return np.array([dC_dw1, dC_dw2, dC_dw3], dtype=object)

# Atualização
def update_param(weights, bp_results, lr):
    weights[0] -= lr * bp_results[0]
    weights[1] -= lr * bp_results[1]
    weights[2] -= lr * bp_results[2]
    return weights

# Treinamento
def model(dataset, epochs, lr):
    weights, bias = init_params()
    for epoch in range(epochs):
        loss = 0
        for x, yt in dataset:
            x = np.insert(x, 0, 1)
            y_pred, cache = forward(weights, bias, x)
            loss += mse(yt, y_pred)
            bp_results = backprop(weights, cache, yt)
            weights = update_param(weights, bp_results, lr)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} - Loss: {loss:.4f}")
    return weights, bias

# Dataset XOR
dataset = [
    (np.array([0, 0]), 0),
    (np.array([0, 1]), 1),
    (np.array([1, 0]), 1),
    (np.array([1, 1]), 0),
]

weights, bias = model(dataset, epochs=20000, lr=0.2)

print("\n--- RESULTADO FINAL ---")
for x, yt in dataset:
    x_input = np.insert(x, 0, 1)
    yp, _ = forward(weights, bias, x_input)
    print(f"Input: {x}, Esperado: {yt}, Predito: {yp:.4f}")
    
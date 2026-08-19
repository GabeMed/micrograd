from value import Value
import numpy as np
from network import NN


def loss(X, y, batch_size=None, model=NN(2, [16, 16, 1])):

    # Por default, usamos todos os dados na perda, mas podemos usar só um batch
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]

    # Forward pass para obter as previsões
    scores = list(map(model, inputs))

    # Perda hinge de SVM
    losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))

    # Termo de regularização para a perda
    alpha = 1e-4
    reg_loss = alpha * sum((p * p for p in model.parameters()))
    total_loss = data_loss + reg_loss

    # Retorne a acurácia para obter métricas de diagnósticos
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)


total_loss, acc = loss()
print(total_loss, acc)

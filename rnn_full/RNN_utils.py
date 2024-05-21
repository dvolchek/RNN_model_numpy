import numpy as np


class Softmax:
    """
    Класс для реализации функции Softmax
    :math:`Softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}e^{x_j}}`
    """
    def __init__(self):
        self.type = 'Softmax'

    def forward(self, Z):
        """
        Непосредственно Softmax
        Parameters
        ----------
        Z : numpy.array
            Вход
        Returns
        -------
        A : numpy.array
            Выход
        """
        self.Z = Z

        t = np.exp(Z - np.max(Z, axis=0))
        self.A =  t / np.sum(t, axis=0, keepdims=True)

        return self.A


class Tanh:
    """
    Класс для реализации гиперболического тангенса
    :math:`Tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}`  
    """
    def __init__(self):
        self.type = 'Tanh'

    def forward(self, Z):
        """
        Непосредственно Tanh
        Parameters
        ----------
        Z : numpy.array
            Вход
        Returns
        -------
        A : numpy.array
            Выход
        """
        self.A = np.tanh(Z)

        return self.A

    def backward(self, dA):
        """
        Вычисления для обратного распространения
        Parameters
        ----------
        dA : numpy.array
            Градиенты выходов
        Returns
        -------
        dZ : numpy.array
            Градиенты входов
        """
        dZ = dA * (1 - np.power(self.A, 2))

        return dZ


class CrossEntropyLoss:
    """
    Класс для реализации вычисления кросс-энтропии
    """
    def __init__(self):
        self.type = 'CELoss'
    
    def forward(self, Y_hat, Y):
        """
        Непосредственное вычисление лосса
        Parameters
        ----------
        Y_hat : numpy.array
            Предсказания
        Y : numpy.array
            Истинные отклики
        
        Returns
        -------
        Numpy.array -- значение лосса
        """
        self.Y = Y
        self.Y_hat = Y_hat

        _loss = - Y * np.log(self.Y_hat)
        loss = np.sum(_loss, axis=0).mean()

        return np.squeeze(loss) 

    def backward(self):
        """
        Вычисления для обратного распространения
        Returns
        -------
        grad : numpy.array
            Градиент лосса
        """
        grad = self.Y_hat - self.Y
        
        return grad


class SGD:
    """
    Стохастический градиентный спуск (momentum)
    """
    def __init__(self, lr=0.0075, beta=0.9):
        """
        Parameters
        ----------
        lr : int, default: 0.0075
            Скорость обучения
        beta : int, default: 0.9
            Параметр бетта
        """
        self.beta = beta
        self.lr = lr

    def optim(self, weights, gradients, velocities=None):
        """
        Parameters
        ---------
        weights : numpy.array
            Веса слоя
        bias : numpy.array
            Смещения
        dW : numpy.array
            Градиенты весов
        db : numpy.array
            Градиенты смещений
        velocities : tuple
            Tuple с velocities для SGD c momentum.
        Returns
        -------
        weights : numpy.array
            Обновленные веса
        bias : numpy.array
            Обновленные смещения
        (V_dW, V_db) : tuple
            Tuple из int'ов, содержащий velocities для весов
            и смещений
        """
        if velocities is None: velocities = [0 for weight in weights]

        velocities = self._update_velocities(
            gradients, self.beta, velocities
        )
        new_weights = []

        for weight, velocity in zip(weights, velocities):
            weight -= self.lr * velocity
            new_weights.append(weight)

        return new_weights, velocities

    def _update_velocities(self, gradients, beta, velocities):
        """
        Обновляет velocities производных для 
        весов и смещений
        """
        new_velocities = []

        for gradient, velocity in zip(gradients, velocities):

            new_velocity = beta * velocity + (1 - beta) * gradient
            new_velocities.append(new_velocity)

        return new_velocities


def one_hot_encoding(input, size):
    """
    One hot encoding
    
    Parameters
    ----------
    input : list
        Список, который нужно закодировать
    size : int
        Размер
        
    Returns
    -------
    output : list
        Результат one hot
    """
    output = []

    for index, num in enumerate(input):
        one_hot = np.zeros((size,1))

        if (num != None):
            one_hot[num] = 1
    
        output.append(one_hot)

    return output
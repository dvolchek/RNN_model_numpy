import numpy as np
from RNN_utils import Tanh, Softmax, CrossEntropyLoss


class RNNModel:
    """
    Модель рекуррентной нейронной сети
    """
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Инициализация параметров ячейки RNN 

        Parameters
        ----------
        input_dim : int
            Размерность входа 
        output_dim : int
            Размерность выхода
        hidden_dim : int
            Размерность скрытого состояния
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        params = self._initialize_parameters(
                input_dim, output_dim, hidden_dim
        )
        self.Wya, self.Wax, self.Waa, self.by, self.b = params
        self.softmax = Softmax()
        self.oparams = None


    # def forward(self, input_X):
    #     """
    #     Прямой проход по сети

    #     Parameters
    #     ----------
    #     input_X : numpy.array or list
    #         Массив входных данных (батч)

    #     Returns
    #     -------
    #     y_preds : list
    #         Массив предсказаний, которые выдает сеть
    #         для каждого входного данного
    #     """
    #     self.input_X = input_X

    #     self.layers_tanh = [Tanh() for x in input_X]
    #     hidden = np.zeros((self.hidden_dim , 1))
        
    #     self.hidden_list = [hidden]
    #     self.y_preds = []

    #     for input_x, layer_tanh in zip(input_X, self.layers_tanh):
    #         input_tanh = np.dot(self.Wax, input_x) + np.dot(self.Waa, hidden) + self.b
    #         hidden = layer_tanh.forward(input_tanh)
    #         self.hidden_list.append(hidden)

    #         input_softmax = np.dot(self.Wya, hidden) + self.by
    #         y_pred = self.softmax.forward(input_softmax)
    #         self.y_preds.append(y_pred)

    #     return self.y_preds

    def forward(self):
        """
        Прямой проход по сети

        Parameters
        ----------
        input_X : numpy.array or list
            Массив входных данных (батч)

        Returns
        -------
        y_preds : list
            Массив предсказаний, которые выдает сеть
            для каждого входного данного
        """
        print('Hello')  


    def loss(self, Y):
        """
        Вычисление лосса (Кросс энтропия)

        Parameters
        ----------
        Y : numpy.array or list
            Массив истинных откликов

        Returns
        -------
        cost : int
            Значение лосса
        """
        self.Y = Y
        self.layers_loss = [CrossEntropyLoss() for y in self.Y]
        cost = 0
        
        for y_pred, y, layer in zip(self.y_preds, self.Y, self.layers_loss):
            cost += layer.forward(y_pred, y)
        
        return cost
    

    def backward(self):  
        """
        Обратное распространение

        Вычисление градиентов, обновление весов
        """
        gradients = self._define_gradients()
        self.dWax, self.dWaa, self.dWya, self.db, self.dby, dhidden_next = gradients

        for index, layer_loss in reversed(list(enumerate(self.layers_loss))):
            dy = layer_loss.backward()

            # hidden actual
            hidden = self.hidden_list[index + 1]
            hidden_prev = self.hidden_list[index]

            # gradients y
            self.dWya += np.dot(dy, hidden.T)
            self.dby += dy
            dhidden = np.dot(self.Wya.T, dy) + dhidden_next
    
            # gradients a
            dtanh = self.layers_tanh[index].backward(dhidden)
            self.db += dtanh
            self.dWax += np.dot(dtanh, self.input_X[index].T)
            self.dWaa += np.dot(dtanh, hidden_prev.T)
            dhidden_next = np.dot(self.Waa.T, dtanh)


    def clip(self, clip_value):
        """
        Клиппинг градиентов во избежание взрывов градиентов

        Parameters
        ----------
        clip_value : int
            Пороговое значение для отсечения
        """
        for gradient in [self.dWax, self.dWaa, self.dWya, self.db, self.dby]:
            np.clip(gradient, -clip_value, clip_value, out=gradient)


    def optimize(self, method):
        """
        Обновление параметров модели в соответствии с методом оптимизации

        Parameters
        ----------
        method: Class
            Метод оптимизации
        """
        weights = [self.Wya, self.Wax, self.Waa, self.by, self.b]
        gradients = [self.dWya, self.dWax, self.dWaa, self.dby, self.db]

        weights, self.oparams = method.optim(weights, gradients, self.oparams)
        self.Wya, self.Wax, self.Waa, self.by, self.b = weights
        
    
    def generate_names(
        self, index_to_character
    ):
        """
        Генерация имен

        Parameters
        ----------
        index_to_character : dict
            Словарь для получения символов на основе индексов

        Returns
        -------
        name : list
            Сгенерированное имя
        """
        letter = None
        indexes = list(index_to_character.keys())

        letter_x = np.zeros((self.input_dim, 1))
        name = []

        # Очень похоже на forward prop
        layer_tanh = Tanh()
        hidden = np.zeros((self.hidden_dim , 1))

        while letter != '\n' and len(name) < 15:

            input_tanh = np.dot(self.Wax, letter_x) + np.dot(self.Waa, hidden) + self.b
            hidden = layer_tanh.forward(input_tanh)

            input_softmax = np.dot(self.Wya, hidden) + self.by
            y_pred = self.softmax.forward(input_softmax)

            index = np.random.choice(indexes, p=y_pred.ravel())
            letter = index_to_character[index]

            name.append(letter)

            letter_x = np.zeros((self.input_dim, 1))
            letter_x[index] = 1

        return "".join(name)


    def _initialize_parameters(self, input_dim, output_dim, hidden_dim):
        """
        Инициализация параметров

        Parameters
        ----------
        input_dim : int
            Размерность входа
        output_dim : int
            Разменость выхода
        hidden_dim : int

        Returns
        -------
        weights_y : numpy.array
        weights_ax : numpy.array
        weights_aa : numpy.array
        bias_y : numpy.array
        bias : numpy.array
        """
        den = np.sqrt(hidden_dim)

        weights_y = np.random.randn(output_dim, hidden_dim) / den
        bias_y = np.zeros((output_dim, 1))

        weights_ax = np.random.randn(hidden_dim, input_dim) / den
        weights_aa = np.random.randn(hidden_dim, hidden_dim) / den
        bias = np.zeros((hidden_dim, 1))

        return weights_y, weights_ax, weights_aa, bias_y, bias


    def _define_gradients(self):
        """
        Инициализация градиентов
        """
        dWax = np.zeros_like(self.Wax)
        dWaa = np.zeros_like(self.Waa)
        dWya = np.zeros_like(self.Wya)

        db = np.zeros_like(self.b)
        dby = np.zeros_like(self.by)

        da_next = np.zeros_like(self.hidden_list[0])

        return dWax, dWaa, dWya, db, dby, da_next
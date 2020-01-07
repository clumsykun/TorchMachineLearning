import torch
from torch import nn


class LinearRegression():
    def __init__(self,
                 input_size,
                 output_size,
                 criterion='MSE',
                 learning_rate=0.001):
        super(LinearRegression, self).__init__()

        self.model = nn.Linear(input_size, output_size)

        if criterion == 'MSE':
            self.criterion = nn.MSELoss()
        elif criterion == 'L1':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError('不支持的 loss 类型')

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def fit(self, X, y, num_epochs=50):

        for epoch in range(num_epochs):

            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch+1) % 1000 == 0:
                print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

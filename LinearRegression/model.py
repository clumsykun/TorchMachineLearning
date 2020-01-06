import torch
from torch import nn


class LinearRegression():
    def __init__(self, input_size, output_size, learning_rate=0.001):
        super(LinearRegression, self).__init__()

        self.model = nn.Linear(input_size, output_size)
        self.criterion = nn.MSELoss()
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

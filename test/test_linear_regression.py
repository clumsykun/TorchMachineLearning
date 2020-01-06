import numpy as np
from LinearRegression.model import LinearRegression
from torch import Tensor

def test_LinearRegression():
    x = Tensor( np.random.rand(128, 20) * 10 )
    w = Tensor( np.random.rand(20, 1) * 10)
    y = x.matmul(w)
    y = y + Tensor(np.random.rand(128, 1))  # 偏差

    lr = LinearRegression(20, 1)
    lr.fit(x, y, num_epochs=50000)

    state_dict = lr.model.state_dict()
    print(state_dict['weight'])
    print(w.view(-1))

    print(1)

if __name__ == "__main__":
    pass

import numpy as np
from LinearRegression.model import LinearRegression
from torch import Tensor

def test_LinearRegression():
    num_sample = 12800
    in_features = 20
    out_features = 1
    x = Tensor( np.random.rand(num_sample, in_features) * 10 )
    w = Tensor( np.random.rand(in_features, out_features) * 10)
    y = x.matmul(w)
    y = y + Tensor(np.random.rand(num_sample, out_features))  # 偏差

    lr = LinearRegression(in_features, out_features, criterion='L1')
    lr.fit(x, y, num_epochs=100000)

    state_dict = lr.model.state_dict()
    print(state_dict['weight'].view(-1)[:10])
    print(w.view(-1)[:10])

    print(1)


if __name__ == "__main__":
    pass

#単層ニューラルネットワークによる予測をするやつ
from torch import nn

class SLPNet(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.fc = nn.Linear(input_size, output_size, bias=False)
    nn.init.normal_(self.fc.weight, 0.0, 1.0)
  def forward(self, x):
    x = self.fc(x)
    return x

model = SLPNet(300, 4)
y_hat_1 = torch.softmax(model(X_train[:1]), dim=-1)

Y_hat = torch.softmax(model.forward(X_train[:4]), dim=-1)
print(y_hat_1)
print(Y_hat)
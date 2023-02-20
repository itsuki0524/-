#損失と勾配の計算をするやつ

criterion = nn.CrossEntropyLoss()

l_1 = criterion(model(X_train[:1]), y_train[:1])
model.zero_grad()
l_1.backward()
print(f'損失: {l_1:.4f}')
print(f'勾配:\n{model.fc.weight.grad}')

l = criterion(model(X_train[:4]), y_train[:4])
model.zero_grad()
l.backward()
print(f'損失: {l:.4f}')
print(f'勾配:\n{model.fc.weight.grad}')
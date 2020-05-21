import torch
import numpy as np

# PyTorchでデータや数値を扱うためのtorch.TensorはNumpyと似てる。

x = torch.empty(2, 3)
print(x)
x = torch.rand(2, 3)
print(x)
x = torch.zeros(2, 3, dtype=torch.float)
print(x)
x = torch.ones(2, 3, dtype=torch.float)
print(x)
x = torch.tensor([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]])
print(x)
print(x.dtype)

y = x.new_ones(2, 3)
print(y)
print(y.dtype)
y = torch.ones_like(x, dtype=torch.int)
print(y)

'''
　PyTorchのテンソルでは、メソッド名の最後にアンダースコア（_）がある場合（例えばadd_()）
「テンソル内部の置換（in-place changes）が起こること」を意味する。
一方、アンダースコア（_）がない通常の計算の場合（例えばadd()）は、計算元のテンソル内部は変更されずに、
戻り値として「新たなテンソル」が取得できる。
'''
print(x + y)
torch.add(x, y)
torch.add(x, y, out=x)
x.add_(y)

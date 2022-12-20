# 数据操作
# 简单介绍如何使用torch去做基本操作

# 首先，我们导入 torch。请注意，虽然它被称为PyTorch，但我们应该导入 torch 而不是 pytorch
import torch

# 张量表示由一个数值组成的数组，这个数组可能有多个维度
x = torch.arange(12)
var = x.shape  # 张量的形状
print(x.numel())  # 张量中元素的总数

# 要改变一个张量的形状而不改变元素数量和元素值，可以调用 reshape 函数
X = x.reshape(3, 4)
var2 = X.shape
print(X.numel())

# 使用全0、全1、其他常量或者从特定分布中随机采样的数字
x1 = torch.zeros((2, 3, 4))  # 2层3行4列
x2 = torch.ones((2, 3, 4))
x3 = torch.randn(3, 4)

# 通过提供包含数值的 Python 列表（或嵌套列表）来为所需张量中的每个元素赋予确定值
x4 = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# 常见的标准算术运算符（+、-、*、/ 和 **）都可以被升级为按元素运算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x5 = x + y
x6 = x - y
x7 = x * y
x8 = x / y
x9 = x ** y

# 按元素方式应用更多的计算
x10 = torch.exp(x)

# 我们也可以把多个张量 连结（concatenate） 在一起
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
x11 = torch.cat((X, Y), dim=0)  # 续接
x12 = torch.cat((X, Y), dim=1)  # 同向量接

# 通过 逻辑运算符 构建二元张量
X == Y

# 对张量中的所有元素进行求和会产生一个只有一个元素的张量
x13 = X.sum()

# 即使形状不同，我们仍然可以通过调用 广播机制 （broadcasting mechanism） 来执行按元素操作
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
x14 = a + b

# 可以用 [-1] 选择最后一个元素，可以用 [1:3] 选择第二个和第三个元素
X[-1], X[1:3]

# 除读取外，我们还可以通过指定索引来将元素写入矩阵
X[1, 2] = 9

# 为多个元素赋值相同的值，我们只需要索引所有元素，然后为它们赋值
X[0:2, :] = 12
X

# 运行一些操作可能会导致为新结果分配内存
before = id(Y)
Y = Y + X
id(Y) == before

# 执行原地操作
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

# 如果在后续计算中没有重复使用 X，我们也可以使用 X[:] = X + Y 或 X += Y 来减少操作的内存开销
before = id(X)
X += Y
id(X) == before

# 转换为 NumPy 张量
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)

# 将大小为1的张量转换为 Python 标量
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)

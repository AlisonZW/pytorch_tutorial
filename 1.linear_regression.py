import torch
import matplotlib.pyplot as plt
import numpy as np 


# arr = np.ones((3,4))
#
# t = torch.tensor(arr,device="cuda:0")
#
# print(t)

t = torch.ones((3, 9))

#x = torch.cat([t, t], dim=1)

#y = torch.stack([t, t], dim=2)

# list_of_tensor = torch.chunk(t, chunks=3, dim=0)
# for idx, t in enumerate(list_of_tensor):
#     print("{}: {}".format(idx, t))
#print(y)

#

#x = torch.randperm(8)
# t = torch.rand((2,4))
# t_reshape = torch.reshape(t, (4,2))
# t_transpose = torch.transpose(t, dim0=1, dim1=0)
# print(t)
# print(t_reshape)
# print(t_transpose)

# t = torch.randn((3, 3))
# t1= torch.ones_like(t)
#
# t2= torch.add(t, 10, t1)
# print(t2)


torch.manual_seed(10)
x = torch.rand(30, 1) * 10
#print(x)

y = 3*x + 2 + torch.randn(30,1)

w = torch.randn((1), requires_grad=True)
b = torch.zeros((1), requires_grad=True)

lr = 0.01
for iteration in range(1000):
    wx = torch.mul(x, w)
    y_pred = torch.add(wx, b)

    loss = (0.5*(y_pred - y)**2).mean()
    loss.backward()

    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    w.grad.zero_()
    b.grad.zero_()

    if iteration % 20 == 0:

        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)

        if loss.data.numpy() < 0.3:
            print("w:{}, b:{}".format(w, b))
            break
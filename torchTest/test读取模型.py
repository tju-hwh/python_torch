import torch
import matplotlib.pyplot as plt

def GetNet():
    net = torch.load('net1.pkl')
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    zzz = net(x)
    plt.scatter(x.data.numpy(), zzz.data.numpy())
    plt.plot(x.data.numpy(), zzz.data.numpy())

    plt.show()

if __name__ == '__main__':
    GetNet()
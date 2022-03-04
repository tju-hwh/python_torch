import torch
import matplotlib.pyplot as plt


def Create():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    optimisizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()
    return net1,optimisizer,loss_func

def Train(x,y,net,optimisizer,loss_func):
    for i in range(100):
        prediction = net(x)
        loss = loss_func(prediction,y)
        optimisizer.zero_grad()
        loss.backward()
        optimisizer.step()
    return net

if __name__ == '__main__':
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    net1,optimisizer,loss_func = Create()
    net1 = Train(x,y,net1,optimisizer,loss_func)
    predictions = net1(x)

    torch.save(net1,'net1.pkl')

    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),predictions.data.numpy())

    plt.show()



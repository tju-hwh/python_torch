import torch
import torch.utils.data as Data
# 超参数
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

class Net(torch.nn.Sequential):
    torch.nn.Linear(1,20),
    torch.nn.ReLU,
    torch.nn.Linear(20,1)


net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
nets = [net_SGD,net_Momentum,net_RMSprop,net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(),lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()

#记录误差
losses_his = [[], [], [], []]   # 记录 training 时不同神经网络的 loss

for epoch in range(EPOCH):
    print("Epoch :",epoch,"\n")
    for step,(batch_x,batch_y) in enumerate(loader):
        #开始训练
        for net,opt,los in zip(nets,optimizers,losses_his):
            output = net(batch_x)
            loss = loss_func(output,y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            los.append(loss.data[0])

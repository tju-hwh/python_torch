import torch
import torch.utils.data as Data

if __name__ == '__main__':

    BATCH_SIZE = 8
    x = torch.linspace(1,10,10)
    y = torch.linspace(10,1,10)

    #tensor形式的数据库
    torch_dataset = Data.TensorDataset(x,y)
    #loader 让训练变成一小批一小批的
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,#定义训练时是否要随机打乱
        num_workers=2,  #windows的话去掉,直接放在main方法里不用去掉。 但2worker比1worker慢
    )

    #epoch 把整体10个数据训练三次
    for epoch in range(3):
        for step ,(batch_x,batch_y) in enumerate(loader):   #step 是enumerate的索引
            #训练
            print("epoch:",epoch,"  step:",step,"  batch_x",batch_x.numpy(),"  batch_y",batch_y.numpy())


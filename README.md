# Code-Annotation
## 代码注释
```python
import ...

# 初始化参数
def get_args():...

class Worker:
    def __init__(self, args):
        self.opt = args

        # 判定设备
        self.device = torch.device('cuda:0' if args.is_cuda else 'cpu')
        #将模型和数据移到GPU或CPU上进行训练和推理。如果args.is_cuda为True，则将设备设置为cuda:0，否则设置为cpu。
        kwargs = {
            'num_workers': args.num_workers,
            #表示设置训练过程中的数据加载器使用的进程数量。其中，args.num_workers 是从命令行参数中获取的。它的值决定了数据加载器在读取数据时使用的并行进程数量。这个参数可以有效地加速数据加载，并提高模型训练的效率。但是，进程数量过高可能会导致系统资源占用过多，而影响其他进程的运行。
            'pin_memory': True,
        } if args.is_cuda else {}

        # 载入数据
        train_dataset = datasets.ImageFolder(
            args.train_dir,
            #指定了训练数据的目录，args.train_dir是一个参数，代表了训练数据的目录路径，该路径会被传递给训练代码中的相关函数，使得数据可以被正确地读取和使用
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),#对图片进行随机大小裁剪，并将其调整为256x256的大小。
                transforms.ToTensor()#将PIL图像或numpy.ndarray转换为张量（Tensor）类型。transforms.ToTensor()将图像像素的值从0-255转换为0-1的范围内的浮点数，并将其存储为张量。
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
        val_dataset = datasets.ImageFolder(
           ...
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,#将定义数据集用于模型训练。train_dataset是一个变量，它包含了我们的训练数据。在这里，我们使用dataset=train_dataset来指定我们要使用的数据集。这个操作将数据集传递给模型训练器，这样它就可以使用数据来训练我们的模型。
            batch_size=args.batch_size,#batch_size是指每一次模型训练时，输入的数据分成的小块的大小。这个值决定了一次训练中跑多少个样本。
            shuffle=True,#shuffle=True在模型训练中的作用是使每个epoch中的训练数据顺序随机化，从而增加训练的随机性和稳定性。这样可以防止模型在顺序训练过程中出现输入相关的过拟合现象。
            **kwargs
        )
        self.val_loader = DataLoader(
            ...
        )

        # 挑选神经网络、参数初始化
        ...

        self.model = net.to(self.device)#将模型(net)移动到指定的设备(device)上进行训练

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),#self.model.parameters() 是一个函数，它返回可训练参数的生成器。在模型训练中，它通常用于传递给优化器(optimizer)的参数，以便调整模型参数以最小化损失函数(loss function)。
            lr=args.lr#lr=args.lr 是将命令行参数中传入的学习率赋值给 lr 变量，其中 args.lr 是命令行参数中指定的学习率。这个代码会在模型训练时使用指定的学习率，以控制模型参数的调整速度。
        )

        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()#定义模型训练过程中的损失函数，即交叉熵损失函数。它的作用是计算模型预测结果与目标值之间的差异，并根据这个差异来反向传播更新模型参数。交叉熵损失函数适用于多分类任务，因为它能够将模型预测的概率分布与真实概率分布之间的差异最小化。

        # warm up 学习率调整部分
        ...

    def train(...):...

    def val(...):...


if __name__ == '__main__':
    # 初始化
    ...
    worker = Worker(args=args)#在模型训练中，worker = Worker(args=args)的作用是创建一个Worker对象，以便在本地或远程计算机上进行多进程训练。
    # Worker对象是 PyTorch 的 DistributedDataParallel 组件的一部分，它利用多进程并行计算来加速模型训练过程，并帮助在多个GPU、多个计算机上训练模型。

    # 训练与验证
    for epoch in range(1, args.epochs + 1):...

```
## 在终端启动训练
输入以下命令
```python
python train.py 
```
训练结果

![image](https://user-images.githubusercontent.com/128216499/233022052-cf8513d9-8dee-404e-9d1a-4715d982a9c3.png)


## 更改参数设置并启动训练（有启动训练的命令即可）
更改为以下参数

![image](https://user-images.githubusercontent.com/128216499/233016745-2f3f9ccd-efb1-4961-8129-41a55b9d2bbf.png)

输入以下命令
```python
python train.py --epochs 10 --save_station 1 --model ResNet34
```
参数更改后的训练结果

![image](https://user-images.githubusercontent.com/128216499/233022951-2c348367-d878-4f29-9ae3-cf4de779ce82.png)

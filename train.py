import argparse
import time
import json
import os

from tqdm import tqdm
from models import *
# from efficientnet_pytorch import EfficientNet
from torch import nn
from torch import optim
# from torch.optim.lr_scheduler import *
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tools import warmup_lr


# 初始化参数
def get_args():
    """在下面初始化你的参数.
    """
    parser = argparse.ArgumentParser(description='基于Pytorch实现的分类任务')

    # exp
    parser.add_argument('--time_exp_start', type=str,
                        default=time.strftime('%m-%d-%H-%M', time.localtime(time.time())))
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--val_dir', type=str, default='data/val')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--save_station', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--is_mps', type=bool, default=False)
    parser.add_argument('--is_cuda', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)

    # dataset
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--data_mean', type=tuple, default=[.5, .5, .5])
    parser.add_argument('--data_std', type=tuple, default=[.5, .5, .5])

    # model
    parser.add_argument('--model', type=str, default='ResNet18',
                        choices=[
                            'ResNet18',
                            'ResNet34',
                            'ResNet50',
                            'ResNet18RandomEncoder',
                        ])

    # scheduler
    parser.add_argument('--warmup_epoch', type=int, default=1)

    # 通过json记录参数配置
    args = parser.parse_args()
    args.directory = 'dictionary/%s/Hi%s/' % (args.model, args.time_exp_start)
    log_file = os.path.join(args.directory, 'log.json')
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    with open(log_file, 'w') as log:
        json.dump(vars(args), log, indent=4)

    # 返回参数集
    return args


class Worker:
    def __init__(self, args):
        self.opt = args

        # 判定设备
        self.device = torch.device('cuda:0' if args.is_cuda else 'cpu')#将模型和数据移到GPU或CPU上进行训练和推理。如果args.is_cuda为True，则将设备设置为cuda:0，否则设置为cpu。
        kwargs = {
            'num_workers': args.num_workers,#表示设置训练过程中的数据加载器使用的进程数量。其中，args.num_workers 是从命令行参数中获取的。它的值决定了数据加载器在读取数据时使用的并行进程数量。
#这个参数可以有效地加速数据加载，并提高模型训练的效率。但是，进程数量过高可能会导致系统资源占用过多，而影响其他进程的运行。
            'pin_memory': True,
        } if args.is_cuda else {}

        # 载入数据
        train_dataset = datasets.ImageFolder(
            args.train_dir,#指定了训练数据的目录，args.train_dir是一个参数，代表了训练数据的目录路径，该路径会被传递给训练代码中的相关函数，使得数据可以被正确地读取和使用
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),#对图片进行随机大小裁剪，并将其调整为256x256的大小。
                transforms.ToTensor()#将PIL图像或numpy.ndarray转换为张量（Tensor）类型。transforms.ToTensor()将图像像素的值从0-255转换为0-1的范围内的浮点数，并将其存储为张量。
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
        val_dataset = datasets.ImageFolder(
            args.val_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,#将定义数据集用于模型训练。train_dataset是一个变量，它包含了我们的训练数据。在这里，我们使用dataset=train_dataset来指定我们要使用的数据集。这个操作将数据集传递给模型训练器，这样它就可以使用数据来训练我们的模型。
            batch_size=args.batch_size,#batch_size是指每一次模型训练时，输入的数据分成的小块的大小。这个值决定了一次训练中跑多少个样本。
            shuffle=True,#shuffle=True在模型训练中的作用是使每个epoch中的训练数据顺序随机化，从而增加训练的随机性和稳定性。这样可以防止模型在顺序训练过程中出现输入相关的过拟合现象。
            **kwargs
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        # 挑选神经网络、参数初始化
        net = None
        if args.model == 'ResNet18':
            net = ResNet18(num_cls=args.num_classes)
        elif args.model == 'ResNet34':
            net = ResNet34(num_cls=args.num_classes)
        elif args.model == 'ResNet50':
            net = ResNet50(num_cls=args.num_classes)
        elif args.model == 'ResNet18RandomEncoder':
            net = ResNet18RandomEncoder(num_cls=args.num_classes)
        assert net is not None

        self.model = net.to(self.device)#将模型(net)移动到指定的设备(device)上进行训练

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),#self.model.parameters() 是一个函数，它返回可训练参数的生成器。在模型训练中，它通常用于传递给优化器(optimizer)的参数，以便调整模型参数以最小化损失函数(loss function)。
            lr=args.lr#lr=args.lr 是将命令行参数中传入的学习率赋值给 lr 变量，其中 args.lr 是命令行参数中指定的学习率。这个代码会在模型训练时使用指定的学习率，以控制模型参数的调整速度。
        )

        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()#定义模型训练过程中的损失函数，即交叉熵损失函数。它的作用是计算模型预测结果与目标值之间的差异，并根据这个差异来反向传播更新模型参数。交叉熵损失函数适用于多分类任务，因为它能够将模型预测的概率分布与真实概率分布之间的差异最小化。

        # warm up 学习率调整部分
        self.per_epoch_size = len(train_dataset) // args.batch_size
        self.warmup_step = args.warmup_epoch * self.per_epoch_size
        self.max_iter = args.epochs * self.per_epoch_size
        self.global_step = 0

    def train(self, epoch):
        self.model.train()
        bar = tqdm(enumerate(self.train_loader))
        for batch_idx, (data, target) in bar:
            self.global_step += 1
            data, target = data.to(self.device), target.to(self.device)

            # 训练中...
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()
            lr = warmup_lr.adjust_learning_rate_cosine(
                self.optimizer, global_step=self.global_step,
                learning_rate_base=self.opt.lr,
                total_steps=self.max_iter,
                warmup_steps=self.warmup_step
            )

            # 更新进度条
            bar.set_description(
                'train epoch {} >> [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {:.6f} >> '.format(
                    epoch,
                    batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item(),
                    lr
                )
            )
        bar.close()

    def val(self):
        self.model.eval()
        validating_loss = 0
        num_correct = 0
        with torch.no_grad():
            bar = tqdm(self.val_loader)
            for data, target in bar:
                # 测试中...
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                validating_loss += self.loss_function(output, target).item()  # 累加 batch loss
                pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率神经元下标
                num_correct += pred.eq(target.view_as(pred)).sum().item()
            bar.close()

        # 打印验证结果
        validating_loss /= len(self.val_loader)
        print('val >> Average loss: {:.4f}, Accuracy: {}/{} ({:.03f}%)\n'.format(
            validating_loss,
            num_correct,
            len(self.val_loader.dataset),
            100. * num_correct / len(self.val_loader.dataset))
        )

        # 返回重要信息，用于生成模型保存命名
        return 100. * num_correct / len(self.val_loader.dataset), validating_loss


if __name__ == '__main__':
    # 初始化
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(0)
    args = get_args()
    worker = Worker(args=args)#在模型训练中，worker = Worker(args=args)的作用是创建一个Worker对象，以便在本地或远程计算机上进行多进程训练。
    # Worker对象是 PyTorch 的 DistributedDataParallel 组件的一部分，它利用多进程并行计算来加速模型训练过程，并帮助在多个GPU、多个计算机上训练模型。

    # 训练与验证
    for epoch in range(1, args.epochs + 1):
        worker.train(epoch)
        val_acc, val_loss = worker.val()
        if epoch > args.save_station:
            save_dir = args.directory + '%s-epochs-%d-model-val-acc-%.3f-loss-%.6f.pt' \
                       % (args.model, epoch, val_acc, val_loss)
            torch.save(worker.model, save_dir)

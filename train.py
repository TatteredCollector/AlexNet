import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import torch.optim as optim
from tqdm import tqdm
from model.AlexNet import AlexNet

"""
训练代码大致顺序
1.判断使用GPU还是CPU
2.数据预处理
3.数据加载 
4.加载模型，设置使用设备
5.损失函数和优化器定义
6.迭代训练 （双循环）
"""


def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))
    # torchvision.transforms: 常用的图片变换归一化处理
    # torchvision.transforms.Compose()类。这个类的主要作用是串联多个图片变换的操作。
    # Compose里面的参数实际上就是个列表，而这个列表里面的元素就是你想要执行的transform操作。
    data_transform = {
        "train": transforms.Compose([
            # 是指将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；
            transforms.RandomResizedCrop(224),
            # 以给定的概率随机水平旋转给定的图像，默认为0.5；
            transforms.RandomHorizontalFlip(),
            # to_tensor()函数源码看出其接受PIL Image或numpy.ndarray格式，功能如下：
            # 先由HWC转置为CHW格式；
            # 再转为float类型；
            # 最后，每个像素除以255 数据范围在[0,1]
            transforms.ToTensor(),

            # 过normalize()的变换后变成了均值为0 方差为1（其实就是最大最小值为1和-1）
            # 每个样本图像变成了均值为0  方差为1 的标准正态分布，这就是最普通（科学研究价值最大的）的样本数据了
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        ]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                   ])
    }
    data_root = os.getcwd()
    image_path = os.path.join(data_root, 'data')
    assert os.path.exists(image_path), "{} path not exist.".format(image_path)
    # ImageFolder是一个通用的数据加载器，它要求我们以下面这种格式来组织数据集的训练、验证或者测试图片。
    # root/cat/123.png
    # root/dog/xxx.png
    train_data = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                      transform=data_transform['train'])
    # root：图片存储的根目录，即各类别文件夹所在目录的上一级目录。
    # transform：对图片进行预处理的操作（函数），原始图片作为输入，返回一个转换后的图片。
    # target_transform：对图片类别进行预处理的操作，输入为 target，输出对其的转换。如果不传该参数，即对 target 不做任何转换，返回的顺序索引 0,1, 2…
    # loader：表示数据集加载方式，通常默认加载方式即可。
    # is_valid_file：获取图像文件的路径并检查该文件是否为有效文件的函数(用于检查损坏文件)
    # train_data 返回值有三个
    # self.classes：用一个 list 保存类别名称
    # self.class_to_idx：key=类别 value=对应索引的字典，与不做任何转换返回的 target 对应
    # self.imgs：保存(img-path, class) tuple的 list
    train_num = len(train_data)

    flower_list = train_data.class_to_idx
    # key=类别，val=索引值--> key=索引，val = 类别
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # 这里写入json文件是为了方便后面检测时读入类别字典
    # json.dumps()是把python对象转换成json对象的一个过程，生成的是字符串,
    # indent=int 数根据数据格式缩进显示，读起来更加清晰。
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as f:
        f.write(json_str)

    batch_size = 32
    # 设置多个进程来加载数据集
    # 一般开始是将num_workers设置为等于计算机上的CPU数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("using {} dataloader workers every process".format(nw))
    # Dataloader将Dataset或其子类封装成一个迭代器
    # 这个迭代器可以迭代输出Dataset的内容
    # 同时可以实现多进程、shuffle、不同采样策略，数据校对等等处理过程
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                                            transform=data_transform['val'])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net = AlexNet(num_classes=5, init_weight=True)

    # 模型加载 设置设备
    net.to(device)
    #
    # 先对每个训练样本求损失，而后再求平均损失
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 20
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        net.train()
        run_loss = 0.0

        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):

            images, labels = data
            print(labels)
            # 梯度清零
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            # 梯度回传 更新
            loss.backward()
            optimizer.step()

            run_loss += loss.item()

            train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                      epochs,
                                                                      loss)
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_label = val_data
                outputs = net(val_images.to(device))
                # dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
                predict_y = torch.max(outputs, dim=1)[1]
                # torch.max函数会返回两个tensor，第一个tensor是每行的最大值；
                # 第二个tensor是每行最大值的索引。

                # 例如：A = [1,2,3,4]
                #      B = [1,1,2,2]
                # 那么torch.eq(A,B)得到的结果就是[1,0,0,0]
                # torch.eq().sum()就是将所有值相加，但是得到的仍然是一个tensor,
                # 本例中torch.eq(A,B).sum()得到的结果就是[1](1+0+0+0),
                # 最后一步torch.eq(A,B).sum().item()得到的就是这个tensor中的值了，即1。
                acc += torch.eq(predict_y, val_label.to(device)).sum().item()
        val_accurate = acc / val_num
        print("[epoch {}] train_loss:{:.3f}  val_accuracy: {:.3f}.".format(epoch + 1,
                                                                           run_loss / train_steps,
                                                                           val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            # 仅保存学习到的参数
            torch.save(net.state_dict(), save_path)

    print("Finished !")


if __name__ == '__main__':
    train()

from torch import nn
from torchvision import models
import torch


class BaseConv(nn.Module):
    """
    Net类继承nn.Module，super(Net, self).__init__()
    就是对继承自父类nn.Module的属性进行初始化。
    而且是用nn.Module的初始化方法来初始化继承的属性。
    """

    def __init__(self, in_channels, out_channels, ker_size, stride, padding, bia=False):
        super(BaseConv, self).__init__()
        self.myconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ker_size, stride, padding, bias=bia),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )
        '''
        inplace = True ,会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址上覆盖
        '''

    def forward(self, x):
        x = self.myconv(x)

        return x


class AlexNet(nn.Module):
    """
    如果子类(Puple)继承父类(Person)不做初始化，那么会自动继承父类(Person)属性name。
    如果子类(Puple_Init)继承父类(Person)做了初始化，且不调用super初始化父类构造函数，那么子类(Puple_Init)不会自动继承父类的属性(name)。
    如果子类(Puple_super)继承父类(Person)做了初始化，且调用了super初始化了父类的构造函数，那么子类(Puple_Super)会继承父类的(name)属性。
    """

    def __init__(self, num_classes=1000, init_weight=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # N = (W − F + 2P )/S+1 卷积和池化计算公式 除法向下取整
            # 224*224*3-->27*27*64
            BaseConv(in_channels=3, out_channels=64, ker_size=11, stride=4, padding=2),
            # 27*27*64-->13*13*192
            BaseConv(in_channels=64, out_channels=192, ker_size=5, stride=1, padding=2),
            # 13*13*192-->13*13*384
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 13*13*384-->13*13*256
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 13 * 13 * 256 -->6 * 6 * 256
            BaseConv(in_channels=256, out_channels=256, ker_size=3, stride=1, padding=1)
        )
        self.classfier = nn.Sequential(
            # nn.Flatten(),按第一维度展开

            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=6 * 6 * 256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes))
        if init_weight:
            self._init_weight()

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        # x = x.reshape(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        # 数据形状调整 (x.size(0), -1)将tensor的结构转换为了(batchsize, channels*x*y)
        # print(x.shape)
        x = self.classfier(x)
        return x

    # 初始化 权重

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 从给定均值和标准差的正态分布N(mean, std)中生成值，
                # 填充输入的张量或变量
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


'''
每个python模块（python文件，也就是此处的 test.py 和 import_test.py）都包含内置的变量 __name__，
当该模块被直接执行的时候，__name__ 等于文件名（包含后缀 .py ）；
如果该模块 import 到其他模块中，则该模块的 __name__ 等于模块名称（不包含后缀.py）。
而 “__main__” 始终指当前执行模块的名称（包含后缀.py）。进而当模块被直接执行时，__name__ == 'main' 结果为真。
'''

if __name__ == '__main__':
    model = AlexNet()
    Input = torch.randn(8, 3, 224, 224)
    out = model(Input)
    print(model)
    print(out.shape)

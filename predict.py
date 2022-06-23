import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model.AlexNet import AlexNet


def predict():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                              std=(0.5, 0.5, 0.5))])

    # 加载图像
    img_path = "./flowers.jpg"
    assert os.path.exists(img_path), "file: '{}' not exist!".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # plt.imshow()函数负责对图像进行处理，并显示其格式，
    # 而plt.show()则是将plt.imshow()处理后的函数显示出来。
    # plt.show()

    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: {} not exist!".format(json_path)
    json_file = open(json_path)
    class_indict = json.load(json_file)

    model = AlexNet(num_classes=5, init_weight=False)

    weight_path = "./AlexNet.pth"
    assert os.path.exists(weight_path), "file: {} not exist!".format(weight_path)
    model.load_state_dict(torch.load(weight_path))

    model.to(device)
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        # 由于CrossEntroyLoss函数包含了softmax 推理阶段没有loss
        # 单独使用softmax
        pre = torch.softmax(output, dim=0)
        # torch.argmax(x,dim)会返回x 在dim维度上张量最大值的索引
        pre_class = torch.argmax(pre).numpy()

        context = "class: {}  prob: {:.3f}".format(class_indict[str(pre_class)],
                                                   pre[pre_class].numpy())
        plt.title(context)

        for i in range(len(pre)):
            print("class: {}  prob: {:.3f}".format(class_indict[str(i)],
                                                   pre[i].numpy()))
    plt.show()


if __name__ == "__main__":
    predict()

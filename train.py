import cv2
from fastai import *
from fastai.vision import *
from torchvision.models import resnet18, resnet34, resnet50


def load_image(base_name, image_size):
    r = load_image_channel("{}_red.png".format(base_name, id), image_size)
    g = load_image_channel("{}_green.png".format(base_name, id), image_size)
    b = load_image_channel("{}_blue.png".format(base_name, id), image_size)
    y = load_image_channel("{}_yellow.png".format(base_name, id), image_size)
    return np.stack([r, g, b, y], axis=2)


def load_image_channel(file_path, image_size):
    channel = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if channel.shape[0] != image_size:
        channel = cv2.resize(channel, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return channel


class ResNet(nn.Module):
    def __init__(self, type, num_classes):
        super().__init__()

        if type == "resnet18":
            self.resnet = resnet18(pretrained=True)
            num_fc_in_channels = 512
        elif type == "resnet34":
            self.resnet = resnet34(pretrained=True)
            num_fc_in_channels = 512
        elif type == "resnet50":
            self.resnet = resnet50(pretrained=True)
            num_fc_in_channels = 2048
        else:
            raise Exception("Unsupported resnet model type: '{}".format(type))

        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1.weight.data[:, 0:3, :, :] = self.resnet.conv1.weight.data
        self.resnet.conv1 = conv1

        self.resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.resnet.fc = nn.Sequential(
            nn.BatchNorm1d(num_fc_in_channels),
            nn.Dropout(0.5),
            nn.Linear(num_fc_in_channels, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)


class HpaImageItemList(ImageItemList):
    def open(self, fn):
        return pil2tensor(PIL.Image.fromarray(load_image(fn[:-4], 256)), np.float32) / 255.


data = (
    HpaImageItemList
        .from_csv('../../hpa', 'train.csv', folder='train', suffix='.png')
        .random_split_by_pct()
        .label_from_df(sep=' ')
        .databunch()
)

data.show_batch(rows=3)

# learner = create_cnn(data, lambda p: ResNet("resnet18", 28), metrics=accuracy)
learner = Learner(data, ResNet("resnet18", 28), metrics=accuracy)

learner.fit(1)

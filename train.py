import cv2
import PIL
import torch.nn as nn
import torch.nn.functional as F
from fastai import *
from fastai.vision import *
from sklearn.metrics import f1_score as skl_f1_score


def load_image(base_name, image_size):
    r = load_image_channel("{}_red.png".format(base_name), image_size)
    g = load_image_channel("{}_green.png".format(base_name), image_size)
    b = load_image_channel("{}_blue.png".format(base_name), image_size)
    y = load_image_channel("{}_yellow.png".format(base_name), image_size)
    return np.stack([r, g, b, y], axis=2)


def load_image_channel(file_path, image_size):
    channel = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if channel.shape[0] != image_size:
        channel = cv2.resize(channel, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return channel


def f1_score(prediction_logits, targets, threshold=0.5):
    predictions = torch.sigmoid(prediction_logits)
    return f1_score_from_probs(predictions, targets, threshold)


def f1_score_from_probs(predictions, targets, threshold=0.5):
    binary_predictions = (predictions > threshold).float()
    return torch.tensor(skl_f1_score(targets, binary_predictions, average="macro")).float().to(predictions.device)


def focal_loss(input, target, gamma=2.0):
    assert target.size() == input.size(), \
        "Target size ({}) must be the same as input size ({})".format(target.size(), input.size())

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
    loss = (invprobs * gamma).exp() * loss

    return loss.sum(dim=1).mean()


def resnet(type, pretrained, num_classes):
    if type == "resnet18":
        resnet = models.resnet18(pretrained=pretrained)
        num_fc_in_channels = 512
    elif type == "resnet34":
        resnet = models.resnet34(pretrained=pretrained)
        num_fc_in_channels = 512
    elif type == "resnet50":
        resnet = models.resnet50(pretrained=pretrained)
        num_fc_in_channels = 2048
    else:
        raise Exception("Unsupported resnet model type: '{}".format(type))

    conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv1.weight.data[:, 0:3, :, :] = resnet.conv1.weight.data
    conv1.weight.data[:, 3, :, :] = resnet.conv1.weight.data[:, 0, :, :]
    resnet.conv1 = conv1

    resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
    resnet.fc = nn.Sequential(
        nn.BatchNorm1d(num_fc_in_channels),
        nn.Dropout(0.5),
        nn.Linear(num_fc_in_channels, num_classes),
    )

    return resnet


class HpaImageItemList(ImageItemList):
    def open(self, fn):
        return pil2tensor(PIL.Image.fromarray(load_image(fn[:-4], 256)), np.float32) / 255.


def create_image(fn):
    return pil2tensor(PIL.Image.fromarray(load_image(fn[:-4], 256)), np.float32) / 255.


data = (
    HpaImageItemList
        .from_csv('/storage/kaggle/hpa', 'train.csv', folder='train', suffix='.png', create_func=create_image)
        .random_split_by_pct()
        .label_from_df(sep=' ')
        .databunch()
)

#data.show_batch(rows=3)

learner = create_cnn(
    data,
    lambda pretrained: resnet('resnet34', pretrained=pretrained, num_classes=28),
    loss_func=focal_loss,
    metrics=[f1_score])

# learner = Learner(data, ResNet("resnet18", 28), metrics=accuracy)

learner.fit(1)

learner.unfreeze()
learner.fit(10)

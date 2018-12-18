import cv2
from fastai import *
from fastai.callbacks import *
from fastai.vision import *
from pretrainedmodels.models.senet import se_resnext50_32x4d, senet154
from sklearn.metrics import f1_score as skl_f1_score

input_dir = '/storage/kaggle/hpa'
output_dir = '/artifacts'


def load_image(base_name, image_size):
    r = load_image_channel('{}_red.png'.format(base_name), image_size)
    g = load_image_channel('{}_green.png'.format(base_name), image_size)
    b = load_image_channel('{}_blue.png'.format(base_name), image_size)
    y = load_image_channel('{}_yellow.png'.format(base_name), image_size)
    return np.stack([r, g, b, y], axis=2)


def load_image_channel(file_path, image_size):
    channel = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if channel.shape[0] != image_size:
        channel = cv2.resize(channel, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return channel


def f1_score(prediction_logits, targets, threshold=0.5):
    predictions = torch.sigmoid(prediction_logits)
    binary_predictions = (predictions > threshold).float()
    return skl_f1_score(targets, binary_predictions, average='macro')


class F1Score(Callback):
    def on_epoch_begin(self, **kwargs):
        self.prediction_logits = []
        self.targets = []

    def on_batch_end(self, last_output, last_target, **kwargs):
        self.prediction_logits.extend(last_output.cpu().data.numpy())
        self.targets.extend(last_target.cpu().data.numpy())

    def on_epoch_end(self, **kwargs):
        self.metric = f1_score(torch.tensor(self.prediction_logits), torch.tensor(self.targets))


def focal_loss(input, target, gamma=2.0):
    assert target.size() == input.size(), \
        'Target size ({}) must be the same as input size ({})'.format(target.size(), input.size())

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
    loss = (invprobs * gamma).exp() * loss

    return loss.sum(dim=1).mean()


class PaperspaceLrLogger(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)
        self.batch = 0
        print('{"chart": "lr", "axis": "batch"}')

    def on_batch_begin(self, train, **kwargs):
        if train:
            self.batch += 1
            print('{"chart": "lr", "x": %d, "y": %.4f}' % (self.batch, self.learn.opt.lr))


def one_hot_to_categories(one_hot_categories):
    one_hot_categories_np = one_hot_categories.cpu().data.numpy()
    return [np.squeeze(np.argwhere(p == 1)) for p in one_hot_categories_np]


def calculate_categories(prediction_logits, threshold):
    predictions = torch.sigmoid(prediction_logits)
    predictions_np = predictions.cpu().data.numpy()
    return [np.squeeze(np.argwhere(p > threshold), axis=1) for p in predictions_np]


def calculate_best_threshold(prediction_logits, targets):
    thresholds = np.linspace(0, 1, 51)
    scores = [f1_score(prediction_logits, targets, threshold=t) for t in thresholds]

    best_score_index = np.argmax(scores)

    return thresholds[best_score_index], scores[best_score_index], scores


def create_resnet(type, pretrained, num_classes):
    if type == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        num_fc_in_channels = 512
    elif type == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        num_fc_in_channels = 512
    elif type == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_fc_in_channels = 2048
    else:
        raise Exception('Unsupported model model type: "{}"'.format(type))

    conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv1.weight.data[:, 0:3, :, :] = model.conv1.weight.data
    conv1.weight.data[:, 3, :, :] = model.conv1.weight.data[:, 0, :, :].clone()
    model.conv1 = conv1

    model.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
    model.fc = nn.Linear(num_fc_in_channels, num_classes)

    return model


def resnet_split(m):
    return (m[0][6], m[1])


def create_senet(type, num_classes):
    if type == 'seresnext50':
        model = se_resnext50_32x4d(pretrained='imagenet')
        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif type == 'senet154':
        model = senet154(pretrained='imagenet')
        conv1 = nn.Conv2d(4, 64, 3, stride=2, padding=1, bias=False)
    else:
        raise Exception('Unsupported model model type: ''{}'''.format(type))

    senet_layer0_children = list(model.layer0.children())
    conv1.weight.data[:, 0:3, :, :] = senet_layer0_children[0].weight.data
    model.layer0 = nn.Sequential(*([conv1] + senet_layer0_children[1:]))

    model.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
    model.dropout = nn.Dropout(0.5)
    model.last_linear = nn.Linear(2048, num_classes)

    return model


def create_image(fn):
    return Image(pil2tensor(PIL.Image.fromarray(load_image(fn, 256)), np.float32) / 255.)


def write_submission(prediction_categories, filename):
    submission_df = pd.read_csv('{}/sample_submission.csv'.format(input_dir), index_col='Id', usecols=['Id'])
    submission_df['Predicted'] = [' '.join(map(str, c)) for c in prediction_categories]
    submission_df.to_csv(filename)


class HpaImageItemList(ImageItemList):
    def open(self, fn):
        return create_image(fn)


# shutil.copytree('/storage/models/hpa/fastai/models', '{}/models'.format(output_dir))

protein_stats = ([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])
tfms = get_transforms(flip_vert=True, xtra_tfms=zoom_crop(scale=(0.8, 1.2), do_rand=True))

test_images = (
    HpaImageItemList
        .from_csv(input_dir, 'sample_submission.csv', folder='test', create_func=create_image)
)

data = (
    HpaImageItemList
        .from_csv(input_dir, 'train.csv', folder='train', create_func=create_image)
        # .use_partial_data(sample_pct=0.2, seed=42)
        .random_split_by_pct(valid_pct=0.2, seed=42)
        .label_from_df(sep=' ', classes=[str(i) for i in range(28)])
        .transform(tfms)
        .add_test(test_images)
        # .databunch(bs=64, num_workers=8)
        .databunch(bs=64)
        .normalize(protein_stats)
)

# data.show_batch(rows=3)


learn = create_cnn(
    data,
    lambda pretrained: create_resnet('resnet34', pretrained, num_classes=28),
    pretrained=True,
    ps=0.5,
    split_on=resnet_split,
    path=Path(output_dir),
    loss_func=focal_loss,
    metrics=[F1Score()])

learn.callbacks = [
    EarlyStoppingCallback(learn, monitor='f1_score', mode='max', patience=5, min_delta=1e-3),
    SaveModelCallback(learn, monitor='val_loss', mode='min', name='model_best_loss'),
    SaveModelCallback(learn, monitor='f1_score', mode='max', name='model_best_f1')
]

# learn.load('model_best_f1')

# print(learn.summary)

# learn.lr_find()
# learn.recorder.plot()

lr = 3e-3
learn.freeze()
learn.fit(5, lr=lr)

lr = 3e-3
learn.unfreeze()
learn.fit_one_cycle(10, max_lr=lr)
learn.fit_one_cycle(10, max_lr=lr)
learn.fit_one_cycle(10, max_lr=slice(lr / 10, lr))
learn.fit_one_cycle(10, max_lr=slice(lr / 10, lr))

learn.load('model_best_f1')

valid_prediction_logits, valid_prediction_categories_one_hot = learn.get_preds(ds_type=DatasetType.Valid)
best_threshold, best_score, _ = calculate_best_threshold(valid_prediction_logits, valid_prediction_categories_one_hot)
print('best threshold / score: {:.3f} / {:.3f}'.format(best_threshold, best_score))

test_prediction_logits, _ = learn.get_preds(ds_type=DatasetType.Test)
test_prediction_categories = calculate_categories(test_prediction_logits, best_threshold)
write_submission(test_prediction_categories, '{}/submission.csv'.format(output_dir))
np.save('{}/test_prediction_logits.npy'.format(output_dir), test_prediction_logits.cpu().data.numpy())

valid_prediction_logits, valid_prediction_categories_one_hot = learn.TTA(ds_type=DatasetType.Valid)
best_threshold, best_score, _ = calculate_best_threshold(valid_prediction_logits, valid_prediction_categories_one_hot)
print('best threshold / score: {:.3f} / {:.3f}'.format(best_threshold, best_score))

test_prediction_logits, _ = learn.TTA(ds_type=DatasetType.Test)
test_prediction_categories = calculate_categories(test_prediction_logits, best_threshold)
write_submission(test_prediction_categories, '{}/submission_tta.csv'.format(output_dir))
np.save('{}/test_prediction_logits_tta.npy'.format(output_dir), test_prediction_logits.cpu().data.numpy())

learn.load('model_best_loss')

valid_prediction_logits, valid_prediction_categories_one_hot = learn.TTA(ds_type=DatasetType.Valid)
best_threshold, best_score, _ = calculate_best_threshold(valid_prediction_logits, valid_prediction_categories_one_hot)
print('best threshold / score: {:.3f} / {:.3f}'.format(best_threshold, best_score))

test_prediction_logits, _ = learn.TTA(ds_type=DatasetType.Test)
test_prediction_categories = calculate_categories(test_prediction_logits, best_threshold)
write_submission(test_prediction_categories, '{}/submission_best_loss_tta.csv'.format(output_dir))
np.save('{}/test_prediction_logits_best_loss_tta.npy'.format(output_dir), test_prediction_logits.cpu().data.numpy())

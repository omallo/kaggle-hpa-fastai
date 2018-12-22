import cv2
from fastai import *
from fastai.callbacks import *
from fastai.vision import *
from pretrainedmodels.models.senet import se_resnext50_32x4d, senet154
from sklearn.metrics import f1_score as skl_f1_score

input_dir = '/storage/kaggle/hpa'
output_dir = '/artifacts'
base_model_dir = None
image_size = 256
batch_size = 32
num_cycles = 2
use_progressive_image_resizing = False
progressive_image_size_start = 128
progressive_image_size_end = 512


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


def f1_loss(logits, targets):
    epsilon = 1e-6
    beta = 1
    batch_size = logits.size()[0]

    p = F.sigmoid(logits)
    l = targets
    num_pos = torch.sum(p, 1) + epsilon
    num_pos_hat = torch.sum(l, 1) + epsilon
    tp = torch.sum(l * p, 1)
    precise = tp / num_pos
    recall = tp / num_pos_hat
    fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + epsilon)
    loss = fs.sum() / batch_size
    return 1 - loss


def focal_f1_combined_loss(logits, targets, alpha=0.5):
    return alpha * focal_loss(logits, targets) + (1 - alpha) * f1_loss(logits, targets)


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


def create_resnet(type, pretrained):
    if type == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif type == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif type == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise Exception('Unsupported model type: "{}"'.format(type))

    conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv1.weight.data[:, 0:3, :, :] = model.conv1.weight.data
    conv1.weight.data[:, 3, :, :] = model.conv1.weight.data[:, 0, :, :].clone()
    model.conv1 = conv1

    return model


def resnet34(pretrained):
    return create_resnet('resnet34', pretrained)


def resnet50(pretrained):
    return create_resnet('resnet50', pretrained)


def resnet_split(m):
    return (m[0][6], m[1])


def create_senet(type, pretrained):
    if type == 'seresnext50':
        model = se_resnext50_32x4d(pretrained='imagenet' if pretrained else None)
        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif type == 'senet154':
        model = senet154(pretrained='imagenet' if pretrained else None)
        conv1 = nn.Conv2d(4, 64, 3, stride=2, padding=1, bias=False)
    else:
        raise Exception('Unsupported model type: ''{}'''.format(type))

    senet_layer0_children = list(model.layer0.children())
    conv1.weight.data[:, 0:3, :, :] = senet_layer0_children[0].weight.data
    conv1.weight.data[:, 3, :, :] = senet_layer0_children[0].weight.data[:, 0, :, :].clone()
    model.layer0 = nn.Sequential(*([conv1] + senet_layer0_children[1:]))

    return model


def seresnext50(pretrained):
    return create_senet('seresnext50', pretrained)


def create_image(fn):
    return Image(pil2tensor(PIL.Image.fromarray(load_image(fn, image_size)), np.float32) / 255.)


def write_submission(prediction_categories, filename):
    submission_df = pd.read_csv('{}/sample_submission.csv'.format(input_dir), index_col='Id', usecols=['Id'])
    submission_df['Predicted'] = [' '.join(map(str, c)) for c in prediction_categories]
    submission_df.to_csv(filename)


class HpaImageItemList(ImageItemList):
    def open(self, fn):
        return create_image(fn)


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
        .databunch(bs=batch_size)
        .normalize(protein_stats)
)

# data.show_batch(rows=3)


learn = create_cnn(
    data,
    resnet34,
    pretrained=True,
    cut=-2,
    ps=0.5,
    split_on=resnet_split,
    path=Path(output_dir),
    loss_func=focal_loss,
    metrics=[F1Score()])

learn.callbacks = [
    # EarlyStoppingCallback(learn, monitor='f1_score', mode='max', patience=5, min_delta=1e-3),
    SaveModelCallback(learn, monitor='val_loss', mode='min', name='model_best_loss'),
    SaveModelCallback(learn, monitor='f1_score', mode='max', name='model_best_f1'),
    MixUpCallback(learn, alpha=0.4, stack_x=False, stack_y=False),  # stack_y=True leads to error
]

# print(learn.summary)

if base_model_dir:
    shutil.copytree('{}/models'.format(base_model_dir), '{}/models'.format(output_dir))
    learn.load('model_best_f1')

# learn.lr_find()
# learn.recorder.plot()

lr = 0.003

if use_progressive_image_resizing:
    image_sizes = np.linspace(progressive_image_size_start, progressive_image_size_end, num_cycles, dtype=np.int32)
else:
    image_sizes = np.array([image_size] * num_cycles)
print('Image sizes: {}'.format(image_sizes))

image_size = image_sizes[0]
learn.freeze()
learn.fit(3, lr=lr)
learn.unfreeze()
for c in range(num_cycles):
    image_size = image_sizes[c]
    learn.fit_one_cycle(10, max_lr=lr)
learn.fit_one_cycle(10, max_lr=slice(lr / 10, lr))

learn.load('model_best_f1')

valid_prediction_logits, valid_prediction_categories_one_hot = learn.TTA(ds_type=DatasetType.Valid)
best_threshold, best_score, _ = calculate_best_threshold(valid_prediction_logits, valid_prediction_categories_one_hot)
print('best threshold / score: {:.3f} / {:.3f}'.format(best_threshold, best_score))

test_prediction_logits, _ = learn.TTA(ds_type=DatasetType.Test)
test_prediction_categories = calculate_categories(test_prediction_logits, best_threshold)
write_submission(test_prediction_categories, '{}/submission_best_f1.csv'.format(output_dir))
np.save('{}/test_prediction_logits_best_f1.npy'.format(output_dir), test_prediction_logits.cpu().data.numpy())

learn.load('model_best_loss')

valid_prediction_logits, valid_prediction_categories_one_hot = learn.TTA(ds_type=DatasetType.Valid)
best_threshold, best_score, _ = calculate_best_threshold(valid_prediction_logits, valid_prediction_categories_one_hot)
print('best threshold / score: {:.3f} / {:.3f}'.format(best_threshold, best_score))

test_prediction_logits, _ = learn.TTA(ds_type=DatasetType.Test)
test_prediction_categories = calculate_categories(test_prediction_logits, best_threshold)
write_submission(test_prediction_categories, '{}/submission_best_loss.csv'.format(output_dir))
np.save('{}/test_prediction_logits_best_loss.npy'.format(output_dir), test_prediction_logits.cpu().data.numpy())

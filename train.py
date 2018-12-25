import cv2
import scipy
import torchvision
from fastai import *
from fastai.callbacks import *
from fastai.vision import *
from pretrainedmodels.models.nasnet import nasnetalarge
from pretrainedmodels.models.senet import se_resnext50_32x4d, senet154
from sklearn.metrics import f1_score as skl_f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data.sampler import WeightedRandomSampler

input_dir = '/storage/kaggle/hpa'
output_dir = '/artifacts'
base_model_dir = None  # '/storage/models/hpa/resnet34'
image_size = 512
batch_size = 16
lr = 0.001
num_cycles = 7
cycle_len = 10
use_sampling = False
use_progressive_image_resizing = True
progressive_image_size_start = 128
progressive_image_size_end = 512
do_train = True

name_label_dict = {
    0: ('Nucleoplasm', 12885),
    1: ('Nuclear membrane', 1254),
    2: ('Nucleoli', 3621),
    3: ('Nucleoli fibrillar center', 1561),
    4: ('Nuclear speckles', 1858),
    5: ('Nuclear bodies', 2513),
    6: ('Endoplasmic reticulum', 1008),
    7: ('Golgi apparatus', 2822),
    8: ('Peroxisomes', 53),
    9: ('Endosomes', 45),
    10: ('Lysosomes', 28),
    11: ('Intermediate filaments', 1093),
    12: ('Actin filaments', 688),
    13: ('Focal adhesion sites', 537),
    14: ('Microtubules', 1066),
    15: ('Microtubule ends', 21),
    16: ('Cytokinetic bridge', 530),
    17: ('Mitotic spindle', 210),
    18: ('Microtubule organizing center', 902),
    19: ('Centrosome', 1482),
    20: ('Lipid droplets', 172),
    21: ('Plasma membrane', 3777),
    22: ('Cell junctions', 802),
    23: ('Mitochondria', 2965),
    24: ('Aggresome', 322),
    25: ('Cytosol', 8228),
    26: ('Cytoplasmic bodies', 328),
    27: ('Rods &amp; rings', 11)
}

n_labels = 50782


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


@dataclass
class MultiTrainSaveModelCallback(TrackerCallback):
    name: str = 'bestmodel'

    def on_train_begin(self, **kwargs):
        super().on_train_begin(**kwargs)
        if not hasattr(self, 'best_global'):
            self.best_global = self.best
            self.cycle = 1
        else:
            self.cycle += 1

    def on_epoch_end(self, epoch, **kwargs):
        current = self.get_monitor_value()
        if current is not None and self.operator(current, self.best):
            self.best = current
            self.learn.save(f'{self.name}_{self.cycle}')
        if current is not None and self.operator(current, self.best_global):
            self.best_global = current
            self.learn.save(f'{self.name}')


@dataclass
class MultiTrainEarlyStoppingCallback(TrackerCallback):
    min_delta: int = 0
    patience: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.operator == np.less:
            self.min_delta *= -1

    def on_train_begin(self, **kwargs):
        if not hasattr(self, 'best'):
            super().on_train_begin(**kwargs)
            self.wait = 0
            self.early_stopped = False

    def on_epoch_end(self, epoch, **kwargs):
        current = self.get_monitor_value()
        if current is None:
            return
        if self.operator(current - self.min_delta, self.best):
            self.best, self.wait = current, 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                print(f'Epoch {epoch}: early stopping')
                self.early_stopped = True
                return True


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


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def f1_soft(prediction_logits, targets, th=0.0, d=25.0):
    prediction_logits = sigmoid_np(d * (prediction_logits - th))
    targets = targets.astype(np.float)
    score = 2.0 * (prediction_logits * targets).sum(axis=0) / ((prediction_logits + targets).sum(axis=0) + 1e-6)
    return score


def calculate_best_threshold(prediction_logits, targets, per_class):
    if per_class:
        prediction_logits = prediction_logits.cpu().data.numpy()
        targets = targets.cpu().data.numpy()
        params = np.zeros(28)
        wd = 1e-5
        error = lambda p: np.concatenate((f1_soft(prediction_logits, targets, p) - 1.0, wd * p), axis=None)
        p, success = scipy.optimize.leastsq(error, params)
        return p
    else:
        thresholds = np.linspace(0, 1, 51)
        scores = [f1_score(prediction_logits, targets, threshold=t) for t in thresholds]
        best_score_index = np.argmax(scores)
        return thresholds[best_score_index]


def create_resnet(type, pretrained):
    if type == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif type == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif type == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif type == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    elif type == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
    else:
        raise Exception('Unsupported model type: "{}"'.format(type))

    conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv1.weight.data[:, 0:3, :, :] = model.conv1.weight.data
    conv1.weight.data[:, 3, :, :] = model.conv1.weight.data[:, 0, :, :].clone()
    model.conv1 = conv1

    return model


def resnet18(pretrained):
    return create_resnet('resnet18', pretrained)


def resnet34(pretrained):
    return create_resnet('resnet34', pretrained)


def resnet50(pretrained):
    return create_resnet('resnet50', pretrained)


def resnet101(pretrained):
    return create_resnet('resnet101', pretrained)


def resnet152(pretrained):
    return create_resnet('resnet152', pretrained)


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


def senet(pretrained):
    return create_senet('senet154', pretrained)


def create_nasnet(pretrained):
    model = nasnetalarge(num_classes=1000, pretrained='imagenet' if pretrained else None)

    conv1 = nn.Conv2d(in_channels=4, out_channels=96, kernel_size=3, padding=0, stride=2, bias=False)

    nasnet_conv0_children = list(model.conv0.children())
    conv1.weight.data[:, 0:3, :, :] = nasnet_conv0_children[0].weight.data
    conv1.weight.data[:, 3, :, :] = nasnet_conv0_children[0].weight.data[:, 0, :, :].clone()
    model.conv0 = nn.Sequential(*([conv1] + nasnet_conv0_children[1:]))

    return model


def nasnet(pretrained):
    return create_nasnet(pretrained)


def create_image(fn):
    return Image(pil2tensor(PIL.Image.fromarray(load_image(fn, image_size)), np.float32) / 255.)


def write_submission(prediction_categories, filename):
    submission_df = pd.read_csv('{}/sample_submission.csv'.format(input_dir), index_col='Id', usecols=['Id'])
    submission_df['Predicted'] = [' '.join(map(str, c)) for c in prediction_categories]
    submission_df.to_csv(filename)


def cls_wts(label_dict, mu=0.5):
    prob_dict, prob_dict_bal = {}, {}
    max_ent_wt = 1 / 28
    for i in range(28):
        prob_dict[i] = label_dict[i][1] / n_labels
        if prob_dict[i] > max_ent_wt:
            prob_dict_bal[i] = prob_dict[i] - mu * (prob_dict[i] - max_ent_wt)
        else:
            prob_dict_bal[i] = prob_dict[i] + mu * (max_ent_wt - prob_dict[i])
    return prob_dict, prob_dict_bal


def calculate_balance_weights(ds):
    prob_dict, prob_dict_bal = cls_wts(name_label_dict, mu=0.4)
    class_weights = np.array([prob_dict_bal[c] / prob_dict[c] for c in range(28)])
    class_frequencies = np.array([name_label_dict[k][1] for k in sorted(name_label_dict.keys())])

    weights = []
    labels = []
    for y in ds.y:
        w = class_weights * y.data
        weights.append(np.max(w))

        f = class_frequencies * y.data
        f[f == 0] = 1000000
        labels.append(np.argmin(f))

    return weights, labels, class_weights.tolist()


class StratifiedSampler(Sampler):
    def __init__(self, class_vector, batch_size):
        super().__init__(None)
        self.class_vector = class_vector
        self.batch_size = batch_size

    def gen_sample_array(self):
        n_splits = math.ceil(len(self.class_vector) / self.batch_size)
        splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.5)
        train_index, test_index = next(splitter.split(np.zeros(len(self.class_vector)), self.class_vector))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


class HpaSampler(Sampler):
    def __init__(self, weights, labels):
        super().__init__(None)
        self.weights = weights
        self.labels = labels

    def __iter__(self):
        weighted_random_sampler = WeightedRandomSampler(self.weights, len(self.weights))
        weighted_indexes = list(iter(weighted_random_sampler))
        weighted_labels = [self.labels[i] for i in weighted_indexes]

        stratified_sampler = StratifiedSampler(weighted_labels, batch_size)
        batch_indexes = list(iter(stratified_sampler))

        assert len(weighted_indexes) == len(batch_indexes)

        final_indexes = [weighted_indexes[b] for b in batch_indexes]

        return iter(final_indexes)

    def __len__(self):
        return len(self.weights)


class HpaImageDataBunch(ImageDataBunch):
    @classmethod
    def create(cls, train_ds: Dataset, valid_ds: Dataset, test_ds: Optional[Dataset] = None, path: PathOrStr = '.',
               bs: int = 64,
               num_workers: int = defaults.cpus, tfms: Optional[Collection[Callable]] = None,
               device: torch.device = None,
               collate_fn: Callable = data_collate, no_check: bool = False) -> 'DataBunch':
        "Create a `DataBunch` from `train_ds`, `valid_ds` and maybe `test_ds` with a batch size of `bs`."
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = bs

        if use_sampling:
            train_weights, train_weight_labels, _ = calculate_balance_weights(train_ds)
            train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
        else:
            train_sampler = None

        dls = [DataLoader(d, b, shuffle=s, sampler=p, drop_last=(s and b > 1), num_workers=num_workers) for d, b, s, p
               in
               zip(datasets, (bs, val_bs, val_bs, val_bs), (False, False, False, False),
                   (train_sampler, None, None, None))]
        return cls(*dls, path=path, device=device, tfms=tfms, collate_fn=collate_fn, no_check=no_check)


class HpaImageItemList(ImageItemList):
    _bunch = HpaImageDataBunch

    def open(self, fn):
        return create_image(fn)


def shuffle_tfm(image, **kwargs):
    if np.random.rand() < 0.5:
        return image

    dst_image = image.clone()

    shuffled_cells = np.arange(9)
    np.random.shuffle(shuffled_cells)
    cell_size = int(image.shape[1] // 3)

    for i, c in enumerate(shuffled_cells):
        src_x = int(i // 3) * cell_size
        src_y = int(i % 3) * cell_size

        dst_x = int(c // 3) * cell_size
        dst_y = int(c % 3) * cell_size

        cell = image[:, src_x:src_x + cell_size, src_y:src_y + cell_size]

        cell_pil = torchvision.transforms.functional.to_pil_image(cell)
        if np.random.rand() < 0.5:
            cell_pil = torchvision.transforms.functional.hflip(cell_pil)
        if np.random.rand() < 0.5:
            cell_pil = torchvision.transforms.functional.vflip(cell_pil)
        if np.random.rand() < 0.5:
            cell_pil = torchvision.transforms.functional.rotate(cell_pil, angle=np.random.choice([90, 180, 270]))
        cell = torchvision.transforms.functional.to_tensor(cell_pil)

        dst_image[:, dst_x:dst_x + cell_size, dst_y:dst_y + cell_size] = cell

    return dst_image


protein_stats = ([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])

tfms = get_transforms(
    flip_vert=True,
    max_rotate=20,
    max_zoom=1.2,
    xtra_tfms=[*zoom_crop(scale=(0.8, 1.2), do_rand=True), TfmPixel(shuffle_tfm)()])

test_images = (
    HpaImageItemList
        .from_csv(input_dir, 'sample_submission.csv', folder='test', create_func=create_image)
)

data = (
    HpaImageItemList
        .from_csv(input_dir, 'train.csv', folder='train', create_func=create_image)
        # .use_partial_data(sample_pct=0.005, seed=42)
        .random_split_by_pct(valid_pct=0.2, seed=42)
        .label_from_df(sep=' ', classes=[str(i) for i in range(28)])
        .transform(tfms)
        .add_test(test_images)
        # .databunch(bs=64, num_workers=8)
        .databunch(bs=batch_size)
        .normalize(protein_stats)
)

# data.show_batch(rows=3)

if base_model_dir is not None:
    shutil.copytree('{}/models'.format(base_model_dir), '{}/models'.format(output_dir))

learn = create_cnn(
    data,
    senet,
    pretrained=True,
    cut=-3,
    ps=0.5,
    # split_on=resnet_split,
    path=Path(output_dir),
    loss_func=focal_loss,
    metrics=[F1Score()])

early_stopper = \
    MultiTrainEarlyStoppingCallback(learn, monitor='f1_score', mode='max', patience=cycle_len, min_delta=1e-3)
best_f1_model_saver = MultiTrainSaveModelCallback(learn, monitor='f1_score', mode='max', name='model')
mixup = MixUpCallback(learn, alpha=0.4, stack_x=False, stack_y=False)  # stack_y=True leads to error

learn.callbacks = [
    early_stopper,
    best_f1_model_saver,
    mixup
]

# print(learn.summary)

if base_model_dir is not None:
    learn.load('model')

# learn.lr_find()
# learn.recorder.plot()

if use_progressive_image_resizing:
    image_sizes = np.linspace(progressive_image_size_start, progressive_image_size_end, num_cycles, dtype=np.int32)
else:
    image_sizes = np.array([image_size] * num_cycles)
print('Image sizes: {}'.format(image_sizes))

if do_train:
    if base_model_dir is None:
        image_size = image_sizes[0]
        learn.freeze()
        learn.fit(3, lr=lr)

    learn.unfreeze()
    for c in range(num_cycles):
        image_size = image_sizes[c]
        learn.fit_one_cycle(cycle_len, max_lr=lr)
        if early_stopper.early_stopped:
            break
    if not early_stopper.early_stopped:
        learn.fit_one_cycle(cycle_len, max_lr=slice(lr / 10, lr))

    print('best f1 score: {:.6f}'.format(best_f1_model_saver.best_global))

learn.load('model')

valid_prediction_logits, valid_prediction_categories_one_hot = learn.TTA(ds_type=DatasetType.Valid)
np.save('{}/valid_prediction_logits.npy'.format(output_dir), valid_prediction_logits.cpu().data.numpy())
np.save('{}/valid_prediction_categories.npy'.format(output_dir), valid_prediction_categories_one_hot.cpu().data.numpy())

test_prediction_logits, _ = learn.TTA(ds_type=DatasetType.Test)
np.save('{}/test_prediction_logits.npy'.format(output_dir), test_prediction_logits.cpu().data.numpy())

best_threshold = calculate_best_threshold(valid_prediction_logits, valid_prediction_categories_one_hot, per_class=False)
best_score = f1_score(valid_prediction_logits, valid_prediction_categories_one_hot, threshold=best_threshold)
print('best threshold / score: {} / {:.6f}'.format(best_threshold, best_score))
test_prediction_categories = calculate_categories(test_prediction_logits, best_threshold)
write_submission(test_prediction_categories, '{}/submission_single_threshold.csv'.format(output_dir))

best_threshold = calculate_best_threshold(valid_prediction_logits, valid_prediction_categories_one_hot, per_class=True)
best_score = f1_score(valid_prediction_logits, valid_prediction_categories_one_hot, threshold=best_threshold)
print('best threshold / score: {} / {:.6f}'.format(best_threshold, best_score))
test_prediction_categories = calculate_categories(test_prediction_logits, best_threshold)
write_submission(test_prediction_categories, '{}/submission_class_threshold.csv'.format(output_dir))

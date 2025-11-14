# coding: utf8
from collections import Counter, OrderedDict
from torch.utils.data.dataloader import default_collate
from itertools import chain
import six
import torch
import numpy as np
import h5py
import os
import warnings
import shutil

from .dataset import Dataset
from .vocab import Vocab
from .utils import get_tokenizer

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import clip

def _convert_image_to_rgb(image):
    return image.convert('RGB')

class RawField(object):
    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, *args, **kwargs):
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)
        return default_collate(batch)


class Merge(RawField):
    def __init__(self, *fields):
        super(Merge, self).__init__()
        self.fields = fields

    def preprocess(self, x):
        return tuple(f.preprocess(x) for f in self.fields)

    def process(self, batch, *args, **kwargs):
        if len(self.fields) == 1:
            batch = [batch, ]
        else:
            batch = list(zip(*batch))
        out = list(f.process(b, *args, **kwargs) for f, b in zip(self.fields, batch))
        return out


class ImageDetectionsField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, detections_path=None,
                 labels_path=None, max_detections=100, sort_by_prob=False,
                 load_in_tmp=True, pre_name='RN101'):
        self.max_detections = max_detections
        self.detections_path = detections_path
        self.labels_path = labels_path
        self.sort_by_prob = sort_by_prob

        preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])
        self.transform = preprocess
        tmp_detections_path = os.path.join('/tmp', os.path.basename(detections_path)) if detections_path else None

        if load_in_tmp and detections_path:
            if not os.path.isfile(tmp_detections_path):
                if shutil.disk_usage("/tmp")[-1] < os.path.getsize(detections_path):
                    warnings.warn('Loading from %s, because /tmp has no enough space.' % detections_path)
                else:
                    warnings.warn("Copying detection file to /tmp")
                    shutil.copyfile(detections_path, tmp_detections_path)
                    warnings.warn("Done.")
                    self.detections_path = tmp_detections_path
            else:
                self.detections_path = tmp_detections_path

        super(ImageDetectionsField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x, avoid_precomp=False):
        """
        兼容相对/绝对路径，并避免重复拼接根前缀：
        1) 原样路径
        2) 相对路径拼上 features_path
        3) 旧绝对前缀改到 features_path
        4) 仅文件名则尝试 train2014/、val2014/
        """
        root = getattr(self, 'detections_path', None)
        if root is None:
            raise ValueError("[ImageDetectionsField] detections_path 为空，请用 --features_path 指向图片根目录")

        root_abs = os.path.abspath(root)
        path = x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else x
        if not isinstance(path, str):
            raise TypeError(f"[ImageDetectionsField.preprocess] 非字符串路径: {type(path)} -> {path}")

        candidates = []
        candidates.append(path)
        candidates.append(os.path.abspath(path))

        if not os.path.isabs(path):
            p_rel = path[7:] if path.startswith('images/') else path
            candidates.append(os.path.join(root_abs, p_rel.lstrip('/')))

        base = os.path.basename(path)
        if base == path or ('/' not in path and '\\' not in path):
            candidates.append(os.path.join(root_abs, 'train2014', base))
            candidates.append(os.path.join(root_abs, 'val2014', base))

        old_prefixes = [
            '/root/datasets/coco/images',
            '/dataset/coco/images',
            '/data/coco/images',
        ]
        for op in old_prefixes:
            if path.startswith(op):
                candidates.append(os.path.join(root_abs, path[len(op):].lstrip('/')))

        seen = set()
        uniq = []
        for c in candidates:
            c_norm = os.path.normpath(c)
            if c_norm not in seen:
                seen.add(c_norm)
                uniq.append(c_norm)

        resolved = None
        for c in uniq:
            if os.path.exists(c):
                resolved = c
                break

        if resolved is None:
            debug_msg = "\n".join(["  - " + c for c in uniq[:8]])
            raise FileNotFoundError(
                f"[ImageDetectionsField.preprocess] 找不到图像文件: {path}\n"
                f"features_path(root) = {root}\n"
                f"尝试过的候选路径（前几项）：\n{debug_msg}"
            )

        try:
            image_id = int(os.path.basename(resolved).split('_')[-1].split('.')[0])
        except Exception:
            image_id = abs(hash(os.path.basename(resolved))) % (10**9)

        precomp_data = self.transform(Image.open(resolved).convert('RGB'))

        if self.labels_path:
            f = h5py.File(self.labels_path, 'r')
            labels = f['%d_labels' % image_id][()]
            return precomp_data, labels.astype(np.float32)

        return image_id, precomp_data, np.random.randn(100, 2048).astype(np.float32)


class TextField(RawField):
    vocab_cls = Vocab
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,
        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }
    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    def __init__(self, use_vocab=True, init_token=None, eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False, tokenize=(lambda s: s.split()),
                 remove_punctuation=False, include_lengths=False, batch_first=True, pad_token="<pad>",
                 unk_token="<unk>", pad_first=False, truncate_first=False, vectors=None, nopoints=True):
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.remove_punctuation = remove_punctuation
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        self.vocab = None
        self.vectors = vectors
        if nopoints:
            self.punctuations.append("..")

        super(TextField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        if six.PY2 and isinstance(x, six.string_types) and not isinstance(x, six.text_type):
            x = six.text_type(x, encoding='utf-8')
        if self.lower:
            x = six.text_type.lower(x)
        x = self.tokenize(x.rstrip('\n'))
        if self.remove_punctuation:
            x = [w for w in x if w not in self.punctuations]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device=None):
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in arg.fields.items() if field is self]
            else:
                sources.append(arg)

        for data in sources:
            for x in data:
                x = self.preprocess(x)
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))

        specials = list(OrderedDict.fromkeys([
            tok for tok in [self.unk_token, self.pad_token, self.init_token, self.eos_token]
            if tok is not None
        ]))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def pad(self, minibatch):
        minibatch = list(minibatch)
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return padded, lengths
        return padded

    def numericalize(self, arr, device=None):
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but input is not a (data, lengths) tuple.")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)
            var = torch.tensor(arr, dtype=self.dtype, device=device)
        else:
            if self.vectors:
                arr = [[self.vectors[x] for x in ex] for ex in arr]
            numericalization_func = {
                **self.dtypes
            }[self.dtype]
            arr = [numericalization_func(x) if isinstance(x, six.string_types) else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)
            var = torch.cat([torch.cat([a.unsqueeze(0) for a in ar]).unsqueeze(0) for ar in arr])

        if not self.batch_first:
            var.t_()
        var = var.contiguous()
        if self.include_lengths:
            return var, lengths
        return var

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode([word_idxs, ], join_words)[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode([word_idxs, ], join_words)[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[int(wi)]
                if word == self.eos_token:
                    break
                caption.append(word)
            if join_words:
                caption = ' '.join(caption)
            captions.append(caption)
        return captions

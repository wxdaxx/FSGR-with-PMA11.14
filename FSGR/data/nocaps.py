import os
import json
from PIL import Image

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from utils.utils import nested_tensor_from_tensor_list

def _convert_image_to_rgb(image):
    return image.convert('RGB')

class NocapsDataset:
    def __init__(
        self,
        vocab,
        ann_path,
        root,
        pad_idx=3,
    ):
        anns = json.load(open(ann_path))
        self.imageid_to_anns = {ann['img_id']: ann for ann in anns}
        self.root = root
        self.image_ids = list(self.imageid_to_anns.keys())
        self.vocab = vocab
        self.pad_idx = pad_idx

        preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self.transform = preprocess
        

    def __getitem__(self, index: int):
        item = {}
        item['image_id'] = self.image_ids[index]
        ann = self.imageid_to_anns[item['image_id']]

        img_path = os.path.join(self.root, ann['image'])
        item['sample'] = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            item['sample'] = self.transform(item['sample'])
        return item

    def __len__(self):
        return len(self.image_ids)


class NoCapsCollator:

    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, batch_list):
        batch = {}
        imgs = [item['sample'] for item in batch_list]
        batch['samples'] = nested_tensor_from_tensor_list(imgs).tensors.to(self.device)
        batch['image_id'] = [item['image_id'] for item in batch_list]
        return batch

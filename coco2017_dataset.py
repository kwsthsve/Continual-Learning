import random
import torch
import torch.utils.data as data
import os
import nltk
from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):

    # COCO Custom Dataset compatible with torch.utils.data.DataLoader.

    def __init__(self, root, json, vocab, train, transform=None):

        """
        Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """

        self.root = root
        self.coco = COCO(json)
        self.train = train

        # Select all captions for each training image
        # if not self.train:
        #     self.ids = list(sorted(self.coco.imgs.keys()))
        # else:
        #     self.ids = list(self.coco.anns.keys())

        # Select a random caption for each training image
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):

        # Returns one data pair (image and list of captions).

        coco = self.coco
        vocab = self.vocab

        if not self.train:

            img_id = self.ids[index]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            captions = [coco.anns[ann_id]['caption'] for ann_id in ann_ids]

        else:

            # Select all captions for each image
            # ann_id = self.ids[index]
            # caption = coco.anns[ann_id]['caption']
            # img_id = coco.anns[ann_id]['image_id']

            # Select a random caption for each image
            img_id = self.ids[index]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            ann_id = random.choice(ann_ids)
            caption = coco.anns[ann_id]['caption']
            captions = [caption]

        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert captions to token indices
        target = []
        for caption in captions:
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target.append(torch.tensor(caption, dtype=torch.long))

        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):

    """
    Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 224, 224).
            - caption: list of torch tensors of shape (?); variable length.

    Returns (Training Mode):
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.

    Returns (Test Mode):
        images: torch tensor of shape (1, 3, 224, 224).
        all_captions: tuple with list of torch tensors of shape (?).
    """

    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1][0]), reverse=True)

    # Separate images and captions
    images, all_captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Check if we're in training mode (single caption per image) or test mode (multiple captions per image)
    if len(all_captions[0]) > 1:

        # Test mode: multiple captions per image
        return images, all_captions

    else:

        # Training mode: single caption per image

        # Get lengths of each caption
        lengths = [len(cap[0]) for cap in all_captions]

        # Create tensor of captions
        targets = torch.zeros(len(all_captions), 60).long()
        for i, cap in enumerate(all_captions):
            end = lengths[i]
            targets[i, :end] = cap[0][:end]

        return images, targets, lengths


def get_loader(root, json, transform, batch_size, shuffle, vocab, first=True, train=True):

    if not first:
        old_vocab_length = len(vocab)
        vocab.add_captions(annotations=json)
        print(f'\nAdded {len(vocab) - old_vocab_length} tokens to vocabulary!\n')

    # COCO caption dataset
    coco = CocoDataset(root=root, json=json, vocab=vocab, train=train, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return data_loader

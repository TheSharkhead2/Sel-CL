import os
import pickle
from torchvision.datasets.folder import ImageFolder
import torchvision as tv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import time
from IPython import embed


def sample_traning_set(train_imgs, labels, num_class, num_samples):
    random.shuffle(train_imgs)
    class_num = torch.zeros(num_class)
    sampled_train_imgs = []
    for impath in train_imgs:
        label = labels[impath]
        if class_num[label] < (num_samples / num_class):
            sampled_train_imgs.append(impath)
            class_num[label] += 1
        if len(sampled_train_imgs) >= num_samples:
            break
    return sampled_train_imgs


def get_all_classes(*dirs):
    all_classes = set()
    for root_dir in dirs:
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                all_classes.add(class_name)
    return sorted(all_classes)


def get_dataset_classidx(
    root,
    all_classes,
    class_to_idx,
    transform,
    target_transform=None
):
    if target_transform is not None:
        data = ImageFolder(
            root,
            transform=transform,
            target_transform=target_transform
        )
    else:
        data = ImageFolder(
            root,
            transform=transform,
        )

    old_classes = data.classes

    data.class_to_idx = class_to_idx
    data.classes = all_classes

    # need to reset samples and targets
    data.samples = [
        (path, data.class_to_idx[
                old_classes[target]
            ]) for path, target in data.samples]
    data.targets = [s[1] for s in data.samples]

    # PYTHON YOU ARE STUPIDSJFHKLSJAFOUJHDSKl
    return data


def get_dataset(args, transform_train, transform_test):
    # train = iNatDataset(
    #     args,
    #     train=True,
    #     transform=transform_train,
    #     target_transform=transform_test,
    #     download=args.download
    # )
    if args.dataset == "inat100k":
        train_dir = os.path.join(args.root, "train")
        val_dir = os.path.join(args.root, "val")
        test_dir = os.path.join(args.root, "test")

        all_classes = get_all_classes(train_dir, val_dir, test_dir)

        args.num_classes = len(all_classes)

        # Create a universal class_to_idx mapping
        class_to_idx = {class_name: idx for idx,
                        class_name in enumerate(all_classes)}

        # train = get_dataset_classidx(
        #     train_dir,
        #     all_classes,
        #     class_to_idx,
        #     transform_train,
        #     # target_transform=transform_test
        # )
        train = inat_dataset(
            args.root,
            transform_train,
            transform_test,
            "train",
            args.num_classes,
            class_to_idx
        )
        # args.num_classes = len(train.classes)

        # val = get_dataset_classidx(
        #     val_dir,
        #     all_classes,
        #     class_to_idx,
        #     transform_test,
        #     # target_transform=transform_test
        # )
        val = inat_dataset(
            args.root,
            transform_test,
            transform_test,
            "val",
            args.num_classes,
            class_to_idx
        )

        # test = get_dataset_classidx(
        #     test_dir,
        #     all_classes,
        #     class_to_idx,
        #     transform_test
        # )
        test = inat_dataset(
            args.root,
            transform_test,
            transform_test,
            "test",
            args.num_classes,
            class_to_idx
        )

    else:
        # ???
        print(args.dataset + " is not yet supported as a dataset")
        raise Exception(args.dataset + " is not a dataset")

    return train, val, test
    # train_dataset = webvision_dataset(root_dir=args.trainval_root, transform=transform_train, target_transform=transform_test, mode='all', num_class=50)

    # #################################### Test set #############################################
    # test_dataset_1 = webvision_dataset(root_dir=args.trainval_root, transform=transform_test, target_transform=transform_test, mode='test', num_class=50)

    # test_dataset_2 = imagenet_dataset(root_dir=args.val_root, web_root=args.trainval_root, transform=transform_test,  num_class=50)

    # return train_dataset, test_dataset_1, test_dataset_2


class inat_dataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform,
        target_transform,
        mode,
        num_class,
        class_to_idx,
        num_samples=None,
        pred=[],
        probability=[],
        paths=[],
        log=''
    ):
        self.root = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.class_to_idx = class_to_idx

        # assume mode is in ["train", "val", "test"]
        data_root = os.path.join(root_dir, mode)

        if mode == "train":
            self.train_labels = []
            self.targets = []
            self.train_imgs = []
            for root, dirs, files in os.walk(data_root, followlinks=True):
                if root == data_root:
                    continue

                target_class = root.split("/")[-1]  # parent dir == class
                class_idx = class_to_idx[target_class]  # consistent indicies

                for file in files:
                    self.train_imgs.append(os.path.join(target_class, file))
                    self.train_labels.append(class_idx)
                    self.targets.append(class_idx)

            self.targets = np.array(self.targets)
            self.train_imgs = np.array(self.train_imgs)
            self.train_labels = np.array(self.train_labels)

            print("train targets length: " + str(len(self.targets)))
            print("train imgs length: " + str(len(self.train_imgs)))
            print("train labels length: " + str(len(self.train_labels)))

        else:
            self.val_labels = {}
            self.val_imgs = []
            for root, dirs, files in os.walk(data_root, followlinks=True):
                if root == data_root:
                    continue

                target_class = root.split("/")[-1]  # parent dir == class
                class_idx = class_to_idx[target_class]  # consistent indicies

                for file in files:
                    img = os.path.join(target_class, file)
                    self.val_imgs.append(img)
                    self.val_labels[img] = class_idx

            print(self.mode + " imgs length: " + str(len(self.val_imgs)))

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.train_imgs[index]
            target = self.train_labels[index]
            image = Image.open(os.path.join(self.root, self.mode, img_path)).convert('RGB')
            img1 = self.transform(image)
            return img1, target, index
        else:
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(os.path.join(self.root, self.mode, img_path)).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)


class webvision_dataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform,
        target_transform,
        mode,
        num_class,
        num_samples=None,
        pred=[],
        probability=[],
        paths=[],
        log=''
    ):
        self.root = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        if self.mode == 'test':
            with open(os.path.join(self.root, 'info/val_filelist.txt')) as f:
                lines = f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img] = target
        else:
            with open(os.path.join(self.root, 'info/train_filelist_google.txt')) as f:
                lines = f.readlines()
            if num_class == 1000:
                with open(os.path.join(self.root, 'info/train_filelist_flickr.txt')) as f:
                    lines += f.readlines()
            train_imgs = []
            self.train_labels = []
            self.targets = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    train_imgs.append(img)
                    self.train_labels.append(target)
                    self.targets.append(target)
            self.targets=np.array(self.targets)
            if self.mode == 'all':
                if num_samples is not None:
                    self.train_imgs = sample_traning_set(train_imgs, self.train_labels, num_class, num_samples)
                else:
                    self.train_imgs = train_imgs
            self.train_imgs = np.array(self.train_imgs)
            self.train_labels = np.array(self.train_labels)
    def __getitem__(self, index):
        if  self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.train_labels[index]
            image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            img1 = self.transform(image)
            return img1, target, index
        elif self.mode == 'test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(os.path.join(self.root, 'val_images_256/', img_path)).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)


class imagenet_dataset(Dataset):
    def __init__(self, root_dir, web_root, transform, num_class):
        self.root = root_dir
        self.transform = transform
        self.val_data = []
        with open(os.path.join(web_root, 'info/synsets.txt')) as f:
            lines = f.readlines()
        synsets = [x.split()[0] for x in lines]
        for c in range(num_class):
            class_path = os.path.join(self.root, synsets[c])
            imgs = os.listdir(class_path)
            for img in imgs:
                self.val_data.append([c, os.path.join(class_path, img)])

    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.val_data)

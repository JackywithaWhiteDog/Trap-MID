import os
import queue
import shutil
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision import transforms, datasets
from PIL import Image
from argparse import ArgumentParser

from models.classifiers import *

parser = ArgumentParser(description='Reclassify the public dataset with the target model')
parser.add_argument('--model', default='VGG16', help='VGG16 | IR152 | FaceNet64')
parser.add_argument('--data_name', type=str, default='celeba', help='celeba | ffhq | facescrub | mnist')
parser.add_argument('--top_n', type=int, help='the n of top-n selection strategy.')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--save_root', type=str, default='reclassified_public_data')

parser.add_argument('--img_root', type=str, default='datasets/celeba/img_align_celeba')
parser.add_argument('--ckpt_file', type=str)

args = parser.parse_args()


class PublicFFHQ(torch.utils.data.Dataset):
    def __init__(self, root='datasets/ffhq/thumbnails128x128/', transform=None):
        super(PublicFFHQ, self).__init__()
        self.root = root
        self.transform = transform
        self.images = []
        self.path = self.root

        num_classes = len([lists for lists in os.listdir(
            self.path) if os.path.isdir(os.path.join(self.path, lists))])

        for idx in range(num_classes):
            class_path = os.path.join(self.path, str(idx * 1000).zfill(5))
            for _, _, files in os.walk(class_path):
                for img_name in files:
                    self.images.append(os.path.join(class_path, img_name))

    def __getitem__(self, index):

        img_path = self.images[index]
        # print(img_path)
        img = Image.open(img_path)
        if self.transform != None:
            img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.images)


class PublicCeleba(torch.utils.data.Dataset):
    def __init__(self, file_path='data_files/celeba_ganset.txt',
                 img_root='datasets/celeba/img_align_celeba', transform=None):
        super(PublicCeleba, self).__init__()
        self.file_path = file_path
        self.img_root = img_root
        self.transform = transform
        self.images = []

        name_list, label_list = [], []

        f = open(self.file_path, "r")
        for line in f.readlines():
            img_name = line.strip()
            self.images.append(os.path.join(self.img_root, img_name))

    def __getitem__(self, index):

        img_path = self.images[index]
        img = Image.open(img_path)
        if self.transform != None:
            img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.images)


class PublicFaceScrub(torch.utils.data.Dataset):
    def __init__(self, file_path='data_files/facescrub_ganset.txt',
                 img_root='datasets/facescrub', transform=None):
        super(PublicFaceScrub, self).__init__()
        self.file_path = file_path
        self.img_root = img_root
        self.transform = transform
        self.images = []

        name_list, label_list = [], []

        f = open(self.file_path, "r")
        for line in f.readlines():
            img_name = line.strip()
            img_path = os.path.join(self.img_root, img_name)
            try:
                if img_path.endswith(".png") or img_path.endswith(".jpg"):
                    img = Image.open(img_path)
                    if img.size != (64, 64):
                        img = img.resize((64, 64), Image.ANTIALIAS)
                    img = img.convert('RGB')
                    self.images.append((img, img_path))
            except:
                continue

    def __getitem__(self, index):

        img, img_path = self.images[index]
        if self.transform != None:
            img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.images)


def top_n_selection(args, T, data_loader):
    """
    Top-n selection strategy.
    :param args: top-n, save_path
    :param T: target model
    :param data_loader: dataloader of
    :return:
    """
    print("=> start inference ...")
    all_images_prob = None
    all_images_path = None
    all_images = None
    # get the predict confidence of each image in the public data
    with torch.no_grad():
        for i, (images, img_path) in enumerate(data_loader):
            bs = images.shape[0]
            images = images.cuda()
            logits = T(images)[-1]
            prob = F.softmax(logits, dim=1)  # (bs, 1000)
            prob = prob.cpu()
            if i == 0:
                all_images_prob = prob
                if args.data_name == 'mnist':
                    all_images = transforms.Resize((28, 28))(images)
                else:
                    all_images_path = img_path
            else:
                all_images_prob = torch.cat([all_images_prob, prob], dim=0)
                if args.data_name == 'mnist':
                    all_images = torch.cat([all_images, transforms.Resize((28, 28))(images)], dim=0)
                else:
                    all_images_path = all_images_path + img_path

    print("=> start reclassify ...")
    save_path = args.save_root
    print(" top_n: ", args.top_n)
    print(" save_path: ", save_path)
    # top-n selection
    for class_idx in range(args.num_classes):
        bs = all_images_prob.shape[0]
        ccc = 0
        # maintain a priority queue
        q = queue.PriorityQueue()
        class_idx_prob = all_images_prob[:, class_idx]

        for j in range(bs):
            current_value = float(class_idx_prob[j])
            # Maintain a priority queue with confidence as the priority
            if q.qsize() < args.top_n:
                q.put([current_value, j])
            else:
                current_min = q.get()
                if current_value < current_min[0]:
                    q.put(current_min)
                else:
                    q.put([current_value, j])
        # reclassify and move the images
        for m in range(q.qsize()):
            q_value = q.get()
            q_prob = round(q_value[0], 4)

            ori_save_path = os.path.join(save_path, str(class_idx))
            if not os.path.exists(ori_save_path):
                os.makedirs(ori_save_path)

            new_image_path = os.path.join(ori_save_path, str(ccc) + '_' + str(q_prob) + '.png')

            if all_images_path:
                q_image_path = all_images_path[q_value[1]]
                shutil.copy(q_image_path, new_image_path)
            else:
                torchvision.utils.save_image(
                    all_images[q_value[1]],
                    new_image_path,
                    normalize=True,
                    padding=0
                )
            ccc += 1


print(args)
assert not os.path.exists(args.save_root)
print("=> load target model ...")

model_name_T = args.model
if model_name_T == "VGG16":
    T = VGG16(args.num_classes)
elif model_name_T == 'IR152':
    T = IR152(args.num_classes)
elif model_name_T == "FaceNet64":
    T = FaceNet64(args.num_classes)
elif model_name_T == "IR18":
    T = IR18(args.num_classes)
elif model_name_T == 'VGG16_BiDO':
    T = VGG16_BiDO(args.num_classes)
else:
    raise NotImplementedError(f'Model {model_name_T} not implemented.')
path_T = os.path.join(args.ckpt_file)
T = torch.nn.DataParallel(T).cuda()
ckp_T = torch.load(path_T)
T.load_state_dict(ckp_T['state_dict'], strict=False)
T.eval()

print("=> load public dataset ...")
if args.data_name == 'celeba':
    re_size = 64
    crop_size = 108
    offset_height = (218 - crop_size) // 2
    offset_width = (178 - crop_size) // 2
    crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
    celeba_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(crop),
        transforms.ToPILImage(),
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor()
    ])
    data_set = PublicCeleba(file_path='data_files/celeba_ganset.txt',
                            img_root=args.img_root,
                            transform=celeba_transform)
    data_loader = data.DataLoader(data_set, batch_size=350)
elif args.data_name == 'ffhq':
    re_size = 64
    crop_size = 88
    offset_height = (128 - crop_size) // 2
    offset_width = (128 - crop_size) // 2
    crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
    ffhq_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(crop),
        transforms.ToPILImage(),
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor()
    ])
    data_set = PublicFFHQ(root=args.img_root, transform=ffhq_transform)
elif args.data_name == 'facescrub':
    crop_size = 54
    offset_height = (64 - crop_size) // 2
    offset_width = (64 - crop_size) // 2
    re_size = 64
    crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

    faceScrub_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(crop),
        transforms.ToPILImage(),
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor()
    ])
    data_set = PublicFaceScrub(file_path='data_files/facescrub_ganset.txt',
                               img_root='datasets/facescrub', transform=faceScrub_transform)
elif args.data_name == 'mnist':
    # Expand chennel from 1 to 3 to fit pretrained models
    re_size = 64
    raw_data = datasets.MNIST(
        root=args.img_root,
        train=True,
        transform=transforms.Compose([
            transforms.Resize((re_size, re_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
        ])
    )
    # Take samples with label 5, 6, 7, 8, 9 as the private data
    indices = torch.where(raw_data.targets >= 5)[0]
    data_set = torch.utils.data.Subset(raw_data, indices)
data_loader = data.DataLoader(data_set, batch_size=350)
print(f'=> Dataset size: {len(data_set)}')
top_n_selection(args, T, data_loader)

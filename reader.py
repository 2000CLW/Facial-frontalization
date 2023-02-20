import os
import numpy as np
from skimage import transform
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from PIL import Image
def one_hot(label,depth):
    ones = torch.sparse.torch.eye(depth)
    return ones.index_select(0,label)


class FaceIdPoseDataset(Dataset):
    #  assume images  as B x C x H x W  numpy array
    def __init__(self, root, transform=None):
        super().__init__()
        with open(root,'r') as f:
            data = ['./dataset'+ line.strip() for line in f.readlines()]
        self.data = np.random.permutation(data)
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = Image.open(sample)
        img = img.convert('RGB')
        if sample.find('frontal') !=-1:
            pose_label = 1
        else :
            pose_label = 0
        id_label = int(sample[22:25])
        img= self.transform(img)
        return [img.float(), id_label, pose_label]

    def __len__(self):
        return len(self.data)


def get_batch(root,batch_size):
    data_set = FaceIdPoseDataset(root,
                                 transform=transforms.Compose([
                                     transforms.Resize((110,110)),
                                     transforms.RandomCrop((96,96)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                 ]))
    dataloader = DataLoader(data_set,batch_size=batch_size,
                            shuffle=True,drop_last=True)  #drop_last is necessary,because last iteration may fail
    return dataloader


if __name__=='__main__':
    data_set = get_batch('dataset/data_list.txt',4)
    for i, data_batch in enumerate(data_set):
        print("i:",i)
        img = data_batch[0] # torch.Size([batch_size, 3, 96, 96])
        id = data_batch[1] # id: tensor([470])
        pose = data_batch[2] # pose: tensor([1])或者 pose: tensor([0])
        #print(pose.data.numpy())
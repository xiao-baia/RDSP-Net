import torchvision.transforms as transforms
import os
from PIL import Image

#test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = image_root
        self.gts = gt_root
        self.depths = depth_root
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])


    def load_data(self):
        image = self.rgb_loader(self.images)
        image_pil = image
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts)
        gt_pil = gt

        depth = self.binary_loader(self.depths)
        depth_pil = depth
        depth = self.depths_transform(depth).unsqueeze(0)

        return image_pil, depth_pil, gt_pil, image, gt, depth

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


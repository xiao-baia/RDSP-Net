import torchvision.transforms as transforms


class test_dataset:
    def __init__(self, image, depth, testsize):
        self.testsize = testsize
        self.images = image
        self.depths = depth
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_shape = None
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
        ])

    def load_data(self):
        image = self.transform(self.images).unsqueeze(0)
        gt_shape = self.depths
        depth = self.depths_transform(self.depths).unsqueeze(0)
        return image, gt_shape, depth,

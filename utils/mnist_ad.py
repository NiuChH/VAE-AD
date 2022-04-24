import torch
import torchvision
from PIL import Image


def process_img(img_t, mask_t, transform):
    img = Image.fromarray(img_t.numpy(), mode='L')
    return transform(img), mask_t


class MNIST_AD:
    def __init__(self, ds_config):
        batch_size = ds_config.batch_size
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
        if ds_config.load_train:
            train_ds = torchvision.datasets.MNIST(ds_config.root, train=True, download=True)
            train_normal = train_ds.data[train_ds.targets == ds_config.product]
            mask_zero = torch.zeros(train_normal.shape[1:]).unsqueeze(0)
            data = [process_img(train_normal[i], mask_zero, transform) for i in range(train_normal.shape[0])]
            if ds_config.train_size_ratio < 1:
                data = data[:round(len(data)*ds_config.train_size_ratio)]
            self.train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        if ds_config.load_test:
            test_ds = torchvision.datasets.MNIST(ds_config.root, train=False, download=True)
            test_normal = test_ds.data[test_ds.targets == ds_config.product]
            mask_zero = torch.zeros(test_normal.shape[1:]).unsqueeze(0)
            norm_data = [process_img(test_normal[i], mask_zero, transform) for i in range(test_normal.shape[0])]
            self.test_norm_loader = torch.utils.data.DataLoader(norm_data, batch_size=batch_size, shuffle=False)

            test_anom = test_ds.data[test_ds.targets != ds_config.product]
            mask_one = torch.ones(test_anom.shape[1:]).unsqueeze(0)
            anom_data = [process_img(test_anom[i], mask_one, transform) for i in range(test_anom.shape[0])]
            self.test_anom_loader = torch.utils.data.DataLoader(anom_data, batch_size=batch_size, shuffle=False)

            # self.validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

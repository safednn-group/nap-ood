import torch
import torchvision.transforms as transforms
from datasets import SubDataset, AbstractDomainInterface, ExpandRGBChannels
from torchvision import datasets


class GTSRB(AbstractDomainInterface):
    """
        MNIST: 60,000 train + 10,000 test.
        Ds: (50,000 train + 10,000 valid) + (10,000 test)
        Dv, Dt: 60,000 valid + 10,000 test.
    """

    def __init__(self):
        super(GTSRB, self).__init__()

        im_transformer = transforms.Compose([
            transforms.ToTensor(),
            # Change the image to PIL format, such that resize can be done
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            # Bring it back to tensor
            transforms.ToTensor()
        ])
        root_path = './workspace/datasets/gtsrb'

        train_indices = torch.randperm(26640).int()# torch.arange(0, 26640).int()
        test_indices = torch.randperm(12569).int()
        # in training  26640
        self.D1_train_ind = train_indices[torch.arange(0, 22200).long()]
        self.D1_valid_ind = train_indices[torch.arange(22200, 26640).long()]
        self.D1_test_ind = train_indices[torch.arange(0, 4440).long()]

        self.D2_valid_ind = test_indices[torch.arange(0, 12569).long()]
        self.D2_test_ind = test_indices[torch.arange(0, 2095).long()]

        # self.ds_train = datasets.MNIST(root_path,
        #                                train=True,
        #                                transform=im_transformer,
        #                                download=True)
        #self.ds_train = datasets.ImageFolder(root='/home/kradlak/source_code/od-test/data/GTSRB-Training_fixed/GTSRB/Training', transform=im_transformer)
        self.ds_train = datasets.ImageFolder(
            root='data/GTSRB-Training_fixed/GTSRB/Training', transform=im_transformer)
        # in testing 12569
        # self.ds_test = datasets.MNIST(root_path,
        #                               train=False,
        #                               transform=im_transformer,
        #                               download=True)
        #self.ds_test = datasets.ImageFolder(root='/home/kradlak/source_code/od-test/data/GTSRB_Online-Test-Images-Sorted/GTSRB/Online-Test-sort', transform=im_transformer)
        self.ds_test = datasets.ImageFolder(
            root='data/GTSRB_Online-Test-Images-Sorted/GTSRB/Online-Test-sort',
            transform=im_transformer)


    def get_D1_train(self):
        return SubDataset(self.name, self.ds_train, self.D1_train_ind)

    def get_D1_valid(self):
        return SubDataset(self.name, self.ds_train, self.D1_valid_ind, label=0)

    def get_D1_test(self):
        return SubDataset(self.name, self.ds_test, self.D1_test_ind, label=0)

    def get_D2_valid(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.ds_train, self.D2_valid_ind, label=1, transform=D1.conformity_transform())

    def get_D2_test(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.ds_test, self.D2_test_ind, label=1, transform=D1.conformity_transform())

    def conformity_transform(self):
        return transforms.Compose([ExpandRGBChannels(),
                                   transforms.ToPILImage(),
                                   transforms.Resize((32, 32)),
                                   transforms.ToTensor(),
                                   ])

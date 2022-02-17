from __future__ import print_function

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from utils.iterative_trainer import IterativeTrainerConfig
from utils.logger import Logger
from termcolor import colored

from methods import AbstractModelWrapper, SVMLoss
from methods.base_threshold import ProbabilityThreshold
from datasets import MirroredDataset
import global_vars as Global

class ScoreSVMModelWrapper(AbstractModelWrapper):
    """ The wrapper class for H.
    """
    def __init__(self, base_model):
        super(ScoreSVMModelWrapper, self).__init__(base_model)

        output_size = base_model.output_size()[1].item()
        self.H = nn.Sequential(
                    nn.BatchNorm1d(output_size), # Helps with faster convergence.
                    nn.Linear(output_size, 1),
        )

    def subnetwork_eval(self, x):
        base_output = self.base_model.forward(x, softmax=False) # We want the logits.
        output = base_output.detach()
        return output

    def wrapper_eval(self, x):
        output = self.H(x)
        return output
    
    def classify(self, x):
        return (x>0).long()

class ScoreSVM(ProbabilityThreshold):
    def method_identifier(self):
        output = "ScoreSVM"
        if len(self.add_identifier) > 0:
            output = output + "/" + self.add_identifier
        return output
    
    def get_H_config(self, dataset, will_train=True):
        print("Preparing training D1+D2 (H)")
        print("Mixture size: %s"%colored('%d'%len(dataset), 'green'))

        # 80%, 20% for local train+test
        train_ds, valid_ds = dataset.split_dataset(0.8)
        self.train_dataset_name = self.args.D1
        if self.args.D1 in Global.mirror_augment:
            print(colored("Mirror augmenting %s"%self.args.D1, 'green'))
            new_train_ds = train_ds + MirroredDataset(train_ds)
            train_ds = new_train_ds

        # Initialize the multi-threaded loaders.
        train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers, pin_memory=True)

        # Set up the criterion
        # margin must be non-zero.
        criterion = SVMLoss(margin=1.0).cuda()

        # Set up the model
        model = ScoreSVMModelWrapper(self.base_model).cuda()

        old_valid_loader = valid_loader
        if will_train:
            # cache the subnetwork for faster optimization.
            from methods import get_cached
            from torch.utils.data.dataset import TensorDataset

            trainX, trainY = get_cached(model, train_loader, self.args.device)
            validX, validY = get_cached(model, valid_loader, self.args.device)

            new_train_ds = TensorDataset(trainX, trainY)
            new_valid_ds = TensorDataset(validX, validY)

            # Initialize the new multi-threaded loaders.
            train_loader = DataLoader(new_train_ds, batch_size=2048, shuffle=True, num_workers=0, pin_memory=False)
            valid_loader = DataLoader(new_valid_ds, batch_size=2048, shuffle=True, num_workers=0, pin_memory=False)

            # Set model to direct evaluation (for cached data)
            model.set_eval_direct(True)

        # Set up the config
        config = IterativeTrainerConfig()

        base_model_name = self.base_model.__class__.__name__
        if hasattr(self.base_model, 'preferred_name'):
            base_model_name = self.base_model.preferred_name()

        config.name = '_%s[%s](%s->%s)'%(self.__class__.__name__, base_model_name, self.args.D1, self.args.D2)
        config.train_loader = train_loader
        config.valid_loader = valid_loader
        config.phases = {
                        'train':   {'dataset' : train_loader,  'backward': True},
                        'test':    {'dataset' : valid_loader,  'backward': False},
                        'testU':   {'dataset' : old_valid_loader, 'backward': False},                                                
                        }
        config.criterion = criterion
        config.classification = True
        config.cast_float_label = True
        config.stochastic_gradient = True  
        config.visualize = not self.args.no_visualize  
        config.model = model
        config.optim = optim.Adagrad(model.H.parameters(), lr=1e-1, weight_decay=1.0/len(train_ds))
        config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(config.optim, patience=10, threshold=1e-1, min_lr=1e-8, factor=0.1, verbose=True)
        config.logger = Logger()
        config.max_epoch = 100
        self.model_name = "VGG" if self.add_identifier.find("VGG") >= 0 else ("Resnet" if self.add_identifier.find("Resnet") >= 0 else "")
        self.add_identifier = ""
        return config

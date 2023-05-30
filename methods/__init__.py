import torch
import torch.nn as nn
import numpy as np
import tqdm
import time
from termcolor import colored
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve


class AbstractMethodInterface(object):
    def __init__(self):
        self.base_model = None
        self.unknown_loader = None
        self.args = None
        self.known_loader = None
        self.name = self.__class__.__name__

    def propose_H(self, dataset):
        raise NotImplementedError("%s does not have implementations for this" % (self.name))

    def train_H(self, dataset):
        raise NotImplementedError("%s does not have implementations for this" % (self.name))

    def test_H(self, dataset):
        self.base_model.eval()

        with tqdm.tqdm(total=len(dataset)) as pbar:
            with torch.no_grad():

                correct = 0.0
                all_probs = np.array([])
                labels = np.array([])
                dataset_iter = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False,
                                          num_workers=self.args.workers, pin_memory=True)
                counter = 0
                for i, (image, label) in enumerate(dataset_iter):
                    pbar.update()
                    counter += 1
                    # Get and prepare data.
                    input, target = image.to(self.args.device), label.to(self.args.device)
                    scores = self.get_ood_score(input)
                    if self.inverse:
                        classification = np.where(scores < self.threshold, 1, 0)
                    else:
                        classification = np.where(scores > self.threshold, 1, 0)
                    correct += (classification == label.numpy()).sum()
                    if all_probs.size:
                        labels = np.concatenate((labels, label))
                        all_probs = np.concatenate((all_probs, scores))
                    else:
                        labels = label
                        all_probs = scores

                auroc = roc_auc_score(labels, all_probs)
                p, r, _ = precision_recall_curve(labels, all_probs)
                aupr = auc(r, p)
                print("Final Test average accuracy %s" % (
                    colored('%.4f%%' % (correct / labels.shape[0] * 100), 'red')))
        return correct / labels.shape[0], auroc, aupr

    def method_identifier(self):
        raise NotImplementedError("Please implement the identifier method for %s" % (self.name))

    def _find_threshold(self):
        scores_known = np.array([])
        scores_unknown = np.array([])
        with torch.no_grad():
            for i, (image, label) in enumerate(self.known_loader):

                # Get and prepare data.
                input, target = image.to(self.args.device), label.to(self.args.device)
                scores = self.get_ood_score(input)
                if scores_known.size:
                    scores_known = np.concatenate((scores_known, scores))
                else:
                    scores_known = scores

            for i, (image, label) in enumerate(self.unknown_loader):
                # Get and prepare data.
                input, target = image.to(self.args.device), label.to(self.args.device)
                scores = self.get_ood_score(input)
                if scores_unknown.size:
                    scores_unknown = np.concatenate((scores_unknown, scores))
                else:
                    scores_unknown = scores

        if scores_unknown.mean() < scores_known.mean():
            self.inverse = True
            cut_threshold = np.quantile(scores_known, .05)
        else:
            self.inverse = False
            cut_threshold = np.quantile(scores_known, .95)
        min = np.max([scores_unknown.min(), scores_known.min()])
        max = np.min([scores_unknown.max(), scores_known.max()])
        if self.inverse:
            cut_correct_count = (scores_unknown < cut_threshold).sum()
            cut_correct_count += (scores_known >= cut_threshold).sum()
        else:
            cut_correct_count = (scores_unknown > cut_threshold).sum()
            cut_correct_count += (scores_known <= cut_threshold).sum()
        best_correct_count = 0
        best_threshold = 0
        for i in np.linspace(min, max, num=1000):
            correct_count = 0
            if self.inverse:
                correct_count += (scores_unknown < i).sum()
                correct_count += (scores_known >= i).sum()
            else:
                correct_count += (scores_unknown > i).sum()
                correct_count += (scores_known <= i).sum()
            if best_correct_count < correct_count:
                best_correct_count = correct_count
                best_threshold = i
        if self.inverse:
            if best_threshold < cut_threshold:
                best_correct_count = cut_correct_count
                best_threshold = cut_threshold
        else:
            if best_threshold > cut_threshold:
                best_correct_count = cut_correct_count
                best_threshold = cut_threshold
        self.threshold = best_threshold
        acc = best_correct_count / (scores_known.shape[0] * 2)
        return acc

    def _generate_execution_times(self, loader):
        assert self.args.batch_size == 1
        n_times = 1000
        exec_times = np.ones(n_times)

        trainiter = iter(loader)
        x = trainiter.__next__()[0][0].unsqueeze(0).to(self.args.device)
        with torch.no_grad():
            for i in range(n_times):
                start_time = time.time()
                scores = self.get_ood_score(x)

                _ = np.where(scores > self.threshold, 1, 0)
                exec_times[i] = time.time() - start_time

        exec_times = exec_times.mean()
        print(exec_times)

    def get_ood_score(self, input):
        pass


class AbstractModelWrapper(nn.Module):
    def __init__(self, base_model):
        super(AbstractModelWrapper, self).__init__()
        self.base_model = base_model
        if hasattr(self.base_model, 'eval'):
            self.base_model.eval()
        if hasattr(self.base_model, 'parameters'):
            for parameter in self.base_model.parameters():
                parameter.requires_grad = False

        self.eval_direct = False
        self.cache = {}  # Be careful what you cache! You wouldn't have infinite memory.

    def set_eval_direct(self, eval_direct):
        self.eval_direct = eval_direct

    def train(self, mode=True):
        """ Must override the train mode 
        because the base_model is always in eval mode.
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        # Now revert back the base_model to eval.
        if hasattr(self.base_model, 'eval'):
            self.base_model.eval()
        return self

    def subnetwork_eval(self, x):
        raise NotImplementedError

    def wrapper_eval(self, x):
        raise NotImplementedError

    def subnetwork_cached_eval(self, x, indices, group):
        output = None
        cache = None

        if group in self.cache:
            cache = self.cache[group]
        else:
            cache = {}

        all_indices = [ind in cache for ind in indices]
        if torch.ByteTensor(all_indices).all():
            # Then fetch from the cache.
            all_outputs = [cache[ind] for ind in indices]
            output = torch.cat(all_outputs)
        else:
            output = self.subnetwork_eval(x)
            for i, entry in enumerate(output):
                cache[indices[i]] = entry.unsqueeze_(0)

        self.cache[group] = cache
        return output

    def forward(self, x, indices=None, group=None):
        input = None

        if not self.eval_direct:
            if indices is None:
                input = self.subnetwork_eval(x)
            else:
                input = self.subnetwork_cached_eval(x, indices=indices, group=group)
            input = input.detach()
            input.requires_grad = False
        else:
            input = x

        output = self.wrapper_eval(input)
        return output


class SVMLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(SVMLoss, self).__init__()
        self.margin = margin
        self.size_average = True
        self.reduction = "mean"

    def forward(self, x, target):
        target = target.clone()
        # 0 labels should be set to -1 for this loss.
        target.data[target.data < 0.1] = -1
        error = self.margin - x * target
        loss = torch.clamp(error, min=0)
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def get_cached(model, dataset_loader, device):
    from tqdm import tqdm

    outputX, outputY = [], []
    with torch.no_grad():
        with tqdm(total=len(dataset_loader)) as pbar:
            pbar.set_description('Caching data')
            for i, (image, label) in enumerate(dataset_loader):
                pbar.update()
                input, target = image.to(device), label.to(device)
                new_input = model.subnetwork_eval(input)
                outputX.append(new_input)
                outputY.append(target)
    return torch.cat(outputX, 0), torch.cat(outputY, 0)

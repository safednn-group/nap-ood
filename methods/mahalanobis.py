from __future__ import print_function
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
import global_vars as Global
from datasets import MirroredDataset
from utils.iterative_trainer import IterativeTrainerConfig
from utils.logger import Logger
from sklearn.linear_model import LogisticRegressionCV
import time
import torch
from termcolor import colored
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from methods import AbstractMethodInterface


class Mahalanobis(AbstractMethodInterface):
    def __init__(self, args):
        super(Mahalanobis, self).__init__()
        self.base_model = None
        self.H_class = None
        self.args = args
        self.class_count = 0
        self.default_model = 0
        self.add_identifier = ""
        self.known_loader = None
        self.unknown_loader = None
        self.train_loader = None
        self.train_dataset_name = ""
        self.valid_dataset_name = ""
        self.test_dataset_name = ""
        self.train_dataset_length = 0
        self.model_name = ""

    def propose_H(self, dataset, mirror=True):
        config = self.get_H_config(dataset, mirror)

        from models import get_ref_model_path
        h_path = get_ref_model_path(self.args, config.model.__class__.__name__, dataset.name)
        best_h_path = os.path.join(h_path, 'model.best.pth')

        # trainer = IterativeTrainer(config, self.args)

        if not os.path.isfile(best_h_path):
            raise NotImplementedError("Please use model_setup to pretrain the networks first!")
        else:
            print(colored('Loading H1 model from %s' % best_h_path, 'red'))
            config.model.load_state_dict(torch.load(best_h_path))

        self.base_model = config.model
        self.base_model.eval()
        self.class_count = self.base_model.output_size()[1].item()
        self.add_identifier = self.base_model.__class__.__name__
        self.train_dataset_name = dataset.name
        self.model_name = "VGG" if self.add_identifier.find("VGG") >= 0 else "Resnet"
        if hasattr(self.base_model, 'preferred_name'):
            self.add_identifier = self.base_model.preferred_name()

    def method_identifier(self):
        output = "Mahalanobis"
        # if len(self.add_identifier) > 0:
        #     output = output + "/" + self.add_identifier
        return output

    def get_H_config(self, dataset, mirror):
        if self.args.D1 in Global.mirror_augment and mirror:
            print(colored("Mirror augmenting %s" % self.args.D1, 'green'))
            new_train_ds = dataset + MirroredDataset(dataset)
            dataset = new_train_ds

        self.train_loader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.workers,
                                       pin_memory=True)
        self.train_dataset_length = len(dataset)
        self.input_shape = iter(dataset).__next__()[0].shape
        # Set up the model
        model = Global.get_ref_classifier(self.args.D1)[self.default_model]().to(self.args.device)
        # model.forward()

        # Set up the config
        config = IterativeTrainerConfig()

        base_model_name = self.base_model.__class__.__name__
        if hasattr(self.base_model, 'preferred_name'):
            base_model_name = self.base_model.preferred_name()

        config.name = '_%s[%s](%s->%s)' % (self.__class__.__name__, base_model_name, self.args.D1, self.args.D2)
        config.train_loader = self.train_loader
        config.visualize = not self.args.no_visualize
        config.model = model
        config.logger = Logger()
        return config

    def train_H(self, dataset):
        self.known_loader = DataLoader(dataset.datasets[0], batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=self.args.workers,
                                       pin_memory=True)
        sample_mean, precision, best_regressor, best_magnitude = self._tune_hyperparameters()
        print('saving results...')
        save_dir = os.path.join('workspace/mahalanobis/', self.train_dataset_name, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.save(os.path.join(save_dir, 'results'),
                np.array([sample_mean, precision, best_regressor.coef_, best_regressor.intercept_, best_magnitude]))

    def _tune_hyperparameters(self):
        print('Tuning hyper-parameters...')
        stypes = ['mahalanobis']

        save_dir = os.path.join('workspace/mahalanobis/', self.train_dataset_name, self.model_name, 'tmp')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # set information about feature extaction
        temp_x = torch.rand((2, ) + self.input_shape)
        temp_x = Variable(temp_x).cuda()
        temp_list = self.base_model.feature_list(temp_x)[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1

        print('get sample mean and covariance')
        sample_mean, precision = self._sample_estimator(feature_list)

        print('train logistic regression model')
        m = 500

        train_in = []
        train_in_label = []
        train_out = []

        val_in = []
        val_in_label = []
        val_out = []

        cnt = 0
        for data, target in self.known_loader:
            data = data.numpy()
            target = target.numpy()
            for x, y in zip(data, target):
                cnt += 1
                if cnt <= m:
                    train_in.append(x)
                    train_in_label.append(y)
                elif cnt <= 2 * m:
                    val_in.append(x)
                    val_in_label.append(y)

                if cnt == 2 * m:
                    break
            if cnt == 2 * m:
                break

        print('In', len(train_in), len(val_in))

        criterion = nn.CrossEntropyLoss().cuda()
        adv_noise = 0.05

        for i in range(int(m / self.args.batch_size) + 1):
            if i * self.args.batch_size >= m:
                break
            data = torch.tensor(train_in[i * self.args.batch_size:min((i + 1) * self.args.batch_size, m)])
            target = torch.tensor(train_in_label[i * self.args.batch_size:min((i + 1) * self.args.batch_size, m)])
            data = data.cuda()
            target = target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.base_model(data)

            self.base_model.zero_grad()
            inputs = Variable(data.data, requires_grad=True).cuda()
            output = self.base_model(inputs)
            loss = criterion(output, target)
            loss.backward()

            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
            adv_data = torch.clamp(adv_data, 0.0, 1.0)

            train_out.extend(adv_data.cpu().numpy())

        for i in range(int(m / self.args.batch_size) + 1):
            if i * self.args.batch_size >= m:
                break
            data = torch.tensor(val_in[i * self.args.batch_size:min((i + 1) * self.args.batch_size, m)])
            target = torch.tensor(val_in_label[i * self.args.batch_size:min((i + 1) * self.args.batch_size, m)])
            data = data.cuda()
            target = target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.base_model(data)

            self.base_model.zero_grad()
            inputs = Variable(data.data, requires_grad=True).cuda()
            output = self.base_model(inputs)
            loss = criterion(output, target)
            loss.backward()

            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
            adv_data = torch.clamp(adv_data, 0.0, 1.0)

            val_out.extend(adv_data.cpu().numpy())

        print('Out', len(train_out), len(val_out))

        train_lr_data = []
        train_lr_label = []
        train_lr_data.extend(train_in)
        train_lr_label.extend(np.zeros(m))
        train_lr_data.extend(train_out)
        train_lr_label.extend(np.ones(m))
        train_lr_data = torch.tensor(train_lr_data)
        train_lr_label = torch.tensor(train_lr_label)

        best_fpr = 1.1
        best_magnitude = 0.0

        for magnitude in [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]:
            train_lr_Mahalanobis = []
            total = 0
            for data_index in range(int(np.floor(train_lr_data.size(0) / self.args.batch_size))):
                data = train_lr_data[total: total + self.args.batch_size].cuda()
                total += self.args.batch_size
                Mahalanobis_scores = self._get_Mahalanobis_score(data, sample_mean, precision, num_output,
                                                                 magnitude)
                train_lr_Mahalanobis.extend(Mahalanobis_scores)

            train_lr_Mahalanobis = np.asarray(train_lr_Mahalanobis, dtype=np.float32)
            regressor = LogisticRegressionCV(n_jobs=-1).fit(train_lr_Mahalanobis, train_lr_label[:train_lr_Mahalanobis.shape[0]])

            print('Logistic Regressor params:', regressor.coef_, regressor.intercept_)

            t0 = time.time()
            f1 = open(os.path.join(save_dir, "confidence_mahalanobis_In.txt"), 'w')
            f2 = open(os.path.join(save_dir, "confidence_mahalanobis_Out.txt"), 'w')

            ########################################In-distribution###########################################
            print("Processing in-distribution images")

            count = 0
            for i in range(int(m / self.args.batch_size) + 1):
                if i * self.args.batch_size >= m:
                    break
                images = torch.tensor(val_in[i * self.args.batch_size: min((i + 1) * self.args.batch_size, m)]).cuda()
                # if j<1000: continue
                batch_size = images.shape[0]
                Mahalanobis_scores = self._get_Mahalanobis_score(images, sample_mean, precision,
                                                                 num_output,
                                                                 magnitude)
                confidence_scores = regressor.predict_proba(Mahalanobis_scores)[:, 1]

                for k in range(batch_size):
                    f1.write("{}\n".format(-confidence_scores[k]))

                count += batch_size
                print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time() - t0))
                t0 = time.time()

            ###################################Out-of-Distributions#####################################
            t0 = time.time()
            print("Processing out-of-distribution images")
            count = 0

            for i in range(int(m / self.args.batch_size) + 1):
                if i * self.args.batch_size >= m:
                    break
                images = torch.tensor(val_out[i * self.args.batch_size: min((i + 1) * self.args.batch_size, m)]).cuda()
                # if j<1000: continue
                batch_size = images.shape[0]

                Mahalanobis_scores = self._get_Mahalanobis_score(images, sample_mean, precision,
                                                                 num_output,
                                                                 magnitude)

                confidence_scores = regressor.predict_proba(Mahalanobis_scores)[:, 1]

                for k in range(batch_size):
                    f2.write("{}\n".format(-confidence_scores[k]))

                count += batch_size
                print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time() - t0))
                t0 = time.time()

            f1.close()
            f2.close()

            results = self._metric(save_dir, stypes)
            self._print_results(results, stypes)
            fpr = results['mahalanobis']['FPR']
            if fpr < best_fpr:
                best_fpr = fpr
                best_magnitude = magnitude
                best_regressor = regressor

        print('Best Logistic Regressor params:', best_regressor.coef_, best_regressor.intercept_)
        print('Best magnitude', best_magnitude)

        return sample_mean, precision, best_regressor, best_magnitude

    def test_H(self, dataset):
        self.test_dataset_name = dataset[0].datasets[1].name
        self.known_test_loader = DataLoader(dataset[1], batch_size=self.args.batch_size, shuffle=False,
                                       num_workers=self.args.workers, pin_memory=True)
        self.unknown_test_loader = DataLoader(dataset[2], batch_size=self.args.batch_size, shuffle=False,
                                         num_workers=self.args.workers, pin_memory=True)
        dataset = DataLoader(dataset[0], batch_size=self.args.batch_size, shuffle=False,
                             num_workers=self.args.workers, pin_memory=True)
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(
            os.path.join('workspace/mahalanobis', self.train_dataset_name, self.model_name, 'results.npy'),
            allow_pickle=True)
        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                                                   [0, 0, 1, 1])
        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias
        method_args = dict()
        method_args['sample_mean'] = sample_mean
        method_args['precision'] = precision
        method_args['magnitude'] = magnitude
        method_args['regressor'] = regressor

        return self._eval_ood_detector(dataset, method_args)

    def _sample_estimator(self, feature_list):
        """
        compute sample mean and precision (inverse of covariance)
        return: sample_class_mean: list of class mean
                 precision: list of precisions
        """
        import sklearn.covariance

        self.base_model.eval()
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        correct, total = 0, 0
        num_output = len(feature_list)
        num_sample_per_class = np.empty(self.class_count)
        num_sample_per_class.fill(0)
        list_features = []
        for i in range(num_output):
            temp_list = []
            for j in range(self.class_count):
                temp_list.append(0)
            list_features.append(temp_list)

        for data, target in self.train_loader:
            total += data.size(0)
            # data = data.cuda()
            data = Variable(data)
            data = data.cuda()
            output, out_features = self.base_model.feature_list(data)

            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)

            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag = pred.eq(target.cuda()).cpu()
            correct += equal_flag.sum()

            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] \
                            = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1
                num_sample_per_class[label] += 1

        sample_class_mean = []
        out_count = 0
        for num_feature in feature_list:
            temp_list = torch.Tensor(self.class_count, int(num_feature)).cuda()
            for j in range(self.class_count):
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)
            out_count += 1

        precision = []
        for k in range(num_output):
            X = 0
            for i in range(self.class_count):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

            # find inverse
            group_lasso.fit(X.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().cuda()
            precision.append(temp_precision)

        print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

        return sample_class_mean, precision

    def _get_Mahalanobis_score(self, inputs, sample_mean, precision, num_output, magnitude):
        for layer_index in range(num_output):
            data = Variable(inputs, requires_grad=True)
            data = data.cuda()

            out_features = self.base_model.intermediate_forward(data, layer_index=layer_index)
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features, 2)

            gaussian_score = 0
            for i in range(self.class_count):
                batch_sample_mean = sample_mean[layer_index][i]
                zero_f = out_features.data - batch_sample_mean
                term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
                if i == 0:
                    gaussian_score = term_gau.view(-1, 1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

            # Input_processing
            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
            zero_f = out_features - Variable(batch_sample_mean)
            pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
            loss = torch.mean(-pure_gau)
            loss.backward()

            gradient = torch.ge(data.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            tempInputs = torch.add(data.data, -magnitude, gradient)

            noise_out_features = self.base_model.intermediate_forward(Variable(tempInputs), layer_index=layer_index)
            noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            noise_out_features = torch.mean(noise_out_features, 2)
            noise_gaussian_score = 0
            for i in range(self.class_count):
                batch_sample_mean = sample_mean[layer_index][i]
                zero_f = noise_out_features.data - batch_sample_mean
                term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
                if i == 0:
                    noise_gaussian_score = term_gau.view(-1, 1)
                else:
                    noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

            noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)

            noise_gaussian_score = np.asarray(noise_gaussian_score.cpu().numpy(), dtype=np.float32)
            if layer_index == 0:
                Mahalanobis_scores = noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))
            else:
                Mahalanobis_scores = np.concatenate(
                    (Mahalanobis_scores, noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))), axis=1)

        return Mahalanobis_scores

    def _get_Mahalanobis_score_original(self, test_loader, out_flag, sample_mean, precision,
                              layer_index, magnitude):
        '''
        Compute the proposed Mahalanobis confidence score on input dataset
        return: Mahalanobis score from layer_index
        '''
        self.base_model.eval()
        Mahalanobis = []

        if out_flag == True:
            temp_file_name = '%s/confidence_Ga%s_In.txt' % ('workspace/mahalanobis', str(layer_index))
        else:
            temp_file_name = '%s/confidence_Ga%s_Out.txt' % ('workspace/mahalanobis', str(layer_index))

        g = open(temp_file_name, 'w')

        for data, target in test_loader:

            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, requires_grad=True), Variable(target)

            out_features = self.base_model.intermediate_forward(data, layer_index)
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features, 2)

            # compute Mahalanobis score
            gaussian_score = 0
            for i in range(self.class_count):
                batch_sample_mean = sample_mean[layer_index][i]
                zero_f = out_features.data - batch_sample_mean
                term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
                if i == 0:
                    gaussian_score = term_gau.view(-1, 1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

            # Input_processing
            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
            zero_f = out_features - Variable(batch_sample_mean)
            pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
            loss = torch.mean(-pure_gau)
            loss.backward()

            gradient = torch.ge(data.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
            tempInputs = torch.add(data.data, -magnitude, gradient)

            noise_out_features = self.base_model.intermediate_forward(Variable(tempInputs, volatile=True), layer_index)
            noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            noise_out_features = torch.mean(noise_out_features, 2)
            noise_gaussian_score = 0
            for i in range(self.class_count):
                batch_sample_mean = sample_mean[layer_index][i]
                zero_f = noise_out_features.data - batch_sample_mean
                term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
                if i == 0:
                    noise_gaussian_score = term_gau.view(-1, 1)
                else:
                    noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

            noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
            Mahalanobis.extend(noise_gaussian_score.cpu().numpy())

            for i in range(data.size(0)):
                g.write("{}\n".format(noise_gaussian_score[i]))
        g.close()

        return Mahalanobis

    def _eval_ood_detector(self, dataset, method_args):
        in_save_dir = os.path.join("workspace/mahalanobis", self.train_dataset_name, self.model_name, 'nat')

        if not os.path.exists(in_save_dir):
            os.makedirs(in_save_dir)

        temp_x = torch.rand((2, ) + self.input_shape)
        temp_x = Variable(temp_x).cuda()
        temp_list = self.base_model.feature_list(temp_x)[1]
        num_output = len(temp_list)
        method_args['num_output'] = num_output

        t0 = time.time()

        f1 = open(os.path.join(in_save_dir, "in_scores.txt"), 'w')
        g1 = open(os.path.join(in_save_dir, "in_labels.txt"), 'w')

        ########################################In-distribution###########################################
        print("Processing in-distribution images")

        N = len(self.known_test_loader.dataset)
        count = 0
        all_scores = np.array([])
        for j, data in enumerate(self.known_test_loader):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]

            inputs = images

            scores = self._get_score(inputs, method_args)
            if all_scores.size:
                all_scores = np.concatenate(all_scores, scores)
            else:
                all_scores = scores
            for score in scores:
                f1.write("{}\n".format(score))

            outputs = F.softmax(self.base_model(inputs)[:, :self.class_count], dim=1)
            outputs = outputs.detach().cpu().numpy()
            preds = np.argmax(outputs, axis=1)
            confs = np.max(outputs, axis=1)

            for k in range(preds.shape[0]):
                g1.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))

            count += curr_batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time() - t0))
            t0 = time.time()

        f1.close()
        g1.close()

        t0 = time.time()
        print("Processing out-of-distribution images")
        out_save_dir = os.path.join(in_save_dir, self.test_dataset_name)

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        f2 = open(os.path.join(out_save_dir, "out_scores.txt"), 'w')
        N = len(self.unknown_test_loader.dataset)
        count = 0
        for j, data in enumerate(self.unknown_test_loader):

            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]

            inputs = images

            scores = self._get_score(inputs, method_args)
            all_scores = np.concatenate(all_scores, scores)
            for score in scores:
                f2.write("{}\n".format(score))

            count += curr_batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time() - t0))
            t0 = time.time()

        f2.close()
        from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
        labels = np.zeros(all_scores.shape)
        labels[int(labels.shape[0]/2):] = 1
        auroc = roc_auc_score(labels, all_scores)
        p, r, _ = precision_recall_curve(labels, all_scores)
        aupr = auc(r, p)
        print("Final Test average accuracy %s" % (colored('%.4f%%' % (0 * 100), 'red')))
        print(f"Auroc: {auroc} aupr: {aupr}")
        return 0, auroc, aupr

    def _get_score(self, inputs, method_args):
        sample_mean = method_args['sample_mean']
        precision = method_args['precision']
        magnitude = method_args['magnitude']
        regressor = method_args['regressor']
        num_output = method_args['num_output']

        Mahalanobis_scores = self._get_Mahalanobis_score(inputs, sample_mean, precision, num_output,
                                                         magnitude)
        scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]

        return scores

    def _get_curve(self, dir_name, stypes=['MSP', 'ODIN']):
        tp, fp = dict(), dict()
        fpr_at_tpr95 = dict()
        for stype in stypes:
            known = np.loadtxt('{}/confidence_{}_In.txt'.format(dir_name, stype), delimiter='\n')
            novel = np.loadtxt('{}/confidence_{}_Out.txt'.format(dir_name, stype), delimiter='\n')
            known.sort()
            novel.sort()

            end = np.max([np.max(known), np.max(novel)])
            start = np.min([np.min(known), np.min(novel)])
            num_k = known.shape[0]
            num_n = novel.shape[0]

            threshold = known[round(0.05 * num_k)]

            tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
            fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
            tp[stype][0], fp[stype][0] = num_k, num_n
            k, n = 0, 0
            for l in range(num_k + num_n):
                if k == num_k:
                    tp[stype][l + 1:] = tp[stype][l]
                    fp[stype][l + 1:] = np.arange(fp[stype][l] - 1, -1, -1)
                    break
                elif n == num_n:
                    tp[stype][l + 1:] = np.arange(tp[stype][l] - 1, -1, -1)
                    fp[stype][l + 1:] = fp[stype][l]
                    break
                else:
                    if novel[n] < known[k]:
                        n += 1
                        tp[stype][l + 1] = tp[stype][l]
                        fp[stype][l + 1] = fp[stype][l] - 1
                    else:
                        k += 1
                        tp[stype][l + 1] = tp[stype][l] - 1
                        fp[stype][l + 1] = fp[stype][l]

            fpr_at_tpr95[stype] = np.sum(novel > threshold) / float(num_n)

        return tp, fp, fpr_at_tpr95

    def _metric(self, dir_name, stypes=['MSP', 'ODIN'], verbose=False):
        tp, fp, fpr_at_tpr95 = self._get_curve(dir_name, stypes)
        results = dict()
        mtypes = ['FPR', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']
        if verbose:
            print('      ', end='')
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('')

        for stype in stypes:
            if verbose:
                print('{stype:5s} '.format(stype=stype), end='')
            results[stype] = dict()

            # FPR
            mtype = 'FPR'
            results[stype][mtype] = fpr_at_tpr95[stype]
            if verbose:
                print(' {val:6.3f}'.format(val=100. * results[stype][mtype]), end='')

            # AUROC
            mtype = 'AUROC'
            tpr = np.concatenate([[1.], tp[stype] / tp[stype][0], [0.]])
            fpr = np.concatenate([[1.], fp[stype] / fp[stype][0], [0.]])
            results[stype][mtype] = -np.trapz(1. - fpr, tpr)
            if verbose:
                print(' {val:6.3f}'.format(val=100. * results[stype][mtype]), end='')

            # DTERR
            mtype = 'DTERR'
            results[stype][mtype] = ((tp[stype][0] - tp[stype] + fp[stype]) / (tp[stype][0] + fp[stype][0])).min()
            if verbose:
                print(' {val:6.3f}'.format(val=100. * results[stype][mtype]), end='')

            # AUIN
            mtype = 'AUIN'
            denom = tp[stype] + fp[stype]
            denom[denom == 0.] = -1.
            pin_ind = np.concatenate([[True], denom > 0., [True]])
            pin = np.concatenate([[.5], tp[stype] / denom, [0.]])
            results[stype][mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
            if verbose:
                print(' {val:6.3f}'.format(val=100. * results[stype][mtype]), end='')

            # AUOUT
            mtype = 'AUOUT'
            denom = tp[stype][0] - tp[stype] + fp[stype][0] - fp[stype]
            denom[denom == 0.] = -1.
            pout_ind = np.concatenate([[True], denom > 0., [True]])
            pout = np.concatenate([[0.], (fp[stype][0] - fp[stype]) / denom, [.5]])
            results[stype][mtype] = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])
            if verbose:
                print(' {val:6.3f}'.format(val=100. * results[stype][mtype]), end='')
                print('')

        return results

    def _print_results(self, results, stypes):
        mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

        for stype in stypes:
            print(' OOD detection method: ' + stype)
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100. * results[stype]['FPR']), end='')
            print(' {val:6.2f}'.format(val=100. * results[stype]['DTERR']), end='')
            print(' {val:6.2f}'.format(val=100. * results[stype]['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100. * results[stype]['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100. * results[stype]['AUOUT']), end='')
            print('')

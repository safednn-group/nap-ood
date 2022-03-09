from __future__ import print_function

import numpy as np
import global_vars as Global
from datasets import MirroredDataset
from utils.iterative_trainer import IterativeTrainerConfig
from utils.logger import Logger
from termcolor import colored
from torch.utils.data.dataloader import DataLoader
import torch
import os
import methods.mahalanobis.lib_generation as lib_generation
import methods.mahalanobis.lib_regression as lib_regression
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from tqdm import tqdm

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
        self.workspace_dir = "workspace/mahalanobis"

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
        self.unknown_loader = DataLoader(dataset.datasets[1], batch_size=self.args.batch_size, shuffle=False,
                                         num_workers=self.args.workers,
                                         pin_memory=True)

        self.valid_dataset_name = dataset.datasets[1].name
        self.valid_dataset_length = len(dataset.datasets[0])
        self._tune_hyperparameters()
        print('saving results...')
        save_dir = os.path.join('workspace/mahalanobis/', self.train_dataset_name, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.save(os.path.join(save_dir, 'results'),
                np.array([self.sample_mean, self.precision, self.best_lr.coef_, self.best_lr.intercept_,
                          self.best_magnitude]))

    def test_H(self, dataset):

        dataset = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False,
                             num_workers=self.args.workers, pin_memory=True)

        for i in range(self.num_output):
            M_in = lib_generation.get_Mahalanobis_score(self.base_model, dataset, self.class_count,
                                                        self.workspace_dir + "/test_H", \
                                                        True, self.sample_mean, self.precision, i, self.best_magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)

            if i == 0:
                Mahalanobis_test = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_test = np.concatenate((Mahalanobis_test, M_in.reshape((M_in.shape[0], -1))), axis=1)

            Mahalanobis_test = np.asarray(Mahalanobis_test, dtype=np.float32)

        correct = 0.0
        labels = np.zeros((Mahalanobis_test.shape[0]))
        labels[int(Mahalanobis_test.shape[0] / 2):] = 1

        y_pred = self.best_lr.predict_proba(Mahalanobis_test)[:, 1]
        classification = np.where(y_pred > self.threshold, 1, 0)
        correct += (classification == labels).sum()
        auroc = roc_auc_score(labels, y_pred)
        p, r, _ = precision_recall_curve(labels, y_pred)
        aupr = auc(r, p)
        print("Final Test average accuracy %s" % (colored('%.4f%%' % (correct / labels.shape[0] * 100), 'red')))

        return correct / labels.shape[0], auroc, aupr

    def _tune_hyperparameters(self):
        print('Tuning hyper-parameters...')

        save_dir = os.path.join('workspace/mahalanobis/', self.train_dataset_name, self.model_name, 'tmp')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # set information about feature extaction
        temp_x = torch.rand((2,) + self.input_shape)
        temp_x = Variable(temp_x).cuda()
        temp_list = self.base_model.feature_list(temp_x)[1]
        self.num_output = num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1

        self.sample_mean, self.precision = self._sample_estimator(feature_list)
        m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
        for magnitude in m_list:
            print('Noise: ' + str(magnitude))
            for i in range(num_output):
                M_in = lib_generation.get_Mahalanobis_score(self.base_model, self.known_loader, self.class_count,
                                                            self.workspace_dir, \
                                                            True, self.sample_mean, self.precision, i, magnitude)
                M_in = np.asarray(M_in, dtype=np.float32)
                if i == 0:
                    Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
                else:
                    Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)

            print('Out-distribution: ' + self.valid_dataset_name)
            for i in range(num_output):
                M_out = lib_generation.get_Mahalanobis_score(self.base_model, self.unknown_loader, self.class_count,
                                                             self.workspace_dir, \
                                                             False, self.sample_mean, self.precision, i,
                                                             magnitude)
                M_out = np.asarray(M_out, dtype=np.float32)
                if i == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
            Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out,
                                                                                            Mahalanobis_in)
            file_name = os.path.join(self.workspace_dir,
                                     'Mahalanobis_%s_%s_%s.npy' % (
                                         str(magnitude), self.train_dataset_name, self.valid_dataset_name))
            Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(file_name, Mahalanobis_data)

        score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', 'Mahalanobis_0.002',
                      'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']
        best_tnr, best_result, best_index = 0, 0, 0
        for score in score_list:
            total_X, total_Y = lib_regression.load_characteristics(score, self.train_dataset_name,
                                                                   self.valid_dataset_name, self.workspace_dir)
            X_val, Y_val, X_test, Y_test = lib_regression.block_split(total_X, total_Y, self.valid_dataset_name,
                                                                      self.valid_dataset_length)
            lr = LogisticRegressionCV(n_jobs=-1).fit(X_val, Y_val)
            results, self.threshold = lib_regression.detection_performance(lr, X_test, Y_test, self.workspace_dir)
            if best_tnr < results['TMP']['TNR']:
                best_tnr = results['TMP']['TNR']
                self.best_lr = lr
                self.best_magnitude = float(score.split("_")[1])

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

    def _generate_execution_times(self, loader):
        import time
        import numpy as np
        n_times = 1000
        exec_times = np.ones(n_times)

        trainiter = iter(loader)
        data, target = trainiter.__next__()
        data = data[0].unsqueeze(0).to(self.args.device)
        target = target[0].unsqueeze(0).to(self.args.device)
        self.base_model.eval()
        for t in range(n_times):
            start_time = time.time()
            Mahalanobis = []
            for i in range(self.num_output):
                data, target = data.cuda(), target.cuda()
                data, target = Variable(data, requires_grad=True), Variable(target)

                out_features = self.base_model.intermediate_forward(data, softmax=False, layer_index=i)
                out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
                out_features = torch.mean(out_features, 2)

                # compute Mahalanobis score
                gaussian_score = 0
                for klass in range(self.class_count):
                    batch_sample_mean = self.sample_mean[i][klass]
                    zero_f = out_features.data - batch_sample_mean
                    term_gau = -0.5 * torch.mm(torch.mm(zero_f, self.precision[i]), zero_f.t()).diag()
                    if klass == 0:
                        gaussian_score = term_gau.view(-1, 1)
                    else:
                        gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

                # Input_processing
                sample_pred = gaussian_score.max(1)[1]
                batch_sample_mean = self.sample_mean[i].index_select(0, sample_pred)
                zero_f = out_features - Variable(batch_sample_mean)
                pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(self.precision[i])), zero_f.t()).diag()
                loss = torch.mean(-pure_gau)
                loss.backward()

                gradient = torch.ge(data.grad.data, 0)
                gradient = (gradient.float() - 0.5) * 2
                tempInputs = torch.add(data.data, -self.best_magnitude, gradient)

                with torch.no_grad():
                    noise_out_features = self.base_model.intermediate_forward(Variable(tempInputs),
                                                                              softmax=False,
                                                                              layer_index=i)
                noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1),
                                                             -1)
                noise_out_features = torch.mean(noise_out_features, 2)
                noise_gaussian_score = 0
                for klass in range(self.class_count):
                    batch_sample_mean = self.sample_mean[i][klass]
                    zero_f = noise_out_features.data - batch_sample_mean
                    term_gau = -0.5 * torch.mm(torch.mm(zero_f, self.precision[i]), zero_f.t()).diag()
                    if klass == 0:
                        noise_gaussian_score = term_gau.view(-1, 1)
                    else:
                        noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

                noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
                Mahalanobis.extend(noise_gaussian_score.cpu().numpy())

            M_in = np.asarray(Mahalanobis, dtype=np.float32).reshape(1, -1)

            y_pred = self.best_lr.predict_proba(M_in)[:, 1]
            _ = np.where(y_pred > self.threshold, 1, 0)

            exec_times[t] = time.time() - start_time

        exec_times = exec_times.mean()
        print(exec_times)
        np.savez(
            "results/article_plots/execution_times/" + self.method_identifier() + "_" + self.model_name + "_" + self.train_dataset_name,
            exec_times=exec_times)

import sys
import torch
import numpy as np
from copy import deepcopy


class BaseMonitor(object):
    def __init__(self, class_count, dataset_length, layers_shapes, neurons_to_monitor):
        self.neurons_to_monitor = neurons_to_monitor
        self.layers_shapes = layers_shapes
        self.known_patterns_set = dict()

        self.known_patterns_tensor = torch.Tensor()
        self.class_count = class_count
        self.neurons_count = 0
        self.dataset_length = dataset_length
        for layer in layers_shapes:
            self.neurons_count += layer

        for i in range(class_count):
            self.known_patterns_set[i] = set()

        # self.known_patterns_numpy = dict()
        # for i in range(class_count):
        #     self.known_patterns_numpy[i] = np.array([])

    def set_neurons_to_monitor(self, neurons_to_monitor):
        self.neurons_to_monitor = neurons_to_monitor
        for klass in neurons_to_monitor:
            self.neurons_to_monitor[klass] = np.concatenate(
                (neurons_to_monitor[klass], np.arange(self.layers_shapes[0], self.neurons_count)))


class Monitor(BaseMonitor):

    def __init__(self, class_count, dataset_length, layers_shapes, neurons_to_monitor=None):
        super().__init__(class_count, dataset_length, layers_shapes, neurons_to_monitor)

    def get_comfort_level(self, neuron_values, class_id, omit, ignore_minor_values=True, monitored_class=None):

        monitored_class = monitored_class if monitored_class else class_id
        # zero = np.zeros(neuron_values.shape)
        # neuron_on_off_pattern = np.greater(neuron_values, zero).astype("uint8")
        # if omit:
        #     if ignore_minor_values:
        #         comfort_level = (((self.known_patterns_numpy[class_id] ^ neuron_on_off_pattern) & neuron_on_off_pattern)[
        #             self.neurons_to_monitor[monitored_class]]).sum(axis=1).min()
        #     else:
        #         comfort_level = ((self.known_patterns_numpy[class_id] ^ neuron_on_off_pattern)[
        #             self.neurons_to_monitor[monitored_class]]).sum(axis=1).min()
        # else:
        #     if ignore_minor_values:
        #         comfort_level = ((self.known_patterns_numpy[class_id] ^ neuron_on_off_pattern) & neuron_on_off_pattern).sum(axis=1).min()
        #     else:
        #         comfort_level = (self.known_patterns_numpy[class_id] ^ neuron_on_off_pattern).sum(axis=1).min()
        mat_torch = torch.zeros(neuron_values.shape, device=neuron_values.device)
        neuron_on_off_pattern = neuron_values.gt(mat_torch).type(torch.uint8).to(neuron_values.device)
        shape = self.known_patterns_tensor[class_id].cpu().numpy().shape
        if len(shape) == 3:
            reshape = (shape[1], shape[0], shape[2])
        else:
            reshape = shape
        if omit:  # nie obsłużone
            if ignore_minor_values:
                comfort_level = (((self.known_patterns_tensor[class_id].reshape(
                    reshape) ^ neuron_on_off_pattern) & neuron_on_off_pattern)[
                    self.neurons_to_monitor[monitored_class]]).sum(dim=2).min(dim=0)
            else:
                comfort_level = ((self.known_patterns_tensor[class_id].reshape(reshape) ^ neuron_on_off_pattern)[
                    self.neurons_to_monitor[monitored_class]]).sum(dim=2).min(dim=0)
        else:
            if ignore_minor_values:
                # comfort_level = ((self.known_patterns_tensor[class_id].reshape(
                #     reshape) ^ neuron_on_off_pattern) & neuron_on_off_pattern).sum(dim=2).min(dim=0)
                comfort_level = []
                for i in range(neuron_on_off_pattern.shape[0]):
                    lvl = ((self.known_patterns_tensor[class_id[i], :, :] ^ neuron_on_off_pattern[i]) & neuron_on_off_pattern[i]).sum(dim=1).min()
                    comfort_level.append(lvl)
            else:
                comfort_level = (self.known_patterns_tensor[class_id].reshape(reshape) ^ neuron_on_off_pattern).sum(
                    dim=2).min(dim=0)
        if type(comfort_level) == list:
            comfort_level = np.array(comfort_level)
        else:
            comfort_level = comfort_level.values.numpy()
        return comfort_level

    def add_neuron_pattern(self, neuron_values, label):
        # if type(neuron_values) == torch.Tensor:
        #     neuron_values_np = neuron_values.cpu().numpy()
        # else:
        #     mat = np.zeros(neuron_values.shape)
        #     abs = np.greater(neuron_values, mat).astype("uint8")
        neuron_values_np = neuron_values.cpu().numpy()
        mat_np = np.zeros(neuron_values_np.shape)
        abs_np = np.greater(neuron_values_np, mat_np).astype("uint8")

        mat_torch = torch.zeros(neuron_values.shape, device=neuron_values.device)
        abs = neuron_values.gt(mat_torch).type(torch.uint8).to(neuron_values.device)

        for example_id in range(neuron_values.shape[0]):
            # dict version
            # if abs[example_id].tobytes() not in self.known_patterns_set[label[example_id]]:
            #     self.known_patterns_set[label[example_id]].add(abs[example_id].tobytes())
            #     if self.known_patterns_numpy[label[example_id]].size:
            #         self.known_patterns_numpy[label[example_id]] = np.vstack([self.known_patterns_numpy[label[example_id]], abs[example_id]])
            #     else:
            #         self.known_patterns_numpy[label[example_id]] = abs[example_id]
            if abs_np[example_id].tobytes() not in self.known_patterns_set[label[example_id]]:
                if self.known_patterns_tensor.numel():
                    self.known_patterns_tensor[label[example_id], len(self.known_patterns_set[label[example_id]])] = abs[
                        example_id]
                else:
                    self.known_patterns_tensor = torch.zeros(
                        (self.class_count, self.dataset_length // self.class_count) + abs[example_id].shape,
                        dtype=torch.uint8, device=abs.device)
                    self.known_patterns_tensor[label[example_id], len(self.known_patterns_set[label[example_id]])] = abs[
                        example_id]

                self.known_patterns_set[label[example_id]].add(abs_np[example_id].tobytes())


class EuclideanMonitor(BaseMonitor):

    def __init__(self, class_count, dataset_length, layers_shapes, neurons_to_monitor=None):
        super().__init__(class_count, dataset_length, layers_shapes, neurons_to_monitor)

    def get_comfort_level(self, neuron_values, class_id, omit, ignore_minor_values=True, monitored_class=None):
        comfort_level = sys.maxsize
        monitored_class = monitored_class if monitored_class else class_id
        for known_pattern in self.known_patterns_set[class_id]:
            if omit:
                if ignore_minor_values:
                    abs_values = np.abs(
                        np.array(neuron_values) - np.frombuffer(known_pattern, dtype=neuron_values.dtype))
                    abs_values = np.where(neuron_values > 0, abs_values, 0)
                    level = abs_values[self.neurons_to_monitor[monitored_class]].sum()
                else:
                    level = (np.abs(np.array(neuron_values) - np.frombuffer(known_pattern, dtype=neuron_values.dtype))[
                        self.neurons_to_monitor[monitored_class]]).sum()
            else:
                if ignore_minor_values:
                    abs_values = np.abs(
                        np.array(neuron_values) - np.frombuffer(known_pattern, dtype=neuron_values.dtype))
                    level = np.where(neuron_values > 0, abs_values, 0).sum()
                else:
                    level = np.abs(
                        np.array(neuron_values) - np.frombuffer(known_pattern, dtype=neuron_values.dtype)).sum()
            if level < comfort_level:
                comfort_level = level
        return comfort_level

    def add_neuron_pattern(self, neuron_values, label):
        if (not (type(neuron_values) == np.ndarray)) or (
                not (type(label) == np.ndarray)):
            raise TypeError('Input should be numpy array')

        for example_id in range(neuron_values.shape[0]):
            # print(f"add {neuron_values[example_id].shape}, bytes: {len(neuron_values[example_id].tobytes())}")
            self.known_patterns_set[label[example_id]].add(neuron_values[example_id].tobytes())

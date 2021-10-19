import sys
import torch
import numpy as np
from copy import deepcopy


class BaseMonitor(object):
    def __init__(self, class_count, device, layers_shapes, neurons_to_monitor):
        self.neurons_to_monitor = neurons_to_monitor
        self.layers_shapes = layers_shapes
        self.known_patterns_set = dict()


        self.class_count = class_count
        self.neurons_count = 0
        self.class_patterns_count = 0
        self.device = device
        for layer in layers_shapes:
            self.neurons_count += layer

        for i in range(class_count):
            self.known_patterns_set[i] = set()

        self.known_patterns_tensor = dict()
        for i in range(class_count):
            self.known_patterns_tensor[i] = torch.Tensor()

        self.known_patterns_tensor_cd = dict()
        for i in range(class_count):
            self.known_patterns_tensor_cd[i] = torch.Tensor()
        self.known_patterns_dict_cd = dict()
        for i in range(class_count):
            self.known_patterns_dict_cd[i] = 0


    def set_neurons_to_monitor(self, neurons_to_monitor):
        self.neurons_to_monitor = neurons_to_monitor
        for klass in neurons_to_monitor:
            self.neurons_to_monitor[klass] = np.concatenate(
                (neurons_to_monitor[klass], np.arange(self.layers_shapes[0], self.neurons_count)))

    def set_class_patterns_count(self, count):
        self.class_patterns_count = count


class Monitor(BaseMonitor):

    def __init__(self, class_count, device, layers_shapes, neurons_to_monitor=None):
        super().__init__(class_count, device, layers_shapes, neurons_to_monitor)

    def compute_hamming_distance(self, neuron_values, class_id, omit=False, ignore_minor_values=True, monitored_class=None):

        monitored_class = monitored_class if monitored_class else class_id
        mat_torch = torch.zeros(neuron_values.shape, device=neuron_values.device)
        neuron_on_off_pattern = neuron_values.gt(mat_torch).type(torch.uint8).to(torch.device(self.device))
        distance = []
        if omit:
            if ignore_minor_values:
                for i in range(neuron_on_off_pattern.shape[0]):
                    lvl = ((self.known_patterns_tensor[class_id[i]] ^ neuron_on_off_pattern[i]) & neuron_on_off_pattern[
                        i])[:, self.neurons_to_monitor[monitored_class[i]]].sum(dim=1).min()
                    distance.append(lvl.item())
            else:
                for i in range(neuron_on_off_pattern.shape[0]):
                    lvl = (self.known_patterns_tensor[class_id[i]] ^ neuron_on_off_pattern[i])[:,
                          self.neurons_to_monitor[monitored_class[i]]].sum(dim=1).min()
                    distance.append(lvl.item())
        else:
            if ignore_minor_values:
                for i in range(neuron_on_off_pattern.shape[0]):
                    lvl = ((self.known_patterns_tensor[class_id[i]] ^ neuron_on_off_pattern[i]) & neuron_on_off_pattern[
                        i]).sum(dim=1).min()
                    distance.append(lvl.item())
            else:
                for i in range(neuron_on_off_pattern.shape[0]):
                    lvl = (self.known_patterns_tensor[class_id[i]] ^ neuron_on_off_pattern[i]).sum(dim=1).min()
                    distance.append(lvl.item())
        if type(distance) == list:
            distance = np.array(distance)
        else:
            distance = distance.values.cpu().numpy()
        return distance

    def add_neuron_pattern(self, neuron_values, label):
        neuron_values_np = neuron_values.cpu().numpy()
        mat_np = np.zeros(neuron_values_np.shape)
        abs_np = np.greater(neuron_values_np, mat_np).astype("uint8")

        mat_torch = torch.zeros(neuron_values.shape, device=neuron_values.device)
        abs = neuron_values.gt(mat_torch).type(torch.uint8).to(torch.device(self.device))

        for example_id in range(neuron_values.shape[0]):
            if abs_np[example_id].tobytes() not in self.known_patterns_set[label[example_id]]:
                if self.known_patterns_tensor[label[example_id]].numel():
                    self.known_patterns_tensor[label[example_id]][len(self.known_patterns_set[label[example_id]])] = \
                        abs[
                            example_id]
                else:
                    self.known_patterns_tensor[label[example_id]] = torch.ones(
                        (self.class_patterns_count[label[example_id]],) + abs[example_id].shape,
                        dtype=torch.uint8, device=abs.device)

                    self.known_patterns_tensor[label[example_id]][len(self.known_patterns_set[label[example_id]])] = \
                        abs[
                            example_id]
                self.known_patterns_set[label[example_id]].add(abs_np[example_id].tobytes())

            if self.known_patterns_tensor_cd[label[example_id]].numel():
                self.known_patterns_tensor_cd[label[example_id]][self.known_patterns_dict_cd[label[example_id]]] = \
                    abs[
                        example_id]
            else:
                self.known_patterns_tensor_cd[label[example_id]] = torch.ones(
                    (self.class_patterns_count[label[example_id]],) + abs[example_id].shape,
                    dtype=torch.uint8, device=abs.device)

                self.known_patterns_tensor_cd[label[example_id]][self.known_patterns_dict_cd[label[example_id]]] = \
                    abs[
                        example_id]
                self.known_patterns_dict_cd[label[example_id]] += 1

    def cut_duplicates(self):
        for i in self.known_patterns_tensor:
            self.known_patterns_tensor[i] = self.known_patterns_tensor[i][:len(self.known_patterns_set[i]), :]

class EuclideanMonitor(BaseMonitor):

    def __init__(self, class_count, device, layers_shapes, neurons_to_monitor=None):
        super().__init__(class_count, device, layers_shapes, neurons_to_monitor)

    def compute_hamming_distance(self, neuron_values, class_id, omit, ignore_minor_values=True, monitored_class=None):
        comfort_level = sys.maxsize
        monitored_class = monitored_class if monitored_class else class_id
        for known_pattern in self.known_patterns_set[class_id]:
            if omit:
                if ignore_minor_values:
                    abs_values = np.abs(
                        np.array(neuron_values) - np.frombuffer(known_pattern, dtype=neuron_values.dtype))
                    abs_values = np.where(neuron_values > 0, abs_values, 0)
                    distance = abs_values[self.neurons_to_monitor[monitored_class]].sum()
                else:
                    distance = (np.abs(np.array(neuron_values) - np.frombuffer(known_pattern, dtype=neuron_values.dtype))[
                        self.neurons_to_monitor[monitored_class]]).sum()
            else:
                if ignore_minor_values:
                    abs_values = np.abs(
                        np.array(neuron_values) - np.frombuffer(known_pattern, dtype=neuron_values.dtype))
                    distance = np.where(neuron_values > 0, abs_values, 0).sum()
                else:
                    distance = np.abs(
                        np.array(neuron_values) - np.frombuffer(known_pattern, dtype=neuron_values.dtype)).sum()
            if distance < comfort_level:
                comfort_level = distance
        return comfort_level

    def add_neuron_pattern(self, neuron_values, label):
        if (not (type(neuron_values) == np.ndarray)) or (
                not (type(label) == np.ndarray)):
            raise TypeError('Input should be numpy array')

        for example_id in range(neuron_values.shape[0]):
            # print(f"add {neuron_values[example_id].shape}, bytes: {len(neuron_values[example_id].tobytes())}")
            self.known_patterns_set[label[example_id]].add(neuron_values[example_id].tobytes())

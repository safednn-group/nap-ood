import sys

import numpy as np
from copy import deepcopy


class BaseMonitor(object):
    def __init__(self, class_count, layers_shapes, neurons_to_monitor):
        self.neurons_to_monitor = neurons_to_monitor
        self.layers_shapes = layers_shapes
        self.known_patterns_set = dict()
        self.known_patterns_numpy = dict()
        self.neurons_count = 0
        for layer in layers_shapes:
            self.neurons_count += layer

        for i in range(class_count):
            self.known_patterns_set[i] = set()

        for i in range(class_count):
            self.known_patterns_numpy[i] = np.array([])

    def set_neurons_to_monitor(self, neurons_to_monitor):
        self.neurons_to_monitor = neurons_to_monitor
        for klass in neurons_to_monitor:
            self.neurons_to_monitor[klass] = np.concatenate(
                (neurons_to_monitor[klass], np.arange(self.layers_shapes[0], self.neurons_count)))


class Monitor(BaseMonitor):

    def __init__(self, class_count, layers_shapes, neurons_to_monitor=None):
        super().__init__(class_count, layers_shapes, neurons_to_monitor)

    def get_comfort_level(self, neuron_values, class_id, omit, ignore_minor_values=True, monitored_class=None):
        zero = np.zeros(neuron_values.shape)
        neuron_on_off_pattern = np.greater(neuron_values, zero).astype("uint8")
        monitored_class = monitored_class if monitored_class else class_id
        if omit:
            if ignore_minor_values:
                comfort_level = (((self.known_patterns_numpy[class_id] ^ neuron_on_off_pattern) & neuron_on_off_pattern)[
                    self.neurons_to_monitor[monitored_class]]).sum(axis=1).min()
            else:
                comfort_level = ((self.known_patterns_numpy[class_id] ^ neuron_on_off_pattern)[
                    self.neurons_to_monitor[monitored_class]]).sum(axis=1).min()
        else:
            if ignore_minor_values:
                comfort_level = ((self.known_patterns_numpy[class_id] ^ neuron_on_off_pattern) & neuron_on_off_pattern).sum(axis=1).min()
            else:
                comfort_level = (self.known_patterns_numpy[class_id] ^ neuron_on_off_pattern).sum(axis=1).min()
        return comfort_level

    def add_neuron_pattern(self, neuron_values, label):
        if (not (type(neuron_values) == np.ndarray)) or (
                not (type(label) == np.ndarray)):
            raise TypeError('Input should be numpy array')

        mat = np.zeros(neuron_values.shape)
        abs = np.greater(neuron_values, mat).astype("uint8")
        for example_id in range(neuron_values.shape[0]):
            if abs[example_id].tobytes() not in self.known_patterns_set[label[example_id]]:
                self.known_patterns_set[label[example_id]].add(abs[example_id].tobytes())
                if self.known_patterns_numpy[label[example_id]].size:
                    self.known_patterns_numpy[label[example_id]] = np.vstack([self.known_patterns_numpy[label[example_id]], abs[example_id]])
                else:
                    self.known_patterns_numpy[label[example_id]] = abs[example_id]


class EuclideanMonitor(BaseMonitor):

    def __init__(self, class_count, layers_shapes, neurons_to_monitor=None):
        super().__init__(class_count, layers_shapes, neurons_to_monitor)

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

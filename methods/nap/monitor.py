import sys

import numpy as np
from copy import deepcopy


class Monitor(object):

    def __init__(self, class_count, neurons_to_monitor=None, layers_shapes=None):
        self.class_count = class_count
        self.neurons_to_monitor = neurons_to_monitor
        self.layers_shapes = layers_shapes
        self.known_patterns = dict()
        self.neurons_count = 0
        for layer in layers_shapes:
            self.neurons_count += layer

        for i in range(self.class_count):
            self.known_patterns[i] = set()

    def get_comfort_level(self, neuron_values, class_id, omit, ignore_minor_values=True, monitored_class=None):
        zero = np.zeros(neuron_values.shape)
        neuron_on_off_pattern = np.greater(neuron_values, zero).astype("uint8")
        # if omit:
        #     neuron_on_off_pattern = neuron_on_off_pattern[self.neurons_to_monitor[class_id]]
        comfort_level = len(neuron_on_off_pattern)
        monitored_class = monitored_class if monitored_class else class_id
        for known_pattern in self.known_patterns[class_id]:
            if omit:
                if ignore_minor_values:
                    # print(f"onoff {sum(neuron_on_off_pattern)}")
                    # print(f" known{np.frombuffer(known_pattern, dtype=neuron_on_off_pattern.dtype).sum()}")
                    level = (((neuron_on_off_pattern ^ np.frombuffer(known_pattern, dtype=neuron_on_off_pattern.dtype)) & neuron_on_off_pattern)[
                        self.neurons_to_monitor[monitored_class]]).sum()
                else:
                    level = ((neuron_on_off_pattern ^ np.frombuffer(known_pattern, dtype=neuron_on_off_pattern.dtype))[
                        self.neurons_to_monitor[monitored_class]]).sum()
            else:
                if ignore_minor_values:
                    level = ((neuron_on_off_pattern ^ np.frombuffer(known_pattern, dtype=neuron_on_off_pattern.dtype)) & neuron_on_off_pattern).sum()
                else:
                    level = (neuron_on_off_pattern ^ np.frombuffer(known_pattern, dtype=neuron_on_off_pattern.dtype)).sum()
            if level < comfort_level:
                comfort_level = level
        return comfort_level

    def add_neuron_pattern(self, neuron_values, label):
        if (not (type(neuron_values) == np.ndarray)) or (
                not (type(label) == np.ndarray)):
            raise TypeError('Input should be numpy array')

        mat = np.zeros(neuron_values.shape)
        abs = np.greater(neuron_values, mat).astype("uint8")
        for example_id in range(neuron_values.shape[0]):
            self.known_patterns[label[example_id]].add(abs[example_id].tobytes())

    def set_neurons_to_monitor(self, neurons_to_monitor):
        self.neurons_to_monitor = neurons_to_monitor
        for klass in neurons_to_monitor:
            self.neurons_to_monitor[klass] = np.concatenate((neurons_to_monitor[klass], np.arange(self.layers_shapes[0], self.neurons_count)))


class RangeMonitor(object):

    def __init__(self, class_count, neurons_to_monitor=None, layers_shapes=None):
        self.class_count = class_count
        self.neurons_to_monitor = neurons_to_monitor
        self.known_patterns = dict()
        self.layers_shapes = layers_shapes
        self.neurons_count = 0
        for layer in layers_shapes:
            self.neurons_count += layer

        for i in range(self.class_count):
            self.known_patterns[i] = set()

    def get_comfort_level(self, neuron_values, class_id, omit, ignore_minor_values=True, monitored_class=None):
        comfort_level = sys.maxsize
        monitored_class = monitored_class if monitored_class else class_id
        for known_pattern in self.known_patterns[class_id]:
            if omit:
                if ignore_minor_values:
                    abs_values = np.abs(np.array(neuron_values) - np.frombuffer(known_pattern, dtype=neuron_values.dtype))
                    abs_values = np.where(neuron_values > 0, abs_values, 0)
                    level = abs_values[self.neurons_to_monitor[monitored_class]].sum()
                else:
                    level = (np.abs(np.array(neuron_values) - np.frombuffer(known_pattern, dtype=neuron_values.dtype))[
                        self.neurons_to_monitor[monitored_class]]).sum()
            else:
                if ignore_minor_values:
                    abs_values = np.abs(np.array(neuron_values) - np.frombuffer(known_pattern, dtype=neuron_values.dtype))
                    level = np.where(neuron_values > 0, abs_values, 0).sum()
                else:
                    level = np.abs(np.array(neuron_values) - np.frombuffer(known_pattern, dtype=neuron_values.dtype)).sum()
            if level < comfort_level:
                comfort_level = level
        return comfort_level

    def add_neuron_pattern(self, neuron_values, label):
        if (not (type(neuron_values) == np.ndarray)) or (
                not (type(label) == np.ndarray)):
            raise TypeError('Input should be numpy array')

        for example_id in range(neuron_values.shape[0]):
            # print(f"add {neuron_values[example_id].shape}, bytes: {len(neuron_values[example_id].tobytes())}")
            self.known_patterns[label[example_id]].add(neuron_values[example_id].tobytes())

    def set_neurons_to_monitor(self, neurons_to_monitor):
        self.neurons_to_monitor = neurons_to_monitor
        for klass in neurons_to_monitor:
            self.neurons_to_monitor[klass] = np.concatenate((neurons_to_monitor[klass], np.arange(self.layers_shapes[0], self.neurons_count)))
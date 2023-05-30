"""
Neuron Activation Patterns runtime monitor implementation
Author Bartlomiej Olber
Copyright (c) SafeDNN group. All rights reserved.
"""

import torch
import numpy as np


class FullNetMonitor(object):

    def __init__(self, class_count, device, layers_shapes):
        self.class_patterns_count = dict()
        self.device = device
        if not isinstance(layers_shapes, list):
            layers_shapes = [layers_shapes]

        self.layers_shapes = layers_shapes
        self.known_patterns_set = dict()
        self.known_patterns_tensor = dict()

        for i in range(class_count):
            self.known_patterns_set[i] = dict()
            self.known_patterns_tensor[i] = dict()
            for j in range(len(layers_shapes)):
                self.known_patterns_set[i][j] = set()
                self.known_patterns_tensor[i][j] = torch.Tensor()

    def compute_hamming_distance(self, neuron_values, class_id, mean=False):

        mat_torch = torch.zeros(neuron_values.shape, device=neuron_values.device)
        neuron_on_off_pattern = neuron_values.gt(mat_torch).type(torch.uint8).to(torch.device(self.device))
        distance = []

        for i in range(neuron_on_off_pattern.shape[0]):
            full_net_distances = []
            offset = 0
            for shape_id, shape in enumerate(self.layers_shapes):
                if self.known_patterns_tensor[class_id[i]][shape_id].numel():
                    if mean:
                        lvl = (self.known_patterns_tensor[class_id[i]][shape_id] ^ neuron_on_off_pattern[i,
                                                                                   offset:offset + shape]).sum(
                            dim=1).float().mean()
                    else:
                        lvl = (self.known_patterns_tensor[class_id[i]][shape_id] ^ neuron_on_off_pattern[i,
                                                                                   offset:offset + shape]).sum(
                            dim=1).min()
                else:
                    lvl = shape
                offset += shape
                full_net_distances.append(int(lvl))
            distance.append(full_net_distances)
        distance = np.array(distance)
        return distance

    def add_neuron_pattern(self, neuron_values, label):
        neuron_values_np = neuron_values.cpu().numpy()
        mat_np = np.zeros(neuron_values_np.shape)
        abs_np = np.greater(neuron_values_np, mat_np).astype("uint8")

        mat_torch = torch.zeros(neuron_values.shape, device=neuron_values.device)
        abs = neuron_values.gt(mat_torch).type(torch.uint8).to(torch.device(self.device))
        for example_id in range(neuron_values.shape[0]):
            offset = 0
            for shape_id, shape in enumerate(self.layers_shapes):
                abs_np_slice = abs_np[example_id, offset:offset + shape]
                abs_slice = abs[example_id, offset:offset + shape]
                if abs_np_slice.tobytes() not in self.known_patterns_set[label[example_id]][shape_id]:
                    if self.known_patterns_tensor[label[example_id]][shape_id].numel():
                        self.known_patterns_tensor[label[example_id]][shape_id][
                            len(self.known_patterns_set[label[example_id]][shape_id])] = \
                            abs_slice
                    else:
                        self.known_patterns_tensor[label[example_id]][shape_id] = torch.ones(
                            (self.class_patterns_count[label[example_id]],) + abs_slice.shape,
                            dtype=torch.uint8, device=abs.device)

                        self.known_patterns_tensor[label[example_id]][shape_id][
                            len(self.known_patterns_set[label[example_id]][shape_id])] = \
                            abs_slice
                    self.known_patterns_set[label[example_id]][shape_id].add(abs_np_slice.tobytes())
                offset += shape

    def cut_duplicates(self):
        for i in self.known_patterns_tensor:
            for j in self.known_patterns_tensor[i]:
                if self.known_patterns_tensor[i][j].numel():
                    self.known_patterns_tensor[i][j] = self.known_patterns_tensor[i][j][
                                                       :len(self.known_patterns_set[i][j]),
                                                       :]

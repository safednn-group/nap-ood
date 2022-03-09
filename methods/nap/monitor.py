import sys
import torch
import numpy as np
import methods.nap.numba_balltree.ball_tree as bt
import sklearn.neighbors as sn


class FullNetMonitor(object):

    def __init__(self, class_count, device, layers_shapes):
        self.layers_shapes = layers_shapes

        self.class_count = class_count
        self.neurons_count = 0
        self.class_patterns_count = 0
        self.device = device
        for layer in layers_shapes:
            self.neurons_count += layer

        self.known_patterns_set = dict()
        self.known_patterns_tensor = dict()
        self.forest = dict()

        for i in range(class_count):
            self.known_patterns_set[i] = dict()
            self.known_patterns_tensor[i] = dict()
            for j in range(len(layers_shapes)):
                self.known_patterns_set[i][j] = set()
                self.known_patterns_tensor[i][j] = torch.Tensor()

    def set_neurons_to_monitor(self, neurons_to_monitor):
        self.neurons_to_monitor = neurons_to_monitor
        for klass in neurons_to_monitor:
            self.neurons_to_monitor[klass] = np.concatenate(
                (neurons_to_monitor[klass], np.arange(self.layers_shapes[0], self.neurons_count)))

    def set_class_patterns_count(self, count):
        self.class_patterns_count = count

    def compute_hamming_distance(self, neuron_values, class_id, tree=False):

        mat_torch = torch.zeros(neuron_values.shape, device=neuron_values.device)
        neuron_on_off_pattern = neuron_values.gt(mat_torch).type(torch.uint8).to(torch.device(self.device))
        distance = []

        for i in range(neuron_on_off_pattern.shape[0]):
            full_net_distances = []
            offset = 0
            for shape_id, shape in enumerate(self.layers_shapes):
                if tree:
                    lvl = self.forest[class_id[i]][shape_id].query(
                        np.reshape(neuron_on_off_pattern.cpu()[i, offset:offset + shape], (1, -1)))[0]
                else:
                    lvl = (self.known_patterns_tensor[class_id[i]][shape_id] ^ neuron_on_off_pattern[i,
                                                                               offset:offset + shape]).sum(
                        dim=1).min()
                offset += shape
                full_net_distances.append(lvl.item())
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

    def make_forest(self):
        for i in range(self.class_count):
            self.forest[i] = dict()
            for j in range(len(self.known_patterns_tensor[i])):
                # self.forest[i][j] = bt.BallTree(self.known_patterns_tensor[i][j].cpu())
                self.forest[i][j] = sn.BallTree(self.known_patterns_tensor[i][j].cpu())

    def cut_duplicates(self):
        for i in self.known_patterns_tensor:
            for j in self.known_patterns_tensor[i]:
                if self.known_patterns_tensor[i][j].numel():
                    self.known_patterns_tensor[i][j] = self.known_patterns_tensor[i][j][
                                                       :len(self.known_patterns_set[i][j]),
                                                       :]

    def trim_class_zero(self, length):
        for i in range(len(self.layers_shapes)):
            self.known_patterns_tensor[0][i] = self.known_patterns_tensor[0][i][:length, :]

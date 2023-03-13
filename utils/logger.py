import numpy as np
import re

class Measure:
	measurements = []
	measure_name = ''
	measure_normalizer = None
	legend = None

	def __init__(self, measure_name):
		self.measure_name = measure_name
		self.measurements = []

	def reset(self):
		self.measurements = []

	def add_measurement(self, measurement, epoch, iteration=None):
		while len(self.measurements)-1 < epoch:
			self.measurements.append(None)
		if iteration is None:
			self.measurements[epoch] = measurement
		else:
			if self.measurements[epoch] is None:
				self.measurements[epoch] = []
			while len(self.measurements[epoch])-1 < iteration:
				self.measurements[epoch].append(None)
			self.measurements[epoch][iteration] = measurement

	def mean_epoch(self, epoch=-1):
		vals = self.measurements[epoch]
		if type(vals) == list:
			if self.measure_normalizer is None:
				return np.array(vals).mean()
			else:
				return np.array(vals).sum()/self.measure_normalizer
		else:
			return vals

	def sum_epoch(self, epoch=-1):
		vals = self.measurements[epoch]
		if type(vals) == list:
			return np.array(vals).sum()
		else:
			return vals

	def generate_average_XY(self, second_order=False):
		is_average = False
		measure_dim = 1
		for i in range(len(self.measurements)):
			if type(self.measurements[i]) == list:
				is_average = True
				if type(self.measurements[i][0]) == list:
					measure_dim = max(measure_dim, len(self.measurements[i][0]))

		X = np.arange(len(self.measurements))
		Y = np.array(self.measurements)
		skip = False
		if is_average:
			dummy_Y = []
			for i in range(len(Y)):
				if type(Y[i]) == list:
					if self.measure_normalizer is None:
						dummy_Y.append(np.nanmean(np.array(Y[i]), 0))
					else:
						dummy_Y.append(np.nansum(np.array(Y[i]), 0)/self.measure_normalizer)
				if type(Y[i]) == np.ndarray:
					if self.measure_normalizer is None:
						dummy_Y.append(np.nanmean(Y[i], 0))
					else:
						dummy_Y.append(np.nansum(Y[i], 0)/self.measure_normalizer)
				if Y[i] is None:
					if measure_dim > 1:
						dummy_Y.append([float('nan') for q in range(measure_dim)])
					else:
						dummy_Y.append(float('nan'))
					skip = skip or (i==0)
			Y = np.array(dummy_Y)
		else:
			if Y[0] is None:
				skip = True
		if skip:
			X = np.delete(X, 0, axis=0)
			Y = np.delete(Y, 0, axis=0)
		if second_order:
			Y_n = np.zeros(len(Y))
			for i in range(len(Y)):
				Y_n[i] = Y[i].mean()
			Y = Y_n
		return X, Y, is_average


class Logger:
	measures = None

	def __init__(self):
		self.measures = {}

	def log(self, measure_name, measurement, epoch, iteration=None):
		measure = None
		if measure_name in self.measures.keys():
			measure = self.measures[measure_name]
		else:
			measure = Measure(measure_name)
			self.measures[measure_name] = measure

		measure.add_measurement(measurement, epoch, iteration)

	def get_measure(self, measure_name):
		assert measure_name in self.measures.keys(), 'Measure %s is not defined'%measure_name
		return self.measures[measure_name]

	def reset_measure(self, measure_name):
		if measure_name in self.measures.keys():
			self.measures[measure_name].reset()

	def mean_epoch(self, measure_name, epoch=-1):
		measure = self.get_measure(measure_name)
		return measure.mean_epoch(epoch=epoch)

	def sum_epoch(self, measure_name, epoch=-1):
		measure = self.get_measure(measure_name)
		return measure.sum_epoch(epoch=epoch)


	def __str__(self):
		return 'Logger with measures\n(%s)'%(', '.join(self.measures.keys()))

if __name__ == '__main__':
	import random
	logger = Logger()

	for epoch in range(0, 10):
		for iteration in range(0, 20):
			logger.log('mIoU', [random.random()+epoch for c in range(5)], epoch, iteration)
			logger.log('train_loss', random.random()+epoch/10, epoch, iteration)
			logger.log('test_loss', random.random()+epoch/10+0.4, epoch, iteration)

	for epoch in range(1, 10):
		for iteration in range(0, 20):
			logger.log('mIoU2', [random.random()+epoch for c in range(4)], epoch, iteration)

	logger.get_measure('mIoU').legend = ['a', 'b', 'c', 'd', 'e']

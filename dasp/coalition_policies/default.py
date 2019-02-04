from abc import ABC, abstractmethod
import numpy as np


def spaced_elements(array, numElems = 4):
    return [x[len(x)//2] for x in np.array_split(np.array(array), numElems)]


class AbstractPlayerIterator(ABC):

    @abstractmethod
    def get_number_of_players(self):
        pass

    @abstractmethod
    def get_explanation_shape(self):
        pass

    @abstractmethod
    def get_coalition_size(self):
        pass

    @abstractmethod
    def get_steps(self):
        pass

    @abstractmethod
    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        raise StopIteration


class DefaultPlayerIterator(AbstractPlayerIterator):

    def __init__(self, input, steps=None):
        self.input_shape = input.shape[1:]
        self.n_players = np.prod(self.input_shape)
        self.i = 0
        self.kn = steps if steps is not None else self.n_players
        self.ks = spaced_elements(range(self.n_players), self.kn)

    def get_number_of_players(self):
        return self.n_players

    def get_explanation_shape(self):
        return self.input_shape

    def get_coalition_size(self):
        return 1

    def get_steps(self):
        return self.ks

    def __iter__(self):
        self.i = 0
        return self

    def _get_mask(self, i):
        mask = np.zeros(self.n_players, dtype='int32')
        mask[i] = 1
        return mask.reshape(self.input_shape)

    def __next__(self):
        if self.i == self.n_players:
            raise StopIteration
        m = self._get_mask(self.i)
        self.i = self.i + 1
        return m

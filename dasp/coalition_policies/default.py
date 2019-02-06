from abc import ABC, abstractmethod
import numpy as np


def spaced_elements(array, numElems = 4):
    return [x[len(x)//2] for x in np.array_split(np.array(array), numElems)]


class AbstractPlayerIterator(ABC):

    def __init__(self, input, steps=None, random=False):
        self._assert_input_compatibility(input)
        self.input_shape = input.shape[1:]
        self.steps = steps
        self.random = random
        self.n_players = self._get_number_of_players_from_shape()
        self.permutation = np.array(range(self.n_players), 'int32')
        if random is True:
            self.permutation = np.random.permutation(self.permutation)
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

    def __next__(self):
        if self.i == self.n_players:
            raise StopIteration
        m = self._get_mask_for_index(self.i)
        self.i = self.i + 1
        return m

    @abstractmethod
    def _assert_input_compatibility(self, input):
        pass

    @abstractmethod
    def _get_mask_for_index(self, i):
        pass

    @abstractmethod
    def _get_number_of_players_from_shape(self):
        pass


class DefaultPlayerIterator(AbstractPlayerIterator):

    def _assert_input_compatibility(self, input):
        assert len(input.shape) > 1, 'DefaultPlayerIterator requires an input with 2 or more dimensions'

    def _get_number_of_players_from_shape(self):
        return int(np.prod(self.input_shape))

    def _get_mask_for_index(self, i):
        mask = np.zeros(self.n_players, dtype='int32')
        mask[self.permutation[i]] = 1
        return mask.reshape(self.input_shape)




class ImagePlayerIterator(AbstractPlayerIterator):

    def __init__(self, input, steps=None, random=False, merge_channels=False, coalition_patch_size=None):
        self.merge_channels = merge_channels
        self.coalition_patch_size = coalition_patch_size
        super(ImagePlayerIterator, self).__init__(input, steps, random)

    def _assert_input_compatibility(self, input):
        assert len(input.shape) == 4, 'ImagePlayerIterator requires an input with 4 dimensions'

    def _get_number_of_players_from_shape(self):
        return int(np.prod(self.input_shape))

    def _get_mask_for_index(self, i):
        mask = np.zeros(self.n_players, dtype='int32')
        mask[self.permutation[i]] = 1
        return mask.reshape(self.input_shape)



def main():
    iter = DefaultPlayerIterator(np.random.random((1, 12, 4)), random=True)
    print (iter.get_number_of_players())
    print (iter.get_explanation_shape())
    print (next(iter))

if __name__ == "__main__":
    main()
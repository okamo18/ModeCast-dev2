import numpy as np


class RingBuffer:
    def __init__(self, data: np.ndarray):
        self.data = data

    def append(self, value: np.ndarray) -> None:
        if value.shape != self.data.shape[:-1]:
            raise ValueError("Input value dimensions do not match buffer's dimensions")

        self.data = np.roll(self.data, shift=-1, axis=1)
        self.data[:, -1] = value

    def get_all(self) -> np.ndarray:
        return self.data

    @property
    def head(self) -> np.ndarray:
        return self.data[:, 0]

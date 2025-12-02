import numpy as np

from src.module.dmd import DMD
from src.utils.preprocessor import create_variables
from src.utils.ring_buffer import RingBuffer

H = 6
INIT = 10


def data_loader() -> np.ndarray:
    t = np.arange(0, 10, 0.01)

    OBSERVASION_TIME = 1
    trend = 10 * (t / 7 - OBSERVASION_TIME) ** 2
    periodic1 = np.sin(10 * 2 * np.pi * t / 10) / np.exp(-2 * t / 10)
    periodic2 = np.sin(5 * 2 * np.pi * t)
    np.random.seed(123)
    noise = 1.5 * (np.random.rand(len(t)) - 0.5)

    x1 = np.exp(t / 5) * np.sin(2 * np.pi * t)
    x2 = np.cos(3 * np.pi * t)
    x3 = 5 * np.exp(-t / 5)
    x4 = trend + periodic1 + periodic2 + noise

    return np.array(
        [
            (x1 - np.mean(x1)) / np.std(x1),
            (x2 - np.mean(x2)) / np.std(x2),
            (x3 - np.mean(x3)) / np.std(x3),
            (x4 - np.mean(x4)) / np.std(x4),
        ]
    )


def test_windowed_incremental_SVD() -> None:
    model_batch = DMD(n_mode=4 * H)
    model_stream = DMD(n_mode=4 * H, window=INIT)
    x = data_loader()
    X, Y = create_variables(x, h=H)
    Xold = X[:, :INIT]
    Xnew = X[:, INIT:]

    full_Uh, full_Sinv, full_V = model_batch._svd(X)

    model_stream.X = RingBuffer(Xold)
    model_stream._svd(Xold)
    for xt in Xnew.T:
        new_uh, new_sinv, new_v = model_stream._incremental_svd(xt.reshape(-1, 1))

    assert np.allclose(
        (full_Uh.T @ np.linalg.inv(full_Sinv) @ full_V.T)[:, -INIT:],
        new_uh.T @ np.linalg.inv(new_sinv) @ new_v.T,
        atol=5e-1,
    )

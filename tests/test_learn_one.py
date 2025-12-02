import numpy as np

from src.module.dmd import DMD
from src.utils.preprocessor import create_variables

H = 4
INIT = 80 - H
END = 200 - H


def data_loader1() -> np.ndarray:
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

    return np.array([x1, x2, x3, x4])


def data_loader2() -> np.ndarray:
    np.random.seed(0)
    return np.random.rand(4, 200)


def test_learn_one() -> None:
    model_batch = DMD(n_mode=4 * H)
    model_stream = DMD(n_mode=4 * H)
    x = data_loader1()

    print("#################### BATCH ###################")
    X, Y = create_variables(x, h=H)
    model_batch.fit(X[:, :END], Y[:, :END])
    print()

    print("#################### STREAM ##################")
    model_stream.fit(X[:, :INIT], Y[:, :INIT])
    for i, (xt, yt) in enumerate(zip(X[:, INIT:END].T, Y[:, INIT:END].T)):
        model_batch.fit(X[:, : INIT + i + 1], Y[:, : INIT + i + 1])
        model_stream.learn_one(xt.reshape(-1, 1), yt.reshape(-1, 1))
        assert np.allclose(np.real(model_batch.A), np.real(model_stream.A), atol=3e-3)


if __name__ == "__main__":
    test_learn_one()

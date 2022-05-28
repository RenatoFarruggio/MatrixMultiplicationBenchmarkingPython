from typing import Any

import tensorflow as tf
import numpy as np
import torch


class Benchmark:

    first_matrix: Any
    second_matrix: Any
    resulting_matrix: Any

    # This should always be "undefined" in this class
    method_name = "undefined"

    def __init__(self):
        pass

    def set_method_name(self, method_name):
        self.method_name = method_name

    def set_matrices(self, first_array: [], second_array: []):
        self.first_matrix = self.array_to_matrix(first_array)
        self.second_matrix = self.array_to_matrix(second_array)
        pass

    def get_method_name(self):
        return self.method_name


class BenchmarkNumpy(Benchmark):
    def __init__(self):
        super().__init__()
        super().set_method_name("Numpy")

    def array_to_matrix(self, array: []) -> np.ndarray:
        return np.asarray(array)

    def multiply_matrices(self) -> None:
        self.resulting_matrix = np.matmul(self.first_matrix, self.second_matrix)


class BenchmarkTensorflow(Benchmark):
    # version: tensorflow-cpu 2.9.1
    # url: https://pypi.org/project/tensorflow-cpu/
    def __init__(self):
        super().__init__()
        super().set_method_name("Tensorflow")

    def array_to_matrix(self, array: []) -> tf.constant:
        return tf.convert_to_tensor(array, dtype=float)

    def multiply_matrices(self) -> None:
        self.resulting_matrix = tf.matmul(self.first_matrix, self.second_matrix)


class BenchmarkPytorch(Benchmark):
    # version: 1.11.0
    # url: https://pypi.org/project/torch/
    def __init__(self):
        super().__init__()
        super().set_method_name("PyTorch")

    def array_to_matrix(self, array: []) -> torch.Tensor:
        return torch.tensor(array)

    def multiply_matrices(self) -> None:
        self.resulting_matrix = torch.mm(self.first_matrix, self.second_matrix)

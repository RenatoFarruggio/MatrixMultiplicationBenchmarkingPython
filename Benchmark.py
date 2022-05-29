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

    def set_matrices_dense_dense(self, first_array: [], second_array: []):
        self.first_matrix = self.array_to_dense_matrix(first_array)
        self.second_matrix = self.array_to_dense_matrix(second_array)
        pass


    def set_matrices_sparse_dense(self, first_array: [], second_array: []):
        self.first_matrix = self.array_to_sparse_matrix(first_array)
        self.second_matrix = self.array_to_dense_matrix(second_array)
        pass

    def set_matrices_sparse_sparse(self, first_array: [], second_array: []):
        self.first_matrix = self.array_to_sparse_matrix(first_array)
        self.second_matrix = self.array_to_sparse_matrix(second_array)
        pass

    def get_method_name(self):
        return self.method_name


class BenchmarkNumpyDenseDense(Benchmark):
    def __init__(self):
        super().__init__()
        super().set_method_name("Numpy D-D")

    def array_to_dense_matrix(self, array: []) -> np.ndarray:
        return np.asarray(array)

    def array_to_sparse_matrix(self, array: []):
        print("WARNING (custom): Method array_to_sparse_matrix undefined for Numpy! "
              "Using array_to_dense_matrix instead.")
        return self.array_to_dense_matrix(array)

    def multiply_matrices(self) -> None:
        self.resulting_matrix = np.matmul(self.first_matrix, self.second_matrix)

    def set_matrices(self, first_array: [], second_array: []):
        super().set_matrices_dense_dense(first_array, second_array)


class BenchmarkTensorflowDenseDense(Benchmark):
    # version: tensorflow-cpu 2.9.1
    # url: https://pypi.org/project/tensorflow-cpu/
    def __init__(self):
        super().__init__()
        super().set_method_name("Tensorflow D-D")

    def array_to_dense_matrix(self, array: []) -> tf.Tensor:
        return tf.convert_to_tensor(array, dtype=float)

    def multiply_matrices(self) -> None:
        self.resulting_matrix = tf.matmul(self.first_matrix, self.second_matrix)

    def array_to_sparse_matrix(self, array: []):
        return tf.sparse.from_dense(self.array_to_dense_matrix(array))

    def set_matrices(self, first_array: [], second_array: []) -> None:
        super().set_matrices_dense_dense(first_array, second_array)


class BenchmarkTensorflowSparseDense(Benchmark):
    def __init__(self):
        super().__init__()
        super().set_method_name("Tensorflow S-D")

    def array_to_dense_matrix(self, array: []) -> tf.Tensor:
        return tf.convert_to_tensor(array, dtype=float)

    def array_to_sparse_matrix(self, array: []):
        return tf.sparse.from_dense(self.array_to_dense_matrix(array))

    def multiply_matrices(self) -> None:
        self.resulting_matrix = tf.sparse.sparse_dense_matmul(self.first_matrix, self.second_matrix)

    def set_matrices(self, first_array: [], second_array: []) -> None:
        super().set_matrices_sparse_dense(first_array, second_array)


class BenchmarkPytorchDenseDense(Benchmark):
    # version: 1.11.0
    # url: https://pypi.org/project/torch/
    def __init__(self):
        super().__init__()
        super().set_method_name("PyTorch S-D")

    def array_to_dense_matrix(self, array: []) -> torch.Tensor:
        return torch.tensor(array)

    def array_to_sparse_matrix(self, array: []) -> torch.Tensor:
        return torch.tensor(array).to_sparse_coo()

    def multiply_matrices(self) -> None:
        self.resulting_matrix = torch.mm(self.first_matrix, self.second_matrix)

    def set_matrices(self, first_array: [], second_array: []) -> None:
        super().set_matrices_dense_dense(first_array, second_array)


class BenchmarkPytorchSparseDense(Benchmark):
    # version: 1.11.0
    # url: https://pypi.org/project/torch/
    def __init__(self):
        super().__init__()
        super().set_method_name("PyTorch S-D")

    def array_to_dense_matrix(self, array: []) -> torch.Tensor:
        return torch.tensor(array)

    def array_to_sparse_matrix(self, array: []) -> torch.Tensor:
        return torch.tensor(array).to_sparse_coo()

    def multiply_matrices(self) -> None:
        self.resulting_matrix = torch.sparse.mm(self.first_matrix, self.second_matrix)

    def set_matrices(self, first_array: [], second_array: []) -> None:
        super().set_matrices_sparse_dense(first_array, second_array)

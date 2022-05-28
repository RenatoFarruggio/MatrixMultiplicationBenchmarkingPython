from ArrayFactory import ArrayFactory
import Benchmark

import time

def run_benchmarks(warmup_rounds, test_rounds, matrix_sizes) -> None:
    benchmarks = []
    benchmarks.append(Benchmark.BenchmarkNumpy())
    benchmarks.append(Benchmark.BenchmarkTensorflow())
    benchmarks.append(Benchmark.BenchmarkPytorch())

    for matrix_size in matrix_sizes:
        array_factory = ArrayFactory(matrix_size)
        filename = "results-" + str(matrix_size) + ".txt"

        with open(filename, "wt") as f:
            f.write(f"Size: {matrix_size}\n")
            f.write(f"Test rounds: {test_rounds}\n")
            f.write("\n")

        # Run Tests
        for benchmark in benchmarks:
            time_needed_sum = 0
            method_name = benchmark.get_method_name()
            print("Benchmarking for:", method_name)

            for current_round in range(warmup_rounds):
                print(f"Warmup round {current_round + 1}/{ warmup_rounds}")
                benchmark.set_matrices(array_factory.get_new_array(), array_factory.get_new_array())
                benchmark.multiply_matrices()

            for current_round in range(test_rounds):
                print(f"[{matrix_size} | {method_name}] {current_round + 1}/{ test_rounds}...")
                benchmark.set_matrices(array_factory.get_new_array(), array_factory.get_new_array())

                start = time.time_ns()
                benchmark.multiply_matrices()
                end = time.time_ns()
                time_needed = (end-start) / 1000000

                time_needed_sum += time_needed
                print("Time needed:", time_needed, "ms.")

            print("")
            with open(filename, "at") as f:
                avg_time_per_round = time_needed_sum / test_rounds
                time_needed_tmp_str = "{:20.4f}".format(avg_time_per_round)
                f.write(f"{time_needed_tmp_str} ms | {method_name}\n")


if __name__ == '__main__':

    warmup_rounds = 5
    test_rounds = 10
    #matrix_sizes = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    #                2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    matrix_sizes = [100]
    run_benchmarks(warmup_rounds=warmup_rounds, test_rounds=test_rounds, matrix_sizes=matrix_sizes)




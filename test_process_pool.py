import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

def compute_sum_of_squares(n):
    return sum(i*i for i in range(n))

def measure_cpu_usage():
    return psutil.cpu_percent(interval=1)

def serial_computation(num_elements, num_tasks):
    start_time = time.time()
    results = [compute_sum_of_squares(num_elements) for _ in range(num_tasks)]
    end_time = time.time()
    return results, end_time - start_time

def parallel_computation(num_elements, num_tasks, num_workers=32):
    print(f"Number of workers in ProcessPoolExecutor: {num_workers}")

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_sum_of_squares, num_elements) for _ in range(num_tasks)]
        results = [future.result() for future in as_completed(futures)]
    end_time = time.time()
    return results, end_time - start_time, futures

if __name__ == "__main__":
    NUM_ELEMENTS = 5000000
    NUM_TASKS = 50

    _, serial_time = serial_computation(NUM_ELEMENTS, NUM_TASKS)
    print(f"Serial computation time: {serial_time} seconds")

    cpu_usage = []
    _, parallel_time, futures = parallel_computation(NUM_ELEMENTS, NUM_TASKS)

    if futures:
        for future in futures:
            while not future.done():
                cpu_usage.append(measure_cpu_usage())

    if cpu_usage:
        average_cpu_usage = sum(cpu_usage) / len(cpu_usage)
    else:
        average_cpu_usage = 0

    print(f"Parallel computation time: {parallel_time} seconds")
    print(f"Average CPU usage during parallel computation: {average_cpu_usage}%")

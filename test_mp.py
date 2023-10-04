


import multiprocessing
import time

def worker_function(worker_id):
    print(f"Worker {worker_id} started")
    result = 0
    for i in range(100000000):
        result += i
    print(f"Worker {worker_id} finished, result: {result}")

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {num_cores}")

    processes = []

    for i in range(num_cores):
        process = multiprocessing.Process(target=worker_function, args=(i,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("All processes finished")
    time.sleep(10)  # Sleep for 10 seconds to observe CPU usage in Task Manager

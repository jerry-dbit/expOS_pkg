def print_fork_example():
    code = '''
import os
pid = os.fork()
if pid > 0:
    # Parent process
    print("Parent Process:")
    print(f"Parent PID: {os.getpid()}")
    print(f"Child PID: {pid}")
    print(f"Parent's Parent PID: {os.getppid()}")
else:
    # Child process
    print("Child Process:")
    print(f"Child PID: {os.getpid()}")
    print(f"Parent PID: {os.getppid()}")
'''
    print(code)

def WumpusWorld():
    code1 = '''
# Code 1: Fork + Wait
import os
import time

pid = os.fork()
if pid == 0:
    print(f"Child process (PID: {os.getpid()}) is running.")
    os._exit(0)
else:
    print(f"Parent process (PID: {os.getpid()}) is waiting for the child to terminate.")
    os.waitpid(pid, 0)
    print(f"Parent process detected the termination of child process (PID: {pid}).")
'''

    code2 = '''
# Code 2: File Read/Write
import os

# Write to file
data = "Hello, this is a test.\\n"
with open("example.txt", "w") as f:
    f.write(data)

# Read from file
with open("example.txt", "r") as f:
    content = f.read()
    print("Read from file:", content)
'''

    code3 = '''
# Code 3: Process Info
import os

print(f"Process ID (PID): {os.getpid()}")
print(f"Parent Process ID (PPID): {os.getppid()}")
print(f"Real User ID (UID): {os.getuid()}")
print(f"Effective User ID (EUID): {os.geteuid()}")
print(f"Real Group ID (GID): {os.getgid()}")
print(f"Effective Group ID (EGID): {os.getegid()}")

# Set process group ID
pid = os.getpid()
os.setpgid(pid, pid)
print(f"Process group ID: {os.getpgrp()}")
print(f"Session ID: {os.getsid(pid)}")
print(f"Current working directory: {os.getcwd()}")
'''

    inp = int(input("Enter 1 for fork+wait, 2 for file read/write, 3 for process info: "))
    if inp == 1:
        print(code1)
    elif inp == 2:
        print(code2)
    elif inp == 3:
        print(code3)
    else:
        print("Invalid input. Please enter 1, 2, or 3.")

def SchedulingAlgorithm():
    fcfs_code = '''
class Process:
    def __init__(self, at, bt):
        self.at = at  # Arrival Time
        self.bt = bt  # Burst Time
        self.ct = 0   # Completion Time
        self.wt = 0   # Waiting Time
        self.tat = 0  # Turnaround Time

def FCFS(processes):
    total_wt = 0
    total_tat = 0

    processes[0].ct = processes[0].at + processes[0].bt
    processes[0].tat = processes[0].ct - processes[0].at
    processes[0].wt = processes[0].tat - processes[0].bt

    total_wt += processes[0].wt
    total_tat += processes[0].tat

    for i in range(1, len(processes)):
        if processes[i].at > processes[i - 1].ct:
            processes[i].ct = processes[i].at + processes[i].bt
        else:
            processes[i].ct = processes[i - 1].ct + processes[i].bt
        processes[i].tat = processes[i].ct - processes[i].at
        processes[i].wt = processes[i].tat - processes[i].bt

        total_wt += processes[i].wt
        total_tat += processes[i].tat

    print("Process\\tAT\\tBT\\tCT\\tWT\\tTAT")
    for i, p in enumerate(processes):
        print(f\"P{i+1}\\t{p.at}\\t{p.bt}\\t{p.ct}\\t{p.wt}\\t{p.tat}\")

    print(f\"\\nAverage Waiting Time: {total_wt / len(processes):.2f}\")
    print(f\"Average Turnaround Time: {total_tat / len(processes):.2f}\")

processes = [Process(0, 4), Process(1, 3), Process(2, 5), Process(3, 2)]
FCFS(processes)
'''

    sjf_code = '''class Process:
    def __init__(self, pid, at, bt):
        self.process_id = pid
        self.arrival_time = at
        self.burst_time = bt
        self.completion_time = 0
        self.waiting_time = 0
        self.turnaround_time = 0

def find_sjf(processes):
    # Sort based on burst time
    processes.sort(key=lambda x: x.burst_time)

    total_wt = 0
    total_tat = 0

    print("Original Process Details:")
    print("Process\tArrival\tBurst")
    for p in processes:
        print(f"{p.process_id}\t{p.arrival_time}\t{p.burst_time}")

    # First process
    processes[0].completion_time = processes[0].arrival_time + processes[0].burst_time
    processes[0].turnaround_time = processes[0].completion_time - processes[0].arrival_time
    processes[0].waiting_time = processes[0].turnaround_time - processes[0].burst_time

    total_wt += processes[0].waiting_time
    total_tat += processes[0].turnaround_time

    # Remaining processes
    for i in range(1, len(processes)):
        processes[i].completion_time = processes[i - 1].completion_time + processes[i].burst_time
        processes[i].turnaround_time = processes[i].completion_time - processes[i].arrival_time
        processes[i].waiting_time = processes[i].turnaround_time - processes[i].burst_time

        total_wt += processes[i].waiting_time
        total_tat += processes[i].turnaround_time

    print("\nSJF Scheduling Results:")
    print("Process\tArrival\tBurst\tCompletion\tWaiting\tTurnaround")
    for p in processes:
        print(f"{p.process_id}\t{p.arrival_time}\t{p.burst_time}\t{p.completion_time}\t\t{p.waiting_time}\t{p.turnaround_time}")

    print(f"\nAverage Waiting Time: {total_wt / len(processes):.2f}")
    print(f"Average Turnaround Time: {total_tat / len(processes):.2f}")

# Sample input
processes = [
    Process(1, 0, 4),
    Process(2, 1, 3),
    Process(3, 2, 5),
    Process(4, 3, 2)
]

find_sjf(processes)
'''
    rr_code = '''class Process:
    def __init__(self, pid, bt):
        self.process_id = pid
        self.burst_time = bt
        self.remaining_time = bt
        self.waiting_time = 0
        self.turnaround_time = 0

def find_round_robin(processes, quantum):
    time = 0
    completed = 0
    n = len(processes)

    while completed < n:
        for p in processes:
            if p.remaining_time > 0:
                if p.remaining_time > quantum:
                    time += quantum
                    p.remaining_time -= quantum
                else:
                    time += p.remaining_time
                    p.waiting_time = time - p.burst_time
                    p.turnaround_time = time
                    p.remaining_time = 0
                    completed += 1

    total_wt = sum(p.waiting_time for p in processes)
    total_tat = sum(p.turnaround_time for p in processes)

    print("Process\tBurst\tWaiting\tTurnaround")
    for p in processes:
        print(f"{p.process_id}\t{p.burst_time}\t{p.waiting_time}\t{p.turnaround_time}")

    print(f"\nAverage Waiting Time: {total_wt / n:.2f}")
    print(f"Average Turnaround Time: {total_tat / n:.2f}")

# Sample input
processes = [
    Process(1, 8),
    Process(2, 4),
    Process(3, 6),
    Process(4, 5)
]

quantum = 4
find_round_robin(processes, quantum)
'''
    srtf_code = '''class Process:
    def __init__(self, pid, at, bt):
        self.process_id = pid
        self.arrival_time = at
        self.burst_time = bt
        self.remaining_time = bt
        self.completion_time = 0
        self.waiting_time = 0
        self.turnaround_time = 0

def find_srtf(processes):
    processes.sort(key=lambda x: x.arrival_time)

    total_wt = 0
    total_tat = 0
    time = 0
    completed = 0
    n = len(processes)

    print("Original Process Details:")
    print("Process\tArrival\tBurst")
    for p in processes:
        print(f"{p.process_id}\t{p.arrival_time}\t{p.burst_time}")

    while completed < n:
        shortest = None
        min_remaining_time = float('inf')

        for i in range(n):
            if (processes[i].arrival_time <= time and
                processes[i].remaining_time > 0 and
                processes[i].remaining_time < min_remaining_time):
                min_remaining_time = processes[i].remaining_time
                shortest = i

        if shortest is None:
            time += 1
            continue

        processes[shortest].remaining_time -= 1

        if processes[shortest].remaining_time == 0:
            completed += 1
            finish_time = time + 1
            p = processes[shortest]
            p.completion_time = finish_time
            p.turnaround_time = p.completion_time - p.arrival_time
            p.waiting_time = p.turnaround_time - p.burst_time
            total_wt += p.waiting_time
            total_tat += p.turnaround_time

        time += 1

    print("\nSRTF Scheduling Results:")
    print("Process Arrival Burst Completion Waiting Turnaround")
    for p in processes:
        print(f"{p.process_id}\t{p.arrival_time}\t{p.burst_time}\t{p.completion_time}\t\t{p.waiting_time}\t{p.turnaround_time}")

    print(f"\nAverage Waiting Time: {total_wt / n:.2f}")
    print(f"Average Turnaround Time: {total_tat / n:.2f}")

# Sample input
processes = [
    Process(1, 0, 4),
    Process(2, 1, 3),
    Process(3, 2, 5),
    Process(4, 3, 2)
]

find_srtf(processes)
'''

    inp = int(input("Enter 1 for FCFS, 2 for SJF, 3 for Round Robin, 4 for SRTF: "))
    if inp == 1:
        print(fcfs_code)
    elif inp == 2:
        print(sjf_code)
    elif inp == 3:
        print(rr_code)
    elif inp == 4:
        print(srtf_code)
    else:
        print("Invalid input. Please enter 1 to 4.")


def ProducerConsumer():
    code = '''
import threading
import time
import random
# Shared buffer size
BUFFER_SIZE = 10
buffer = []
# Semaphores to control access to the buffer
empty = threading.Semaphore(BUFFER_SIZE) # Semaphore to
count the empty slots
full = threading.Semaphore(0) # Semaphore to count the full slots
mutex = threading.Semaphore(1) # Mutex to ensure mutual
exclusion while accessing the buffer
# Number of items to produce and consume
MAX_ITERATIONS = 20
# Producer thread
def producer():
 for _ in range(MAX_ITERATIONS):
 item = random.randint(1, 100) # Produce a random item
 empty.acquire() # Wait if there are no empty slots
 mutex.acquire() # Ensure mutual exclusion while accessing
the buffer

 buffer.append(item) # Add item to the buffer
 print(f"Produced: {item}, Buffer: {buffer}")

 mutex.release() # Release mutex after updating buffer
 full.release() # Signal that a new item is produced

 time.sleep(random.uniform(0.1, 1)) # Simulate production
time
# Consumer thread
def consumer():
 for _ in range(MAX_ITERATIONS):
 full.acquire() # Wait if there are no full slots
 mutex.acquire() # Ensure mutual exclusion while accessing
the buffer

 item = buffer.pop(0) # Consume an item from the buffer
 print(f"Consumed: {item}, Buffer: {buffer}")

 mutex.release() # Release mutex after updating buffer
 empty.release() # Signal that a slot is now empty

 time.sleep(random.uniform(0.1, 1)) # Simulate consumption
time
# Create producer and consumer threads
producer_thread = threading.Thread(target=producer)
consumer_thread = threading.Thread(target=consumer)
# Start the threads
producer_thread.start()
consumer_thread.start()
# Join the threads to the main thread
producer_thread.join()
consumer_thread.join()
'''
    print(code)

def ProducerConsumer():
    print('''
import threading
import time
import random

# Shared buffer size
BUFFER_SIZE = 10
buffer = []

# Semaphores to control access to the buffer
empty = threading.Semaphore(BUFFER_SIZE)
full = threading.Semaphore(0)
mutex = threading.Semaphore(1)

MAX_ITERATIONS = 20

def producer():
    for _ in range(MAX_ITERATIONS):
        item = random.randint(1, 100)
        empty.acquire()
        mutex.acquire()

        buffer.append(item)
        print(f"Produced: {item}, Buffer: {buffer}")

        mutex.release()
        full.release()
        time.sleep(random.uniform(0.1, 1))

def consumer():
    for _ in range(MAX_ITERATIONS):
        full.acquire()
        mutex.acquire()

        item = buffer.pop(0)
        print(f"Consumed: {item}, Buffer: {buffer}")

        mutex.release()
        empty.release()
        time.sleep(random.uniform(0.1, 1))

producer_thread = threading.Thread(target=producer)
consumer_thread = threading.Thread(target=consumer)

producer_thread.start()
consumer_thread.start()

producer_thread.join()
consumer_thread.join()
''')

def Banker():
    print('''
# Banker's Algorithm to demonstrate Deadlock Avoidance
class BankersAlgorithm:
    def __init__(self, processes, resources):
        self.processes = processes
        self.resources = resources
        self.available = [0] * resources
        self.maximum = []
        self.allocation = []
        self.need = []

    def set_data(self, available, maximum, allocation):
        self.available = available
        self.maximum = maximum
        self.allocation = allocation
        self.need = [[self.maximum[i][j] - self.allocation[i][j] for j in range(self.resources)] for i in range(self.processes)]

    def is_safe(self):
        work = self.available[:]
        finish = [False] * self.processes
        safe_sequence = []

        while len(safe_sequence) < self.processes:
            progress = False
            for i in range(self.processes):
                if not finish[i] and all(self.need[i][j] <= work[j] for j in range(self.resources)):
                    work = [work[j] + self.allocation[i][j] for j in range(self.resources)]
                    finish[i] = True
                    safe_sequence.append(i)
                    progress = True
                    break
            if not progress:
                return False, []
        return True, safe_sequence

    def request_resources(self, process_id, request):
        if any(request[i] > self.need[process_id][i] for i in range(self.resources)):
            print("Error: Process has exceeded its maximum claim.")
            return False
        if any(request[i] > self.available[i] for i in range(self.resources)):
            print("Error: Resources not available.")
            return False

        temp_available = self.available[:]
        temp_allocation = [row[:] for row in self.allocation]
        temp_need = [row[:] for row in self.need]

        for i in range(self.resources):
            temp_available[i] -= request[i]
            temp_allocation[process_id][i] += request[i]
            temp_need[process_id][i] -= request[i]

        self.set_data(temp_available, self.maximum, temp_allocation)
        is_safe, _ = self.is_safe()

        if is_safe:
            self.set_data(temp_available, self.maximum, temp_allocation)
            print(f"Resources allocated to process {process_id}.")
            return True
        else:
            print("Error: Allocation leads to an unsafe state. Request denied.")
            return False

# Example Data
processes = 5
resources = 3
available = [3, 3, 2]

maximum = [
    [7, 5, 3],
    [3, 2, 2],
    [9, 0, 2],
    [2, 2, 2],
    [4, 3, 3]
]

allocation = [
    [0, 1, 0],
    [2, 0, 0],
    [3, 0, 2],
    [2, 1, 1],
    [0, 0, 2]
]

banker = BankersAlgorithm(processes, resources)
banker.set_data(available, maximum, allocation)
request = [1, 0, 2]
banker.request_resources(1, request)

is_safe, safe_sequence = banker.is_safe()
if is_safe:
    print("System is in a safe state.")
    print(f"Safe sequence: {safe_sequence}")
else:
    print("System is not in a safe state.")
''')

def memoryAllocation():
    code1 = '''
def best_fit(block_size, process_size):
    allocation = [-1] * len(process_size)

    for i in range(len(process_size)):
        best_idx = -1
        for j in range(len(block_size)):
            if block_size[j] >= process_size[i]:
                if best_idx == -1 or block_size[j] < block_size[best_idx]:
                    best_idx = j
        if best_idx != -1:
            allocation[i] = best_idx
            block_size[best_idx] -= process_size[i]

    print("Process No.\\tProcess Size\\tBlock No.")
    for i in range(len(process_size)):
        print(f"{i+1}\\t\\t{process_size[i]}\\t\\t{allocation[i]+1 if allocation[i] != -1 else 'Not Allocated'}")

block_size = [100, 500, 200, 300, 600]
process_size = [212, 417, 112, 426]
best_fit(block_size, process_size)
'''

    code2 = ''' 
def worst_fit(block_size, process_size):
    allocation = [-1] * len(process_size)

    for i in range(len(process_size)):
        worst_idx = -1
        for j in range(len(block_size)):
            if block_size[j] >= process_size[i]:
                if worst_idx == -1 or block_size[j] > block_size[worst_idx]:
                    worst_idx = j
        if worst_idx != -1:
            allocation[i] = worst_idx
            block_size[worst_idx] -= process_size[i]

    print("Process No.\\tProcess Size\\tBlock No.")
    for i in range(len(process_size)):
        print(f"{i+1}\\t\\t{process_size[i]}\\t\\t{allocation[i]+1 if allocation[i] != -1 else 'Not Allocated'}")

block_size = [100, 500, 200, 300, 600]
process_size = [212, 417, 112, 426]
worst_fit(block_size, process_size)
'''

    code3 = '''
def first_fit(block_size, process_size):
    allocation = [-1] * len(process_size)

    for i in range(len(process_size)):
        for j in range(len(block_size)):
            if block_size[j] >= process_size[i]:
                allocation[i] = j
                block_size[j] -= process_size[i]
                break

    print("Process No.\\tProcess Size\\tBlock No.")
    for i in range(len(process_size)):
        print(f"{i+1}\\t\\t{process_size[i]}\\t\\t{allocation[i]+1 if allocation[i] != -1 else 'Not Allocated'}")

block_size = [100, 500, 200, 300, 600]
process_size = [212, 417, 112, 426]
first_fit(block_size, process_size)
'''

    inp = int(input("Enter 1 for Best fit, 2 for Worst fit, 3 for First fit: "))
    if inp == 1:
        print(code1)
    elif inp == 2:
        print(code2)
    elif inp == 3:
        print(code3)
    else:
        print("Invalid input. Please enter 1, 2, or 3.")


def FileAllocation():
    Code1 = '''
# Sequential Allocation
class FileSystem:
    def __init__(self, memory_size):
        self.memory = [0] * memory_size

    def sequential_allocation(self, file_size):
        for i in range(len(self.memory) - file_size + 1):
            if all(self.memory[i + j] == 0 for j in range(file_size)):
                for j in range(file_size):
                    self.memory[i + j] = 1
                return list(range(i, i + file_size))
        return None

    def show_memory(self):
        print("Memory State:", self.memory)

fs = FileSystem(memory_size=20)
print("Initial Memory State:")
fs.show_memory()

print("\\nSequential Allocation (File Size 4):")
blocks = fs.sequential_allocation(4)
print("Allocated blocks:" if blocks else "Not enough space", blocks)
fs.show_memory()
'''

    Code2 = '''
# Indexed Allocation
class FileSystem:
    def __init__(self, memory_size):
        self.memory = [0] * memory_size

    def indexed_allocation(self, file_size):
        index_block = None
        for i in range(len(self.memory)):
            if self.memory[i] == 0:
                index_block = i
                break
        if index_block is None:
            return None

        index_pointer = []
        for i in range(len(self.memory)):
            if self.memory[i] == 0 and len(index_pointer) < file_size:
                self.memory[i] = 1
                index_pointer.append(i)

        if len(index_pointer) == file_size:
            self.memory[index_block] = 1
            return [index_block] + index_pointer
        return None

    def show_memory(self):
        print("Memory State:", self.memory)

fs = FileSystem(20)
print("Initial Memory:")
fs.show_memory()
blocks = fs.indexed_allocation(5)
print("Allocated blocks:" if blocks else "Not enough space", blocks)
fs.show_memory()
'''

    Code3 = '''
# Linked Allocation
class FileSystem:
    def __init__(self, memory_size):
        self.memory = [0] * memory_size

    def linked_allocation(self, file_size):
        file_blocks = []
        for i in range(len(self.memory)):
            if self.memory[i] == 0:
                self.memory[i] = 1
                file_blocks.append(i)
                if len(file_blocks) == file_size:
                    break
        return file_blocks if len(file_blocks) == file_size else None

    def show_memory(self):
        print("Memory State:", self.memory)

fs = FileSystem(20)
print("Initial Memory:")
fs.show_memory()
blocks = fs.linked_allocation(3)
print("Allocated blocks:" if blocks else "Not enough space", blocks)
fs.show_memory()
'''

    inp = int(input("Enter 1 for Sequential, 2 for Indexed, 3 for Linked Allocation: "))
    if inp == 1:
        print(Code1)
    elif inp == 2:
        print(Code2)
    elif inp == 3:
        print(Code3)
    else:
        print("Invalid input.")


def PageReplacement():
    code1 = '''
# FIFO Page Replacement
def fifo_page_replacement(pages, frames):
    memory = []
    page_faults = 0

    for page in pages:
        if page not in memory:
            page_faults += 1
            if len(memory) < frames:
                memory.append(page)
            else:
                memory.pop(0)
                memory.append(page)
        print(f"Page: {page} → Memory: {memory}")
    
    print(f"\\nTotal Page Faults (FIFO): {page_faults}")

# Example
pages = [1, 2, 3, 4, 2, 1, 5, 6, 2, 1, 2, 3, 7, 6, 3, 2, 1, 2, 3, 6]
fifo_page_replacement(pages, frames=3)
'''

    code2 = '''
# LRU Page Replacement
def lru_page_replacement(pages, frames):
    memory = []
    page_faults = 0

    for page in pages:
        if page not in memory:
            page_faults += 1
            if len(memory) < frames:
                memory.append(page)
            else:
                memory.pop(0)
                memory.append(page)
        else:
            memory.remove(page)
            memory.append(page)
        print(f"Page: {page} → Memory: {memory}")
    
    print(f"\\nTotal Page Faults (LRU): {page_faults}")

# Example
pages = [1, 2, 3, 4, 2, 1, 5, 6, 2, 1, 2, 3, 7, 6, 3, 2, 1, 2, 3, 6]
lru_page_replacement(pages, frames=3)
'''

    code3 = '''
# Optimal Page Replacement
def optimal_page_replacement(pages, frames):
    memory = []
    page_faults = 0

    for i in range(len(pages)):
        page = pages[i]
        if page not in memory:
            page_faults += 1
            if len(memory) < frames:
                memory.append(page)
            else:
                future = pages[i+1:]
                indexes = []
                for m in memory:
                    if m in future:
                        indexes.append(future.index(m))
                    else:
                        indexes.append(float('inf'))
                victim = indexes.index(max(indexes))
                memory[victim] = page
        print(f"Page: {page} → Memory: {memory}")

    print(f"\\nTotal Page Faults (Optimal): {page_faults}")

# Example
pages = [1, 2, 3, 4, 2, 1, 5, 6, 2, 1, 2, 3, 7, 6, 3, 2, 1, 2, 3, 6]
optimal_page_replacement(pages, frames=3)
'''

    inp = int(input("Enter 1 for FIFO, 2 for LRU, 3 for Optimal: "))
    if inp == 1:
        print(code1)
    elif inp == 2:
        print(code2)
    elif inp == 3:
        print(code3)
    else:
        print("Invalid input. Please enter 1, 2 or 3.")

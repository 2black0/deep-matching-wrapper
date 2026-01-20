import psutil

# Get CPU usage (percentage)
cpu_usage = psutil.cpu_percent(interval=1)
print(f"CPU Usage: {cpu_usage}%")

# Get memory usage
memory_info = psutil.virtual_memory()
print(f"Total Memory: {memory_info.total / (1024**3):.2f} GB")
print(f"Used Memory: {memory_info.used / (1024**3):.2f} GB")
print(f"Memory Usage Percentage: {memory_info.percent}%")

# Iterate over all running processes
for process in psutil.process_iter(['pid', 'name', 'username']):
    print(f"PID: {process.info['pid']}, Name: {process.info['name']}, User: {process.info['username']}")

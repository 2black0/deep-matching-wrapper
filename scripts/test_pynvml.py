import pynvml

try:
    pynvml.nvmlInit()
    print("NVML Initialized Successfully")
    
    driver_version = pynvml.nvmlSystemGetDriverVersion()
    print(f"Driver Version: {driver_version}")
    
    deviceCount = pynvml.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        print(f"Device {i}: {name}")
        
        # Get memory info
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"  Memory Total: {info.total / 1024**2:.2f} MB")
        print(f"  Memory Free: {info.free / 1024**2:.2f} MB")
        print(f"  Memory Used: {info.used / 1024**2:.2f} MB")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    try:
        pynvml.nvmlShutdown()
        print("NVML Shutdown Successfully")
    except:
        pass

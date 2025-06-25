import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

if not all(df['compare'] == 1):
    print("Error: Some CPU and GPU results differ!")

grouped = df.groupby("matrixSize").agg({
    "cpuTime": "mean",
    "gpuTime": "mean",
    "cpuSpeedUp": "mean"
}).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(grouped["matrixSize"], grouped["cpuTime"], label="CPU Time (ms)", marker="o")
plt.plot(grouped["matrixSize"], grouped["gpuTime"], label="GPU Time (ms)", marker="o")
plt.xlabel("Matrix Size (N × N)")
plt.ylabel("Time (ms)")
plt.title("CPU vs GPU Time per Matrix Size")
plt.legend()
plt.grid(True)
plt.savefig("cpu_gpu_time.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(grouped["matrixSize"], grouped["cpuSpeedUp"], label="CPU Speedup", color='green', marker="o")
plt.xlabel("Matrix Size (N × N)")
plt.ylabel("Speedup (GPU Time / CPU Time)")
plt.title("Speedup of CPU over GPU")
plt.grid(True)
plt.savefig("cpu_speedup.png")
plt.show()


import pandas as pd
import json
import matplotlib.pyplot as plt

# Load the JSON data
with open('../../results.json') as f:
    data = json.load(f)

# Extract relevant benchmarks (e.g., 'BM_ImpactOfBatching')
df = pd.DataFrame(data['benchmarks'])
df_filtered = df[df['name'].str.contains('BM_ImpactOfBatching')]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df_filtered['name'], df_filtered['cpu_time'], marker='o')
plt.title('Performance vs Batch Size')
plt.xlabel('Benchmark Name')
plt.ylabel('CPU Time (ns)')
plt.grid(True)
plt.show()
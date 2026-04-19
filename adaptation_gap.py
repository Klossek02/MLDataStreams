import pandas as pd
import matplotlib.pyplot as plt

dump_file = 'dumpFile.csv'
df = pd.read_csv(dump_file)

df['classifications correct (percent)'] = pd.to_numeric(df['classifications correct (percent)'], errors='coerce')

x_instances = df['learning evaluation instances'].values

error_rate = 100.0 - df['classifications correct (percent)'].values

plt.figure(figsize=(10, 5))
plt.plot(x_instances, error_rate, color='purple', label='HAT baseline model', linewidth=2)

plt.axvline(x=500000, color='red', linestyle='--', label='Artificial concept drift')

plt.title('Error of the model over time - adaptation gap analysis')
plt.xlabel('Number of processed instances')
plt.ylabel('Classification error (%)')
plt.legend()
plt.grid(True)
plt.show()
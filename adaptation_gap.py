import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dumpFile.csv', skipinitialspace=True)

df.columns = df.columns.str.strip()

print("Columns:", df.columns.tolist())

col_instances = 'learning evaluation instances'
col_accuracy = 'classifications correct (percent)'


df[col_instances] = pd.to_numeric(df[col_instances], errors='coerce')
df[col_accuracy] = pd.to_numeric(df[col_accuracy], errors='coerce')


df = df.dropna(subset=[col_instances, col_accuracy])

df['error'] = 100 - df[col_accuracy]

plt.figure(figsize=(10, 5))

# model line
plt.plot(df[col_instances], df['error'], color='purple', linewidth=1.5, label='HAT baseline model')

# drift line 
plt.axvline(x=500000, color='red', linestyle='--', linewidth=2, label='Abrupt concept drift')


plt.xlim(0, 1000000)


plt.title('Error of the model over time - adaptation gap analysis')
plt.xlabel('No. of processed instances')
plt.ylabel('Classification error (%)')
plt.legend()
plt.grid(True, alpha=0.6)

plt.show()
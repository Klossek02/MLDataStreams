import pandas as pd
import matplotlib.pyplot as plt
import os

# MOA dump for HAT on Agrawal sudden (function 1 -> 3), 200,000 instances,
# drift at 100,000 — consistent with the synthetic streams in Section 5.
input_csv = 'adaptation_gap_HT.csv'
output_png = os.path.join('latex', 'report', 'figures', 'adaptation_gap.png')

total_instances = 200_000
drift_point = 100_000

df = pd.read_csv(input_csv, skipinitialspace=True)
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
plt.plot(df[col_instances], df['error'], color='purple', linewidth=1.5,
         label='HT baseline model')

# drift line
plt.axvline(x=drift_point, color='red', linestyle='--', linewidth=2,
            label='Abrupt concept drift')

plt.xlim(0, total_instances)

plt.title('Error of the model over time - adaptation gap analysis')
plt.xlabel('No. of processed instances')
plt.ylabel('Classification error (%)')
plt.legend()
plt.grid(True, alpha=0.6)

plt.tight_layout()
plt.savefig(output_png, dpi=150)
print(f"Saved figure: {output_png}")
plt.show()

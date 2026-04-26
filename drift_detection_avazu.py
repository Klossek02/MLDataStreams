import pandas as pd
import matplotlib.pyplot as plt
from river import drift
import time

input_file = './avazu/data/avazu_train_1M.csv' 

df = pd.read_csv(input_file)
clicks = df['click'].values

adwin = drift.ADWIN()


drifts_detected = []
ctr_history = []
window_size = 10000  # local CTR from last 10k displays
current_window_clicks = 0

start_time = time.time()


for i, click in enumerate(clicks):

    adwin.update(click)
    
    current_window_clicks += click
    if i >= window_size:
        current_window_clicks -= clicks[i - window_size]
    
    if i > 0 and i % 5000 == 0:
        current_ctr = (current_window_clicks / min(i, window_size)) * 100
        ctr_history.append((i, current_ctr))
    
    # if ADWIN detects a drift
    if adwin.drift_detected:
        print(f"The drift has been detected in instance: {i}")
        drifts_detected.append(i)

end_time = time.time()
print(f"\n{len(drifts_detected)} drift points have been discovered.")
print(f"Execution time: {round(end_time - start_time, 2)} seconds.")


x_vals = [x[0] for x in ctr_history]
y_vals = [x[1] for x in ctr_history]

plt.figure(figsize=(12, 6))
plt.plot(x_vals, y_vals, label='Local CTR (%)', color='blue', linewidth=1.5)


for i, drift_idx in enumerate(drifts_detected):
    if i == 0:
        plt.axvline(x=drift_idx, color='red', linestyle='--', alpha=0.7, label='ADWIN drift alert')
    else:
        plt.axvline(x=drift_idx, color='red', linestyle='--', alpha=0.7)

plt.title('Concept drift detection on Avazu using ADWIN')
plt.xlabel('Instance number in the stream')
plt.ylabel('Click-through rate (%)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

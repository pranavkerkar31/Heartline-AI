import wfdb
import matplotlib.pyplot as plt

# Read record (without extension)
record = wfdb.rdrecord("record_name")

# Plot ECG
wfdb.plot_wfdb(record=record, title="ECG Signal")
plt.show()

import numpy as np
import os
import matplotlib.pyplot as plt
# Import the saved data and plot it

script_dir = os.path.dirname(os.path.abspath(__file__))

output_file_observer = script_dir + '/Xs_data_observer.npy'
output_file = script_dir + '/Xs_data_.npy'

loaded_Xs_observer = np.load(output_file_observer)
loaded_Xs = np.load(output_file)

print(f"Loaded state trajectory Xs_observer from {output_file_observer}")
print(f"Loaded state trajectory Xs from {output_file}")

dim_x = 6
time_steps = np.arange(loaded_Xs.shape[0])

# Plot the loaded data
plt.figure(figsize=(12, 8))

for i in range(dim_x):
    plt.subplot(dim_x, 1, i + 1)
    plt.step(time_steps, loaded_Xs[:, i], where='post', label=f"Xs[{i}]")
    plt.step(time_steps, loaded_Xs_observer[:, i], where='post', linestyle='--', label=f"Xs_observer[{i}]")
    plt.ylabel(f"x[{i}]")
    plt.grid()
    if i == 0:
        plt.title("Loaded State Trajectories")
    if i == dim_x - 1:
        plt.xlabel("Time step")
    plt.legend()

plt.tight_layout()
plt.show()

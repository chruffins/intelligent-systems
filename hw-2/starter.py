import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Izhikevich model (Regular Spiker)
a = 0.02
b = 0.2
c = -65
d = 8
I_step = 10  # Step input

# Simulation parameters
epsilon = 0.1  # Time step
T = 1000  # Total simulation time in ms
time = np.arange(0, T, epsilon)
V = np.zeros_like(time)
U = np.zeros_like(time)
V[0] = c
spikes = np.zeros_like(time)

# Generate input time series
input_current = np.zeros_like(time)
input_current[time > 50] = I_step

# Simulate the Izhikevich model for 1,000 ms
for i in range(1, len(time)):
    I = I_step if i*epsilon > 50 else 0  # Apply step input after 50ms
    V[i] = V[i-1] + epsilon * (0.04 * V[i-1]**2 + 5*V[i-1] + 140 - U[i-1] + I)
    U[i] = U[i-1] + epsilon * (a * (b*V[i-1] - U[i-1]))
    
    # Log spike if any
    if V[i] >= 30:
        V[i] = c
        U[i] = U[i] + d
        spikes[i] = 1

    
# Plot the graphs
fig, px = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1, 0.5]})

# Membrane potential
px[0].plot(time, V, color='blue')
px[0].set_title('Izhikevich Neuron Model')
px[0].set_ylabel('Membrane potential (V)')
px[0].set_xlim(-50, 1050)


# Spike raster
spike_times = time[spikes > 0.9]
px[1].eventplot(spike_times, color='black')
px[1].set_xlim(-50, 1050)
px[1].set_ylim(0.5, 1.5)
px[1].set_yticks([])
px[1].set_ylabel('Spikes')

# Input
px[2].plot(time, input_current, color='blue')
px[2].set_xlabel('Time (ms)')
px[2].set_ylabel('Input (I)')
px[2].set_ylim(0, I_step*1.2)
px[2].set_xlim(-50, 1050)


plt.tight_layout()
plt.show()

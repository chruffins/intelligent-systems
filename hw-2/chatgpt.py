import numpy as np
import matplotlib.pyplot as plt

# Define the Izhikevich neuron parameters
a = 0.02
b = 0.2
c = -65
d = 8

# Simulation parameters
total_time_steps = 1000
input_range = np.arange(0, 41, 1)
spike_rates = []

# Define the Izhikevich neuron simulation function
def izhikevich_neuron_simulation(I, total_time_steps):
    v = -65  # Initial membrane potential
    u = b * v  # Initial recovery variable

    spike_count = 0  # Counter for spikes

    for t in range(total_time_steps):
        if v >= 30:  # Spike condition
            if t > 200:
                spike_count += 1
            v = c
            u += d

        # Update membrane potential and recovery variable
        dv = (0.04 * v**2 + 5 * v + 140 - u + I)  # Membrane potential change
        du = a * (b * v - u)  # Recovery variable change
        v += dv
        u += du

    # Calculate spiking rate
    spiking_rate = spike_count / (total_time_steps - 200)  # Use the last 800 time-steps
    return spiking_rate

# Simulate and collect spiking rates for different input currents
for I in input_range:
    spiking_rate = izhikevich_neuron_simulation(I, total_time_steps)
    spike_rates.append(spiking_rate)

# Plot spiking rate vs. input current
plt.figure(figsize=(10, 6))
plt.plot(input_range, spike_rates, marker='o', linestyle='-')
plt.xlabel('Input Current (I)')
plt.ylabel('Spiking Rate')
plt.title('Spiking Rate vs. Input Current for Regular Spiking Izhikevich Neuron')
plt.grid(True)
plt.show()

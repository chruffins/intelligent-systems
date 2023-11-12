# This code was written with the help of the Izhikevich Python starter code.
# Chris Lee

import numpy as np
import matplotlib.pyplot as plt

SIMULATION_TIME = 1000
EPSILON = 0.1

REGULAR_SPIKING_PARAMS = (0.02, 0.2, -65, 8)
FAST_SPIKING_PARAMS = (0.1, 0.2, -65, 2)

def fast_or_regular(params) -> str:
    if params == REGULAR_SPIKING_PARAMS:
        return "Regular"
    elif params == FAST_SPIKING_PARAMS:
        return "Fast"

def red_or_blue(params) -> str:
    if params == REGULAR_SPIKING_PARAMS:
        return 'blue'
    else:
        return 'red'

def run_simulation(neuron_params: tuple[float, float, float, float], external_input: float, subplot: object | None = None) -> float:
    """
    external_input: I, the external current put into the neuron.
    subplot: a subplot if you want the result to be plotted or None if not

    Returns R, the number of spikes within the last 800 steps. 
    """
    # declaration and initialization of variables we need to use
    a, b, c, d = neuron_params

    time = np.arange(0, SIMULATION_TIME, EPSILON)
    v = np.zeros_like(time) # keep track of v for each discrete time
    u = np.zeros_like(time) # keep track of u for each discrete time
    spikes = np.zeros_like(time) # recording spikes
    v[0] = c

    for i in range(1, len(time)): 
        v[i] = v[i-1] + EPSILON * (0.04 * v[i-1]**2 + 5*v[i-1] + 140 - u[i-1] + external_input)
        u[i] = u[i-1] + EPSILON * (a * (b*v[i-1] - u[i-1]))

        if v[i] >= 30:
            v[i] = c
            u[i] = u[i] + d
            spikes[i] = 1

    r = len(time[2000:][spikes[2000:] > 0.9]) / 800
    
    if subplot is not None:
        subplot.plot(time, v, color=red_or_blue(neuron_params))
        subplot.set_title(f"Izhikevich {fast_or_regular(neuron_params)} Spiking Neuron Model for I={external_input}")
        subplot.set_ylabel('Membrane Potential (V)')
        subplot.set_xlabel('Time')
        subplot.set_xlim(-50, 1050)
    
    return r

def problem1():
    # need to plot I = 1, 10, 20, 30, and 40
    can_plot: dict[float, int] = {
        1.0: 0,
        10.0: 1,
        20.0: 2,
        30.0: 3,
        40.0: 4
    }

    # creating the window for the voltage-time graphs
    _, px = plt.subplots(5, 1, figsize=(10, 10))

    i_series = np.arange(1, 41, 1)
    r_series = np.zeros_like(i_series, dtype=np.float64)
    
    # run the simulations, the plots will be filled out by the appropriate simulations
    for i in range(40):
        fi = float(i+1) # the number actually used for hash table and input

        r = run_simulation(REGULAR_SPIKING_PARAMS, fi, can_plot.get(fi) is not None and px[can_plot[fi]] or None) # ugly but works
        r_series[i] = r
    
    plt.tight_layout()

    # now creating the window for the input-spike rate graph
    plt.figure()
    plt.plot(i_series, r_series, color='blue')
    plt.title("Relationship between Spike Rate and Input Current")
    plt.xlabel("Input Current")
    plt.ylabel("Spike Rate")
    plt.xlim(0,40)

    plt.show()

def problem2():
    # need to plot I = 1, 10, 20, 30, and 40
    can_plot: dict[float, int] = {
        1.0: 0,
        10.0: 1,
        20.0: 2,
        30.0: 3,
        40.0: 4
    }

    # creating the window for the voltage-time graphs
    _, px = plt.subplots(5, 1, figsize=(10, 10))

    i_series = np.arange(1, 41, 1)
    rregular_series = np.zeros_like(i_series, dtype=np.float64)
    rfast_series = np.zeros_like(i_series, dtype=np.float64)
    
    # run the simulations, the plots will be filled out by the appropriate simulations
    for i in range(40):
        fi = float(i+1) # the number actually used for hash table and input

        r = run_simulation(FAST_SPIKING_PARAMS, fi, can_plot.get(fi) is not None and px[can_plot[fi]] or None) # ugly but works
        rregular = run_simulation(REGULAR_SPIKING_PARAMS, fi, None) # since we need to plot this with the fast spiking neuron
        rfast_series[i] = r
        rregular_series[i] = rregular
    
    plt.tight_layout()

    # now creating the window for the input-spike rate graph
    plt.figure()
    plt.plot(i_series, rfast_series, color='red', label="Fast Spiking")
    plt.plot(i_series, rregular_series, color='blue', label="Regular Spiking")
    plt.title("Relationship between Spike Rate and Input Current")
    plt.xlabel("Input Current")
    plt.ylabel("Spike Rate")
    plt.xlim(0,40)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    problem1()
    problem2()
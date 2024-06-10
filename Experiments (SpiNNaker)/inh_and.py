import os
import pickle

import numpy as np
import spynnaker.pyNN as sim
from matplotlib import pyplot as plt

# Neuron parameters
global_params = {"min_delay": 1.0, "sim_time": 50.0}
neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1, "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}


if __name__ == '__main__':
    # --- Simulation ---
    sim.setup(global_params["min_delay"])

    # --- Predefined objects ---
    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])  # Standard connection
    n_bits = 10

    # -- Network architecture --
    # - Spike injectors -
    op_times = range(10, int(global_params["sim_time"] - 10), 1)
    input_times = [np.random.randint(10, int(global_params["sim_time"] - 10), int(global_params["sim_time"] * 0.5)) for i in range(n_bits)]

    op_pop = sim.Population(1, sim.SpikeSourceArray(spike_times=op_times))
    input_pop = [sim.Population(1, sim.SpikeSourceArray(spike_times=input_times[i])) for i in range(n_bits)]

    # - Populations -
    delay_pop = sim.Population(1, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="delay")
    not_pop = sim.Population(n_bits, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="not")
    nor_pop = sim.Population(1, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="nor")

    # - Connections -
    # OP
    sim.Projection(op_pop, delay_pop, sim.OneToOneConnector(), std_conn)
    for i in range(n_bits):
        sim.Projection(op_pop, sim.PopulationView(not_pop, [i]), sim.OneToOneConnector(), std_conn)
        sim.Projection(input_pop[i], sim.PopulationView(not_pop, [i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

    # NOT
    sim.Projection(delay_pop, nor_pop, sim.OneToOneConnector(), std_conn)
    sim.Projection(not_pop, nor_pop, sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")

    # -- Recording --
    nor_pop.record(["spikes"])
    not_pop.record(["spikes"])

    # -- Run simulation --
    sim.run(global_params["sim_time"])

    # -- Get data from the simulation --
    nor_data = nor_pop.get_data().segments[0]
    not_data = not_pop.get_data().segments[0]

    # - End simulation -
    sim.end()

    # --- Saving test ---
    save_array = [nor_data, not_data]
    test_name = os.path.basename(__file__).split('.')[0]

    cwd = os.getcwd()
    if not os.path.exists(cwd + "/experiments/"):
        os.mkdir(cwd + "/experiments/")

    i = 1
    while os.path.exists(cwd + "/experiments/" + test_name + "_" + str(i) + ".pickle"):
        i += 1

    filename = test_name + "_" + str(i)

    with open("experiments/" + filename + '.pickle', 'wb') as handle:
        pickle.dump(save_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # --- Saving plot ---
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams['font.size'] = '4'
    plt.rcParams["figure.figsize"] = (4, 0.75)

    # Spikes
    plt.plot(op_times, [-1] * len(op_times), 'o', markersize=0.5, color='tab:blue')
    for i in range(n_bits):
        plt.plot(input_times[i], [i] * len(input_times[i]), 'o', markersize=0.5, color='tab:orange')
    plt.plot(np.array(nor_data.spiketrains[0]) - 2, [n_bits] * len(nor_data.spiketrains[0]), 'o', markersize=0.5, color='tab:green')

    plt.xlabel("Time (ms)")
    plt.ylabel("Neurons")
    plt.yticks(range(-2, n_bits + 2), ["", "OP"] + ["Input " + str(i) for i in range(n_bits)] + ["NOR (-2)", ""])

    plt.tight_layout()
    plt.savefig("experiments/" + filename + '.png', transparent=False, facecolor='white', edgecolor='black')
    plt.show()

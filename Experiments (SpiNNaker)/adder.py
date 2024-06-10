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
    n_bits = 3

    # -- Network architecture --
    # - Spike injectors -
    op_times = range(10, int(global_params["sim_time"] - 10), 1)
    a_times = [np.random.randint(10, int(global_params["sim_time"] - 10), int(global_params["sim_time"] * 0.2)) for i in range(n_bits)]
    b_times = [np.random.randint(10, int(global_params["sim_time"] - 10), int(global_params["sim_time"] * 0.2)) for i in range(n_bits)]
    #b_times = [[] for i in range(n_bits)]

    op_pop = sim.Population(1, sim.SpikeSourceArray(spike_times=op_times))
    a_pop = [sim.Population(1, sim.SpikeSourceArray(spike_times=a_times[i])) for i in range(n_bits)]
    b_pop = [sim.Population(1, sim.SpikeSourceArray(spike_times=b_times[i])) for i in range(n_bits)]

    # - Populations -
    if n_bits > 1:
        op_delay_pop = sim.Population(5 * (n_bits - 1), sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="op_delay")
    else:
        op_delay_pop = sim.Population(1, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="op_delay")
    a_delay_pop = [sim.Population(2 + 5 * i, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="a_delay") for i in range(n_bits - 1)]
    b_delay_pop = [sim.Population(2 + 5 * i, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="b_delay") for i in range(n_bits - 1)]

    ha_pop = sim.Population(6, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]})
    if n_bits > 1:
        fa_pop = sim.Population(18 * (n_bits - 1), sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]})

    # - Connections -
    # OP chain
    sim.Projection(op_pop, sim.PopulationView(op_delay_pop, [0]), sim.OneToOneConnector(), std_conn)
    for i in range(1, op_delay_pop.size):
        sim.Projection(sim.PopulationView(op_delay_pop, [i - 1]), sim.PopulationView(op_delay_pop, [i]), sim.OneToOneConnector(), std_conn)

    # OP to NOT
    sim.Projection(op_pop, sim.PopulationView(ha_pop, [0]), sim.OneToOneConnector(), std_conn)
    sim.Projection(op_pop, sim.PopulationView(ha_pop, [1]), sim.OneToOneConnector(), std_conn)
    for i in range(n_bits - 1):
        sim.Projection(sim.PopulationView(op_delay_pop, [1 + 5 * i]), sim.PopulationView(fa_pop, [18 * i]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(op_delay_pop, [1 + 5 * i]), sim.PopulationView(fa_pop, [18 * i + 1]), sim.OneToOneConnector(), std_conn)

    # A and B chains
    for i in range(1, n_bits):
        sim.Projection(a_pop[i], sim.PopulationView(a_delay_pop[i-1], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(b_pop[i], sim.PopulationView(b_delay_pop[i-1], [0]), sim.OneToOneConnector(), std_conn)

    for i in range(n_bits - 1):
        for j in range(1, a_delay_pop[i].size):
            sim.Projection(sim.PopulationView(a_delay_pop[i], [j - 1]), sim.PopulationView(a_delay_pop[i], [j]), sim.OneToOneConnector(), std_conn)
        for j in range(1, b_delay_pop[i].size):
            sim.Projection(sim.PopulationView(b_delay_pop[i], [j - 1]), sim.PopulationView(b_delay_pop[i], [j]), sim.OneToOneConnector(), std_conn)

    # Input to NOT
    sim.Projection(a_pop[0], sim.PopulationView(ha_pop, [0]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
    sim.Projection(b_pop[0], sim.PopulationView(ha_pop, [1]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
    for i in range(n_bits - 1):
        sim.Projection(sim.PopulationView(a_delay_pop[i], [a_delay_pop[i].size - 1]), sim.PopulationView(fa_pop, [18 * i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(b_delay_pop[i], [b_delay_pop[i].size - 1]), sim.PopulationView(fa_pop, [18 * i + 1]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

    # --------- HA internal structure ----------
    # 2
    sim.Projection(sim.PopulationView(op_delay_pop, [0]), sim.PopulationView(ha_pop, [2]), sim.OneToOneConnector(), std_conn)
    sim.Projection(sim.PopulationView(ha_pop, [0]), sim.PopulationView(ha_pop, [2]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
    sim.Projection(sim.PopulationView(ha_pop, [1]), sim.PopulationView(ha_pop, [2]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

    # 3
    sim.Projection(sim.PopulationView(ha_pop, [0]), sim.PopulationView(ha_pop, [3]), sim.OneToOneConnector(), std_conn)
    sim.Projection(sim.PopulationView(ha_pop, [1]), sim.PopulationView(ha_pop, [3]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

    # 4
    sim.Projection(sim.PopulationView(ha_pop, [0]), sim.PopulationView(ha_pop, [4]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
    sim.Projection(sim.PopulationView(ha_pop, [1]), sim.PopulationView(ha_pop, [4]), sim.OneToOneConnector(), std_conn)

    # 5
    sim.Projection(sim.PopulationView(ha_pop, [3]), sim.PopulationView(ha_pop, [5]), sim.OneToOneConnector(), std_conn)
    sim.Projection(sim.PopulationView(ha_pop, [4]), sim.PopulationView(ha_pop, [5]), sim.OneToOneConnector(), std_conn)

    # Connecting to the next adder
    if n_bits > 1:
        sim.Projection(sim.PopulationView(ha_pop, [2]), sim.PopulationView(fa_pop, [2]), sim.OneToOneConnector(), std_conn)

    # --------- FA internal structure ----------
    for i in range(n_bits - 1):
        # 3
        sim.Projection(sim.PopulationView(op_delay_pop, [2 + 5 * i]), sim.PopulationView(fa_pop, [3 + 18 * i]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [0 + 18 * i]), sim.PopulationView(fa_pop, [3 + 18 * i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(fa_pop, [1 + 18 * i]), sim.PopulationView(fa_pop, [3 + 18 * i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # 4
        sim.Projection(sim.PopulationView(fa_pop, [0 + 18 * i]), sim.PopulationView(fa_pop, [4 + 18 * i]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [1 + 18 * i]), sim.PopulationView(fa_pop, [4 + 18 * i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # 5
        sim.Projection(sim.PopulationView(fa_pop, [0 + 18 * i]), sim.PopulationView(fa_pop, [5 + 18 * i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(fa_pop, [1 + 18 * i]), sim.PopulationView(fa_pop, [5 + 18 * i]), sim.OneToOneConnector(), std_conn)

        # 6
        sim.Projection(sim.PopulationView(fa_pop, [2 + 18 * i]), sim.PopulationView(fa_pop, [6 + 18 * i]), sim.OneToOneConnector(), std_conn)

        # 7
        sim.Projection(sim.PopulationView(fa_pop, [3 + 18 * i]), sim.PopulationView(fa_pop, [7 + 18 * i]), sim.OneToOneConnector(), std_conn)

        # 8
        sim.Projection(sim.PopulationView(op_delay_pop, [3 + 5 * i]), sim.PopulationView(fa_pop, [8 + 18 * i]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [4 + 18 * i]), sim.PopulationView(fa_pop, [8 + 18 * i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(fa_pop, [5 + 18 * i]), sim.PopulationView(fa_pop, [8 + 18 * i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # 9
        sim.Projection(sim.PopulationView(fa_pop, [4 + 18 * i]), sim.PopulationView(fa_pop, [9 + 18 * i]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [5 + 18 * i]), sim.PopulationView(fa_pop, [9 + 18 * i]), sim.OneToOneConnector(), std_conn)

        # 10
        sim.Projection(sim.PopulationView(fa_pop, [6 + 18 * i]), sim.PopulationView(fa_pop, [10 + 18 * i]), sim.OneToOneConnector(), std_conn)

        # 11
        sim.Projection(sim.PopulationView(op_delay_pop, [3 + 5 * i]), sim.PopulationView(fa_pop, [11 + 18 * i]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [6 + 18 * i]), sim.PopulationView(fa_pop, [11 + 18 * i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # 12
        sim.Projection(sim.PopulationView(fa_pop, [7 + 18 * i]), sim.PopulationView(fa_pop, [12 + 18 * i]), sim.OneToOneConnector(), std_conn)

        # 13
        sim.Projection(sim.PopulationView(op_delay_pop, [4 + 5 * i]), sim.PopulationView(fa_pop, [13 + 18 * i]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [8 + 18 * i]), sim.PopulationView(fa_pop, [13 + 18 * i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(fa_pop, [11 + 18 * i]), sim.PopulationView(fa_pop, [13 + 18 * i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # 14
        sim.Projection(sim.PopulationView(fa_pop, [9 + 18 * i]), sim.PopulationView(fa_pop, [14 + 18 * i]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [10 + 18 * i]), sim.PopulationView(fa_pop, [14 + 18 * i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # 15
        sim.Projection(sim.PopulationView(fa_pop, [9 + 18 * i]), sim.PopulationView(fa_pop, [15 + 18 * i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(fa_pop, [10 + 18 * i]), sim.PopulationView(fa_pop, [15 + 18 * i]), sim.OneToOneConnector(), std_conn)

        # 16
        sim.Projection(sim.PopulationView(fa_pop, [12 + 18 * i]), sim.PopulationView(fa_pop, [16 + 18 * i]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [13 + 18 * i]), sim.PopulationView(fa_pop, [16 + 18 * i]), sim.OneToOneConnector(), std_conn)
        # Connecting to the next adder
        if i != (n_bits - 2):
            sim.Projection(sim.PopulationView(fa_pop, [16 + 18 * i]), sim.PopulationView(fa_pop, [2 + 18 * (i + 1)]), sim.OneToOneConnector(), std_conn)

        # 17
        sim.Projection(sim.PopulationView(fa_pop, [14 + 18 * i]), sim.PopulationView(fa_pop, [17 + 18 * i]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [15 + 18 * i]), sim.PopulationView(fa_pop, [17 + 18 * i]), sim.OneToOneConnector(), std_conn)

    # -- Recording --
    ha_pop[[5]].record(["spikes"])
    if n_bits > 1:
        fa_neurons = [17 + 18 * i for i in range(n_bits - 1)]
        fa_pop[fa_neurons].record(["spikes"])

    # -- Run simulation --
    sim.run(global_params["sim_time"])

    # -- Get data from the simulation --
    ha_data = ha_pop[[5]].get_data().segments[0]
    if n_bits > 1:
        fa_data = fa_pop[fa_neurons].get_data().segments[0]

    # - End simulation -
    sim.end()

    # --- Saving test ---
    if n_bits > 1:
        save_array = [op_times, a_times, b_times, ha_data, fa_data]
    else:
        save_array = [op_times, a_times, b_times, ha_data]
    test_name = os.path.basename(__file__).split('.')[0] + "_" + str(n_bits) + "_bits"

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
    if n_bits > 1:
        plt.rcParams["figure.figsize"] = (4, 0.8 * (7 + 3 * (n_bits - 2)) / 7)
    else:
        plt.rcParams["figure.figsize"] = (4, 0.8)

    # Spikes
    plt.plot(op_times, [-1] * len(op_times), 'o', markersize=0.5, color='tab:blue')
    for i in range(n_bits):
        plt.plot(a_times[i], [i] * len(a_times[i]), 'o', markersize=0.5, color='tab:orange')
    for i in range(n_bits):
        plt.plot(b_times[i], [i + n_bits] * len(b_times[i]), 'o', markersize=0.5, color='tab:olive')

    plt.plot(np.array(ha_data.spiketrains[0]) - 3, [n_bits * 2] * len(ha_data.spiketrains[0]), 'o', markersize=0.5, color='tab:green')
    for i in range(n_bits - 1):
        plt.plot(np.array(fa_data.spiketrains[i]) - 2 - 5 * (i + 1), [n_bits * 2 + 1 + i] * len(fa_data.spiketrains[i]), 'o', markersize=0.5, color='tab:green')

    plt.xlabel("Time (ms)")
    plt.ylabel("Neurons")
    plt.yticks(range(-2, n_bits * 3 + 1), ["", "OP"] + ["A" + str(i) for i in range(n_bits)] + ["B" + str(i) for i in range(n_bits)] + ["S" + str(i) for i in range(n_bits)] + [""])

    plt.tight_layout()
    plt.savefig("experiments/" + filename + '.png', transparent=False, facecolor='white', edgecolor='black')
    plt.show()

    # NOTE: TO CORRECTLY VISUALIZE THE RESULTS IT IS NECESSARY TO INCREASE THE TIME SCALE FACTOR

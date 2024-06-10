import os
import pickle

import numpy as np
import spynnaker.pyNN as sim
from matplotlib import pyplot as plt

# Neuron parameters
global_params = {"min_delay": 1.0, "sim_time": 150.0}
neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1, "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}


if __name__ == '__main__':
    # --- Simulation ---
    sim.setup(global_params["min_delay"])

    # --- Predefined objects ---
    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])  # Standard connection
    n_bits = 3
    factor = 1
    total_neurons = 0
    total_synapses = 0

    # -- Network architecture --
    # - Spike injectors -
    op0 = np.concatenate([range(10, 19, 1), range(28, 37, 1)], axis=0)
    op1 = range(19, 37, 1)
    op2 = range(37, 46, 1)
    op_times = [op0, op1, op2]

    a0 = [10, 16, 19, 25, 28, 34, 37, 43]
    a1 = [11, 17, 20, 26, 29, 35, 38, 44]
    a2 = [12, 18, 21, 27, 30, 36, 39, 45]
    a_times = [a0, a1, a2]

    b0 = [15, 16, 24, 25, 33, 34, 42, 43]
    b1 = [14, 17, 23, 26, 32, 35, 41, 44]
    b2 = [13, 18, 22, 27, 31, 36, 40, 45]
    b_times = [b0, b1, b2]

    op_pop = [sim.Population(1, sim.SpikeSourceArray(spike_times=op_times[i])) for i in range(n_bits)]
    a_pop = [sim.Population(1, sim.SpikeSourceArray(spike_times=a_times[i])) for i in range(n_bits)]
    b_pop = [sim.Population(1, sim.SpikeSourceArray(spike_times=b_times[i])) for i in range(n_bits)]
    total_neurons += 3 * n_bits

    # - ALU Input Delays -
    alu_delay_pops = [sim.Population(3, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="alu_delay") for i in range(6)]
    total_neurons += 3 * 6

    for i in range(3):
        sim.Projection(a_pop[i], sim.PopulationView(alu_delay_pops[i], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(b_pop[i], sim.PopulationView(alu_delay_pops[3 + i], [0]), sim.OneToOneConnector(), std_conn)
        total_synapses += 2

        for j in range(1, 3):
            sim.Projection(sim.PopulationView(alu_delay_pops[i], [j - 1]), sim.PopulationView(alu_delay_pops[i], [j]), sim.OneToOneConnector(), std_conn)
            sim.Projection(sim.PopulationView(alu_delay_pops[3 + i], [j - 1]), sim.PopulationView(alu_delay_pops[3 + i], [j]), sim.OneToOneConnector(), std_conn)
            total_synapses += 2

    # - Decoder -
    decoder_input = sim.Population(3 * 2, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="decoder_input")
    decoder_middle = sim.Population(3 * 2 + 1, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="decoder_middle")
    decoder_output = sim.Population(4, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="decoder_output")
    total_neurons += 3 * 2 + 3 * 2 + 1 + 4

    # Input layer
    for i in range(3):  # 2 * n
        sim.Projection(op_pop[i], sim.PopulationView(decoder_input, [2 * i]), sim.OneToOneConnector(), std_conn)
        sim.Projection(op_pop[i], sim.PopulationView(decoder_input, [2 * i + 1]), sim.OneToOneConnector(), std_conn)
        total_synapses += 2

        for j in range(i + 1, 3):  # (n - 1) * (1 + (n - 1)) / 2 = (n ^ 2 - n) / 2
            sim.Projection(op_pop[i], sim.PopulationView(decoder_input, [2 * j + 1]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
            total_synapses += 1

    # Middle layer
    for i in range(3):  # 2 * n
        sim.Projection(sim.PopulationView(decoder_input, [2 * i]), sim.PopulationView(decoder_middle, [2 * i]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(decoder_input, [2 * i]), sim.PopulationView(decoder_middle, [2 * i + 1]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        total_synapses += 2

        # OP to NOT -> n ^ 2
        for j in range(3):
            sim.Projection(sim.PopulationView(decoder_input, [2 * i + 1]), sim.PopulationView(decoder_middle, [2 * j + 1]), sim.OneToOneConnector(), std_conn)
            total_synapses += 1

        # OP (+1) -> n
        sim.Projection(sim.PopulationView(decoder_input, [2 * i + 1]), sim.PopulationView(decoder_middle, [6]), sim.OneToOneConnector(), std_conn)
        total_synapses += 1

    # Output layer
    for i in range(1, 5):  # 2 ^ n - 1
        i_bin = format(i, "0" + str(3) + "b")[::-1]

        # Inhibition
        for j in range(len(i_bin)):
            if i_bin[j] == '1':
                sim.Projection(sim.PopulationView(decoder_middle, [2 * j + 1]), sim.PopulationView(decoder_output, [i - 1]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
            else:
                sim.Projection(sim.PopulationView(decoder_middle, [2 * j]), sim.PopulationView(decoder_output, [i - 1]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
            total_synapses += 1

        # Excitation (OP (+1))
        sim.Projection(sim.PopulationView(decoder_middle, [6]), sim.PopulationView(decoder_output, [i - 1]), sim.OneToOneConnector(), std_conn)
        total_synapses += 1

    # - OR -
    or_pops = [sim.Population(3, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}) for i in range(n_bits)]
    total_neurons += 3 * n_bits

    for i in range(3):
        # OP to 0 & 1
        sim.Projection(sim.PopulationView(decoder_output, [0]), sim.PopulationView(or_pops[i], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(decoder_output, [0]), sim.PopulationView(or_pops[i], [1]), sim.OneToOneConnector(), std_conn)
        total_synapses += 2

        # Inputs to NOR
        sim.Projection(sim.PopulationView(alu_delay_pops[i], [2]), sim.PopulationView(or_pops[i], [1]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(alu_delay_pops[3 + i], [2]), sim.PopulationView(or_pops[i], [1]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        total_synapses += 2

        # NOT
        sim.Projection(sim.PopulationView(or_pops[i], [0]), sim.PopulationView(or_pops[i], [2]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(or_pops[i], [1]), sim.PopulationView(or_pops[i], [2]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        total_synapses += 2

    # - XOR -
    xor_pops = [sim.Population(4, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}) for i in range(n_bits)]
    total_neurons += 4 * n_bits

    for i in range(3):
        # OP delayed
        sim.Projection(sim.PopulationView(decoder_output, [1]), sim.PopulationView(xor_pops[i], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(decoder_output, [1]), sim.PopulationView(xor_pops[i], [1]), sim.OneToOneConnector(), std_conn)
        total_synapses += 2

        # Input delayed
        sim.Projection(sim.PopulationView(alu_delay_pops[i], [2]), sim.PopulationView(xor_pops[i], [0]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(alu_delay_pops[3 + i], [2]), sim.PopulationView(xor_pops[i], [1]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        total_synapses += 2

        # Not A and B
        sim.Projection(sim.PopulationView(xor_pops[i], [0]), sim.PopulationView(xor_pops[i], [2]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(xor_pops[i], [1]), sim.PopulationView(xor_pops[i], [2]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        total_synapses += 2

        # A and not B
        sim.Projection(sim.PopulationView(xor_pops[i], [0]), sim.PopulationView(xor_pops[i], [3]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(xor_pops[i], [1]), sim.PopulationView(xor_pops[i], [3]), sim.OneToOneConnector(), std_conn)
        total_synapses += 2

    # - AND -
    and_pops = [sim.Population(4, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}) for i in range(n_bits)]
    total_neurons += 4 * n_bits

    for i in range(3):
        # OP
        sim.Projection(sim.PopulationView(decoder_output, [2]), sim.PopulationView(and_pops[i], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(decoder_output, [2]), sim.PopulationView(and_pops[i], [1]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(decoder_output, [2]), sim.PopulationView(and_pops[i], [2]), sim.OneToOneConnector(), std_conn)
        total_synapses += 3

        # NOT
        sim.Projection(sim.PopulationView(alu_delay_pops[i], [2]), sim.PopulationView(and_pops[i], [1]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(alu_delay_pops[3 + i], [2]), sim.PopulationView(and_pops[i], [2]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        total_synapses += 2

        # NOR
        sim.Projection(sim.PopulationView(and_pops[i], [0]), sim.PopulationView(and_pops[i], [3]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(and_pops[i], [1]), sim.PopulationView(and_pops[i], [3]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(and_pops[i], [2]), sim.PopulationView(and_pops[i], [3]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        total_synapses += 3

    # - ADDER -
    if n_bits > 1:
        op_delay_pop = sim.Population(5 * (n_bits - 1), sim.IF_curr_exp(**neuron_params),
                                      initial_values={'v': neuron_params["v_rest"]}, label="op_delay")
    else:
        op_delay_pop = sim.Population(1, sim.IF_curr_exp(**neuron_params),
                                      initial_values={'v': neuron_params["v_rest"]}, label="op_delay")
    a_delay_pop = [
        sim.Population(2 + 5 * i, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]},
                       label="a_delay") for i in range(n_bits - 1)]
    b_delay_pop = [
        sim.Population(2 + 5 * i, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]},
                       label="b_delay") for i in range(n_bits - 1)]

    ha_pop = sim.Population(6, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]})
    if n_bits > 1:
        fa_pop = sim.Population(18 * (n_bits - 1), sim.IF_curr_exp(**neuron_params),
                                initial_values={'v': neuron_params["v_rest"]})

    # - Connections -
    # OP chain
    sim.Projection(sim.PopulationView(decoder_output, [3]), sim.PopulationView(op_delay_pop, [0]), sim.OneToOneConnector(), std_conn)
    for i in range(1, op_delay_pop.size):
        sim.Projection(sim.PopulationView(op_delay_pop, [i - 1]), sim.PopulationView(op_delay_pop, [i]),
                       sim.OneToOneConnector(), std_conn)

    # OP to NOT
    sim.Projection(sim.PopulationView(decoder_output, [3]), sim.PopulationView(ha_pop, [0]), sim.OneToOneConnector(), std_conn)
    sim.Projection(sim.PopulationView(decoder_output, [3]), sim.PopulationView(ha_pop, [1]), sim.OneToOneConnector(), std_conn)
    for i in range(n_bits - 1):
        sim.Projection(sim.PopulationView(op_delay_pop, [1 + 5 * i]), sim.PopulationView(fa_pop, [18 * i]),
                       sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(op_delay_pop, [1 + 5 * i]), sim.PopulationView(fa_pop, [18 * i + 1]),
                       sim.OneToOneConnector(), std_conn)

    # A and B chains
    for i in range(1, n_bits):
        sim.Projection(sim.PopulationView(alu_delay_pops[i], [2]), sim.PopulationView(a_delay_pop[i - 1], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(alu_delay_pops[3 + i], [2]), sim.PopulationView(b_delay_pop[i - 1], [0]), sim.OneToOneConnector(), std_conn)

    for i in range(n_bits - 1):
        for j in range(1, a_delay_pop[i].size):
            sim.Projection(sim.PopulationView(a_delay_pop[i], [j - 1]), sim.PopulationView(a_delay_pop[i], [j]),
                           sim.OneToOneConnector(), std_conn)
        for j in range(1, b_delay_pop[i].size):
            sim.Projection(sim.PopulationView(b_delay_pop[i], [j - 1]), sim.PopulationView(b_delay_pop[i], [j]),
                           sim.OneToOneConnector(), std_conn)

    # Input to NOT
    sim.Projection(sim.PopulationView(alu_delay_pops[0], [2]), sim.PopulationView(ha_pop, [0]), sim.OneToOneConnector(), std_conn,
                   receptor_type="inhibitory")
    sim.Projection(sim.PopulationView(alu_delay_pops[3], [2]), sim.PopulationView(ha_pop, [1]), sim.OneToOneConnector(), std_conn,
                   receptor_type="inhibitory")
    for i in range(n_bits - 1):
        sim.Projection(sim.PopulationView(a_delay_pop[i], [a_delay_pop[i].size - 1]),
                       sim.PopulationView(fa_pop, [18 * i]), sim.OneToOneConnector(), std_conn,
                       receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(b_delay_pop[i], [b_delay_pop[i].size - 1]),
                       sim.PopulationView(fa_pop, [18 * i + 1]), sim.OneToOneConnector(), std_conn,
                       receptor_type="inhibitory")

    # --------- HA internal structure ----------
    # 2
    sim.Projection(sim.PopulationView(op_delay_pop, [0]), sim.PopulationView(ha_pop, [2]), sim.OneToOneConnector(),
                   std_conn)
    sim.Projection(sim.PopulationView(ha_pop, [0]), sim.PopulationView(ha_pop, [2]), sim.OneToOneConnector(),
                   std_conn, receptor_type="inhibitory")
    sim.Projection(sim.PopulationView(ha_pop, [1]), sim.PopulationView(ha_pop, [2]), sim.OneToOneConnector(),
                   std_conn, receptor_type="inhibitory")

    # 3
    sim.Projection(sim.PopulationView(ha_pop, [0]), sim.PopulationView(ha_pop, [3]), sim.OneToOneConnector(),
                   std_conn)
    sim.Projection(sim.PopulationView(ha_pop, [1]), sim.PopulationView(ha_pop, [3]), sim.OneToOneConnector(),
                   std_conn, receptor_type="inhibitory")

    # 4
    sim.Projection(sim.PopulationView(ha_pop, [0]), sim.PopulationView(ha_pop, [4]), sim.OneToOneConnector(),
                   std_conn, receptor_type="inhibitory")
    sim.Projection(sim.PopulationView(ha_pop, [1]), sim.PopulationView(ha_pop, [4]), sim.OneToOneConnector(),
                   std_conn)

    # 5
    sim.Projection(sim.PopulationView(ha_pop, [3]), sim.PopulationView(ha_pop, [5]), sim.OneToOneConnector(),
                   std_conn)
    sim.Projection(sim.PopulationView(ha_pop, [4]), sim.PopulationView(ha_pop, [5]), sim.OneToOneConnector(),
                   std_conn)

    # Connecting to the next adder
    if n_bits > 1:
        sim.Projection(sim.PopulationView(ha_pop, [2]), sim.PopulationView(fa_pop, [2]), sim.OneToOneConnector(),
                       std_conn)

    # --------- FA internal structure ----------
    for i in range(n_bits - 1):
        # 3
        sim.Projection(sim.PopulationView(op_delay_pop, [2 + 5 * i]), sim.PopulationView(fa_pop, [3 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [0 + 18 * i]), sim.PopulationView(fa_pop, [3 + 18 * i]),
                       sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(fa_pop, [1 + 18 * i]), sim.PopulationView(fa_pop, [3 + 18 * i]),
                       sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # 4
        sim.Projection(sim.PopulationView(fa_pop, [0 + 18 * i]), sim.PopulationView(fa_pop, [4 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [1 + 18 * i]), sim.PopulationView(fa_pop, [4 + 18 * i]),
                       sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # 5
        sim.Projection(sim.PopulationView(fa_pop, [0 + 18 * i]), sim.PopulationView(fa_pop, [5 + 18 * i]),
                       sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(fa_pop, [1 + 18 * i]), sim.PopulationView(fa_pop, [5 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)

        # 6
        sim.Projection(sim.PopulationView(fa_pop, [2 + 18 * i]), sim.PopulationView(fa_pop, [6 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)

        # 7
        sim.Projection(sim.PopulationView(fa_pop, [3 + 18 * i]), sim.PopulationView(fa_pop, [7 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)

        # 8
        sim.Projection(sim.PopulationView(op_delay_pop, [3 + 5 * i]), sim.PopulationView(fa_pop, [8 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [4 + 18 * i]), sim.PopulationView(fa_pop, [8 + 18 * i]),
                       sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(fa_pop, [5 + 18 * i]), sim.PopulationView(fa_pop, [8 + 18 * i]),
                       sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # 9
        sim.Projection(sim.PopulationView(fa_pop, [4 + 18 * i]), sim.PopulationView(fa_pop, [9 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [5 + 18 * i]), sim.PopulationView(fa_pop, [9 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)

        # 10
        sim.Projection(sim.PopulationView(fa_pop, [6 + 18 * i]), sim.PopulationView(fa_pop, [10 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)

        # 11
        sim.Projection(sim.PopulationView(op_delay_pop, [3 + 5 * i]), sim.PopulationView(fa_pop, [11 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [6 + 18 * i]), sim.PopulationView(fa_pop, [11 + 18 * i]),
                       sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # 12
        sim.Projection(sim.PopulationView(fa_pop, [7 + 18 * i]), sim.PopulationView(fa_pop, [12 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)

        # 13
        sim.Projection(sim.PopulationView(op_delay_pop, [4 + 5 * i]), sim.PopulationView(fa_pop, [13 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [8 + 18 * i]), sim.PopulationView(fa_pop, [13 + 18 * i]),
                       sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(fa_pop, [11 + 18 * i]), sim.PopulationView(fa_pop, [13 + 18 * i]),
                       sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # 14
        sim.Projection(sim.PopulationView(fa_pop, [9 + 18 * i]), sim.PopulationView(fa_pop, [14 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [10 + 18 * i]), sim.PopulationView(fa_pop, [14 + 18 * i]),
                       sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # 15
        sim.Projection(sim.PopulationView(fa_pop, [9 + 18 * i]), sim.PopulationView(fa_pop, [15 + 18 * i]),
                       sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(fa_pop, [10 + 18 * i]), sim.PopulationView(fa_pop, [15 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)

        # 16
        sim.Projection(sim.PopulationView(fa_pop, [12 + 18 * i]), sim.PopulationView(fa_pop, [16 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [13 + 18 * i]), sim.PopulationView(fa_pop, [16 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)
        # Connecting to the next adder
        if i != (n_bits - 2):
            sim.Projection(sim.PopulationView(fa_pop, [16 + 18 * i]),
                           sim.PopulationView(fa_pop, [2 + 18 * (i + 1)]), sim.OneToOneConnector(), std_conn)

        # 17
        sim.Projection(sim.PopulationView(fa_pop, [14 + 18 * i]), sim.PopulationView(fa_pop, [17 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(fa_pop, [15 + 18 * i]), sim.PopulationView(fa_pop, [17 + 18 * i]),
                       sim.OneToOneConnector(), std_conn)

    # -- Recording --
    input_delay_neurons = [2]
    or_neurons = [2]
    xor_neurons = [2, 3]
    and_neurons = [3]
    ha_neurons = [5]
    fa_neurons = [17 + 18 * i for i in range(n_bits - 1)]

    for i in range(6):
        alu_delay_pops[i][input_delay_neurons].record("spikes")
    decoder_output.record(["spikes"])
    for i in range(3):
        or_pops[i][or_neurons].record(["spikes"])
        xor_pops[i][xor_neurons].record(["spikes"])
        and_pops[i][and_neurons].record(["spikes"])

    ha_pop[ha_neurons].record(["spikes"])
    fa_pop[fa_neurons].record(["spikes"])

    # -- Run simulation --
    sim.run(global_params["sim_time"])

    # -- Get data from the simulation --
    delayed_input_data = []
    decoder_data = decoder_output.get_data().segments[0]
    or_data = []
    xor_data = []
    and_data = []

    for i in range(6):
        delayed_input_data.append(alu_delay_pops[i][input_delay_neurons].get_data().segments[0])
    for i in range(3):
        or_data.append(or_pops[i][or_neurons].get_data().segments[0])
        xor_data.append(xor_pops[i][xor_neurons].get_data().segments[0])
        and_data.append(and_pops[i][and_neurons].get_data().segments[0])

    ha_data = ha_pop[[5]].get_data().segments[0]
    fa_data = fa_pop[fa_neurons].get_data().segments[0]

    # - End simulation -
    sim.end()

    save_array = [delayed_input_data, decoder_data, or_data, xor_data, and_data, ha_data, fa_data]
    test_name = "alu_2_inputs_3_bits_factor_" + str(factor)
    folder_name = "results_ALU"

    cwd = os.getcwd()
    if not os.path.exists(cwd + "/" + folder_name + "/"):
        os.mkdir(cwd + "/" + folder_name + "/")

    i = 1
    while os.path.exists(cwd + "/" + folder_name + "/" + test_name + "_" + str(i) + ".pickle"):
        i += 1

    filename = test_name + "_" + str(i)

    with open(folder_name + "/" + filename + '.pickle', 'wb') as handle:
        pickle.dump(save_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder_name + "/" + filename + '.pickle', "rb") as handle:
        print(pickle.load(handle))
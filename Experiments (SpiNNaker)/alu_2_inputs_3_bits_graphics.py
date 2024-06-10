import pickle
import matplotlib.pyplot as plt
import numpy as np

filename = "alu_2_inputs_3_bits_factor_1_1"
folder_name = "results_ALU"

with open(folder_name + "/" + filename + '.pickle', "rb") as handle:
    delayed_input_data, decoder_data, or_data, xor_data, and_data, ha_data, fa_data = pickle.load(handle)

plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = '4'
plt.rcParams["figure.figsize"] = (4, 1.5)

adjusted = True

# Input times
for i in range(6):
    plt.plot(delayed_input_data[i].spiketrains[0], [i] * len(delayed_input_data[i].spiketrains[0]), '|', markersize=2, color='tab:red')

# decoder
for i in range(4):
    plt.plot(decoder_data.spiketrains[i], [6 + i] * len(decoder_data.spiketrains[i]), '|', markersize=2, color='tab:blue')

if adjusted:
    # or, xor, and
    for i in range(3):
        plt.plot(np.array(or_data[i].spiketrains[0]) - 2, [10 + i] * len(or_data[i].spiketrains[0]), '|', markersize=2, color='tab:green')
        plt.plot(np.array(xor_data[i].spiketrains[0]) - 2, [13 + i] * len(xor_data[i].spiketrains[0]), '|', markersize=2, color='tab:green')
        plt.plot(np.array(xor_data[i].spiketrains[1]) - 2, [13 + i] * len(xor_data[i].spiketrains[1]), '|', markersize=2, color='tab:green')
        plt.plot(np.array(and_data[i].spiketrains[0]) - 2, [16 + i] * len(and_data[i].spiketrains[0]), '|', markersize=2, color='tab:green')

    # adder
    plt.plot(np.array(ha_data.spiketrains[0]) - 3, [19] * len(ha_data.spiketrains[0]), '|', markersize=2, color='tab:green')
    plt.plot(np.array(fa_data.spiketrains[0]) - 7, [20] * len(fa_data.spiketrains[0]), '|', markersize=2, color='tab:green')
    plt.plot(np.array(fa_data.spiketrains[1]) - 12, [21] * len(fa_data.spiketrains[1]), '|', markersize=2, color='tab:green')
else:
    # or, xor, and
    for i in range(3):
        plt.plot(or_data[i].spiketrains[0], [10 + i] * len(or_data[i].spiketrains[0]), '|', markersize=2, color='tab:green')
        plt.plot(xor_data[i].spiketrains[0], [13 + i] * len(xor_data[i].spiketrains[0]), '|', markersize=2, color='tab:green')
        plt.plot(xor_data[i].spiketrains[1], [13 + i] * len(xor_data[i].spiketrains[1]), '|', markersize=2, color='tab:green')
        plt.plot(and_data[i].spiketrains[0], [16 + i] * len(and_data[i].spiketrains[0]), '|', markersize=2, color='tab:green')

    # adder
    plt.plot(ha_data.spiketrains[0], [19] * len(ha_data.spiketrains[0]), '|', markersize=2, color='tab:green')
    plt.plot(fa_data.spiketrains[0], [20] * len(fa_data.spiketrains[0]), '|', markersize=2, color='tab:green')
    plt.plot(fa_data.spiketrains[1], [21] * len(fa_data.spiketrains[1]), '|', markersize=2, color='tab:green')

plt.xlim([9 + 3, 45 + 4])
plt.xlabel('Time (ms)')
plt.yticks(range(-1, 23), [""] + ["A" + str(i) + " (+3)" for i in range(3)] + ["B" + str(i) + " (+3)" for i in range(3)] +
           ["OP OR", "OP XOR", "OP AND", "OP ADDER"] + ["OR (" + str(i) + ")" for i in range(3)] +
           ["XOR (" + str(i) + ")" for i in range(3)] + ["AND (" + str(i) + ")" for i in range(3)] +
           ["ADDER (" + str(i) + ")" for i in range(3)] + [""])
plt.hlines([9.5, 12.5, 15.5, 18.5], xmin=0, xmax=1000, linewidth=0.1, color="indigo")

plt.tight_layout()
#plt.show()
plt.savefig(folder_name + "/" + filename + '.png', transparent=False, facecolor='white', edgecolor='black')
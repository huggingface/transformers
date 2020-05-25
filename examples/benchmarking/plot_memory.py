#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=10)

x = np.asarray([2**10, 2**11, 2**12, 2**13, 2**14, 2**15])

y_bert_1 = np.asarray([24, 72, 284, 1020, 4059, 16160])
y_ref_1_2 = np.asarray([30, 74, 122, 190, 492, 1227])
y_ref_1_4 = np.asarray([48, 124, 186, 368, 920, 2300])
y_ref_1_2_fac = np.asarray([31, 74, 122, 146, 320, 632])
y_ref_1_4_fac = np.asarray([48, 124, 154, 260, 560, 1084])


y_bert_8 = np.asarray([148, 592, 2180, 8297, 32600, 128000])
y_ref_8_2 = np.asarray([156, 364, 806, 1697, 3791, 9691])
y_ref_8_4 = np.asarray([324, 692, 1426, 3117, 6971, 18289])
y_ref_8_2_fac = np.asarray([124, 316, 634, 1229, 2448, 4891])
y_ref_8_4_fac = np.asarray([260, 568, 1082, 2149, 4357, 8766])

y_16_gb = np.asarray([16000, 16000, 16000, 16000, 16000, 16000])
plt.xlim(min(x), max(x))
plt.ylim(0, 20000)

#plt.scatter(x, y_ref_1_2, c='lime', label="Reformer Layer 2 hashes")
#plt.plot(x, y_ref_1_2, '--', color='lime')
#plt.scatter(x, y_ref_1_4, c='olive', label="Reformer Layer 4 hashes")
#plt.plot(x, y_ref_1_4, '--', color='olive')
#plt.scatter(x, y_ref_1_2_fac, c='lightgreen', label="Reformer Layer 2 hashes factorized")
#plt.plot(x, y_ref_1_2_fac, '--', color='lightgreen')
#plt.scatter(x, y_ref_1_4_fac, c='green', label="Reformer Layer 4 hashes factorized")
#plt.plot(x, y_ref_1_4_fac, '--', color='green')
#plt.scatter(x, y_bert_1, c='maroon', label="Bert Layer")
#plt.plot(x, y_bert_1, '--', c='maroon')
#plt.title("Time usage Bert Layer vs. Reformer Layer for Batch size=1")

plt.scatter(x, y_ref_8_2, c='lime', label="Reformer Layer 2 hashes")
plt.plot(x, y_ref_8_2, '--', color='lime')
plt.scatter(x, y_ref_8_4, c='olive', label="Reformer Layer 4 hashes")
plt.plot(x, y_ref_8_4, '--', color='olive')
plt.scatter(x, y_ref_8_2_fac, c='lightgreen', label="Reformer Layer 2 hashes factorized")
plt.plot(x, y_ref_8_2_fac, '--', color='lightgreen')
plt.scatter(x, y_ref_8_4_fac, c='green', label="Reformer Layer 4 hashes factorized")
plt.plot(x, y_ref_8_4_fac, '--', color='green')
plt.scatter(x, y_bert_8, c='maroon', label="Bert Layer")
plt.plot(x, y_bert_8, '--', c='maroon')
plt.title("Time usage Bert Layer vs. Reformer Layer for Batch size=8")

plt.plot(x, y_16_gb, 'r--', label="16 Giga Bytes")

plt.ylabel("Memory usage in Mega Bytes")
plt.xlabel("Sequence Length")

plt.legend()
plt.show()

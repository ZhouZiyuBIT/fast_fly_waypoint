import matplotlib.pyplot as plt
import numpy as np

import sys, os
BASEPATH = os.path.abspath(__file__).split("fast_fly/", 1)[0]+"fast_fly/"
sys.path += [BASEPATH]

ds = np.array([    2,   1,  0.5, 0.3, 0.25, 0.2, 0.16, 0.14, 0.12, 0.1])
opt_t = np.array([11.448014766052545, 11.787239760116519, 11.94449169709534, 12.001143565203918, 12.009366438439377, 12.024716491983977, 12.042188805546745, 12.044477669521575, 12.061446096920848, 12.065915659231877])
used_t = np.array([[0.09050139188766479, 0.008087538808229579], [0.2618164300918579, 0.05397132655822128], [0.6466596007347107, 0.0972560291615073], [1.2445058226585388, 0.2131515212595687], [1.4218987941741943, 0.2347261118645997], [1.7315568208694458, 0.43539411455102], [1.8991067171096803, 0.20901845092848348], [2.403267741203308, 0.9066604204940346], [2.8310678124427797, 0.9078599775683173], [4.559802675247193, 1.8911455845023564]])

fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(111)
ax_t = ax.twinx()

ax.set_ylabel("Solve times [s]")
ax.set_ylim(0, 7)
ax.plot(1/ds, np.array(used_t)[:,0], linestyle="-", label="Solve time")
ax.fill_between(1/ds, np.array(used_t)[:,0]-np.array(used_t)[:,1], np.array(used_t)[:,0]+np.array(used_t)[:,1], color="skyblue",alpha=0.2)

ax.set_xlabel("1/D [m]")
ax_t.set_ylabel("Optimal times [s]")
ax_t.set_ylim(11.4, 12.2)
ax_t.plot([],[])
ax_t.plot(1/ds, opt_t, linestyle="--", label="Optimal time")

fig.legend(bbox_to_anchor=(0.0, 1), bbox_transform=ax.transAxes, ncol=2, loc="upper left")
fig.tight_layout(pad=0, rect=[0.01, 0.01, 0.99, 0.95])

plt.savefig(BASEPATH+"figs/time_analysis.eps")
plt.show()
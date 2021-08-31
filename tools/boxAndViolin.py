# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:45:05 2020

@author: hoss3301
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


fig, ax = plt.subplots()
violin_handle = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)


colors = plt.cm.gist_ncar(np.linspace(0.2, .8, len(violin_handle['bodies'])))
np.random.seed(0)
colors = plt.cm.gist_ncar(np.random.rand(len(violin_handle['bodies'])))
np.random.shuffle(colors)
colors=['lightyellow','thistle','forestgreen','lightskyblue']
for pc, c in zip(violin_handle['bodies'], colors):
    pc.set_facecolor(c)
    pc.set_edgecolor('black')
    pc.set_alpha(1)
 
    
quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
 
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]






inds = np.arange(1, len(medians) + 1)
ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
    
 
 

num_boxes = len(data)
pos = np.arange(num_boxes) + 1


   
upper_labels = medians
upper_labels = [str(np.round(s, 2)) for s in medians]

weights = ['bold', 'semibold']



for tick, label in zip(range(num_boxes), ax.get_xticklabels()):
    k = tick % 2
    ax.text(pos[tick], .9, upper_labels[tick],
             transform=ax.get_xaxis_transform(),
             horizontalalignment='center', fontsize=12,
             weight=weights[k])

ax.set_ylabel(metrics[z],fontsize=15)
ax.set_ylim(-10,15)
plt.yticks([-10,0,10,15],fontsize=15)
'''ax.set_ylim(0,1.2)
plt.yticks([0,0.5,1],fontsize=15)'''
#ax.set_xticks()
plt.xticks(np.arange(len(fusions)) + 1, fusions,fontsize=12)

plt.show()

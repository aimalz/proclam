# import string
# import itertools
# import random
# import os
# import csv

import numpy as np
import pandas as pd

import proclam
from proclam import *
import matplotlib as mpl
import pylab as pl
mpl.use('Agg')
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['savefig.dpi'] = 250
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['savefig.bbox'] = 'tight'
import matplotlib.pyplot as plt
metricslist = ['Brier', 'LogLoss']
colors = ['teal', 'magenta']
dirname = 'examples/'
markerlist = ['d', 'o', 's', '*']
plasticc = {}
plasticc['label'] = 'plasticc'
#plasticc['names'] = ['Submission_alpha_0.5_190516_1756', 'submission_40_avocado', 'submission_probe99_40_avocado'] 
plasticc['names'] = ['3_MajorTom'] #'2_MikeSilogram' ] #'1_Kyle']

#, '2_MikeSilogram', '3_MajorTom']



list = [6,15,16,42,52,53,62,64,65,67,88,90,92,95,99]
itemlist=['uLens-Point', 'TDE', 'EBE', 'SNCC-II', 'MIRA', 'SNCC-Ibc', 'KN', 'Mdwarf', 'SNIa-91bg', 'AGN', 'SNIa-normal', 'RRlyrae', 'SLSN-I', 'Other']

choices=['All']

#, 'uLens-Point', 'TDE', 'EBE', 'SNCC-II', 'MIRA', 'SNCC-Ibc', 'KN', 'Mdwarf', 'SNIa-91bg', 'AGN', 'SNIa-normal', 'RRlyrae', 'SLSN-I', 'Other']

# 90    SNIa-normal
#62       SNCC-Ibc
#42        SNCC-II
#67      SNIa-91bg
#52         SNIa-x
#64             KN
#95         SLSN-I
#99          Other
#15            TDE
#88            AGN
#92        RRlyrae
#65         Mdwarf
#16            EBE
#53           MIRA
#6    uLens-Point


#Idealized', 'Guess', 'Tunnel', 'Broadbrush', 'Cruise', 'SubsumedTo', 'SubsumedFrom']
def make_class_pairs(data_info_dict):
    return zip(data_info_dict['classifications'], data_info_dict['truth_tables'])

def make_file_locs(data_info_dict):
    names = data_info_dict['names']
    data_info_dict['dirname'] = dirname + data_info_dict['label'] + '/'
#    data_info_dict['classifications'] = ['%s/predicted_prob_%s.csv'%(name, name) for name in names]
#   data_info_dict['truth_tables'] = ['%s/truth_table_%s.csv'%(name, name) for name in names]
    data_info_dict['classifications'] = ['%s/%s.csv'%(name, name) for name in names]
    data_info_dict['truth_tables'] = ['%s/%s_truth.csv'%(name, name) for name in names]
    print(data_info_dict)
    return data_info_dict

def plot_cm(probs, truth, name, loc=''):
    print(np.shape(probs), np.shape(truth), 'checking sizes of probs and truth')
    cm = proclam.metrics.util.prob_to_cm(probs, truth)
    pl.clf()
    plt.matshow(cm.T, vmin=0., vmax=1.)
# plt.xticks(range(max(truth)+1), names)
# plt.yticks(range(max(truth)+1), names)
    plt.xlabel('predicted class')
    plt.ylabel('true class')
    plt.colorbar()
    plt.title(name)
    plt.savefig(loc+name+'_cm.png')
    #plt.show()
    #plt.close()



def read_class_pairs(pair, dataset, cc):#loc='', title=''):
    loc=dataset['dirname']
    title=dataset['label']+' '+ dataset['names'][cc]
    clfile = pair[0]
    truthfile = pair[1]
    print(clfile, truthfile)
    prob_mat = pd.read_csv(loc+clfile)#, delim_whitespace=True)
    nobj = np.shape(prob_mat)[0]
    nclass = np.shape(prob_mat)[1]-1 #since they have object ID as an element
    cols=prob_mat.columns.tolist()
    
    objid = prob_mat[cols[0]]
    pmat=np.array(prob_mat[cols[1:]])
    
    truth_values = pd.read_csv(loc+truthfile) #, delim_whitespace=True)
    nobj_truth = np.shape(truth_values)[0]
    nclass_truth = np.shape(truth_values)[1]-1
    truvals = np.array(truth_values[cols[1:]])
    tvec = np.where(truvals==1)[1]

    
#    pmat = prob_mat[:,1:]
    plot_cm(pmat, tvec, title, loc=loc+dataset['names'][cc]+'/')
    return pmat, tvec


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
        
def per_metric_helper(ax, n, data, metric_names, codes, shapes, colors):
    plot_n = n+1
    in_x = np.arange(len(codes))
    ax_n = ax
    n_factor = 0.1 * (plot_n - 2)
    if plot_n>1:
        ax_n = ax.twinx()
        rot_ang = 270
        label_space = 15.
    else:
        rot_ang = 90
        label_space = 0.
    if plot_n>2:
        ax_n.spines["right"].set_position(("axes", 1. + 0.1 * (plot_n-1)))
        make_patch_spines_invisible(ax_n)
        ax_n.spines["right"].set_visible(True)
    handle = ax_n.scatter(in_x+n_factor*np.ones_like(data[n]), data[n], marker=shapes[n], s=10, color=colors[n], label=metric_names[n])
    ax_n.set_ylabel(metric_names[n], rotation=rot_ang, fontsize=14, labelpad=label_space)
#     ax_n.set_ylim(0.9 * min(data[n]), 1.1 * max(data[n]))
    return(ax, ax_n, handle)

def metric_plot(dataset, metric_names, shapes, colors, choice):
    codes = dataset['names']
    data = dataset['results']
    title = dataset['label']+' results focusing on: '+str(choice)
    fileloc = dataset['dirname']+dataset['label']+'_'+str(choice)+'_results.png'
    xs = np.arange(len(codes))
    pl.clf()
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=1.)
    handles = []
    for n in range(len(metric_names)):
        (ax, ax_n, handle) = per_metric_helper(ax, n, data, metric_names, codes, shapes, colors)
        handles.append(handle)
    plt.xticks(xs, codes)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    plt.xlabel('Classifiers', fontsize=14)
    leg=plt.legend(handles, metric_names, numpoints=1, loc='lower right')
    #leg.draw_frame(False)
    plt.suptitle(title)
    plt.savefig(fileloc)
    #plt.show()
    return


for dataset in [ plasticc]: #mystery, snphotcc,
    dataset = make_file_locs(dataset)
    dataset['class_pairs'] = make_class_pairs(dataset)


for choice in choices:
    pl.clf()
    
    if choice=='All':
        print('ignoring weights for %s'%choice)
        #weights= None
        weights=np.ones(len(list))
        print(len(list), 'length of list')
    else:
        weights= np.zeros(len(list)) #1e-5*np.ones(len(list))
        ind = itemlist.index(choice) 
        print(itemlist)
        print(ind, 'check ind', choice)
        print(itemlist[ind], choice, 'checking choice')
        weights[ind]=1.0
    for dataset in [plasticc]:
        data = np.empty((len(metricslist), len(dataset['names'])))
        
        for cc, pair in enumerate(dataset['class_pairs']):
            probm, truthv = read_class_pairs(pair, dataset, cc)

            for count, metric in enumerate(metricslist):
                print(weights, 'checking huh')
                print(len(weights), 'how many weights?')
                D = getattr(proclam.metrics, metric, weights)()
                hm = D.evaluate(probm, truthv, weights)
                data[count][cc] = hm
        dataset['results'] = data

        metric_plot(dataset, metricslist, markerlist, colors, choice)

#-----------------------------------------------------

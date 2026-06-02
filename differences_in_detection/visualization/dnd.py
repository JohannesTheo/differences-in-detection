import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pywaffle import Waffle
from sklearn.metrics import confusion_matrix


def discretize_stats(stats, base=36, div=1000, decimals=0):

    # Example: Both: 28231 A: 1710 B: 899 Neiter: 5495 - TOTAL: 36335 -> We use 36x1000 blocks
    #
    # NOTE: We round values and resolve rounding errors in the D 'Neither' category so that A,B,C are
    #       represented as correct as possible.

    stats = np.round(stats / div, decimals)
    sv = sum(stats)
    if sv > base:
        stats[-1] -= (sv-base)
    if sv < base:
        stats[-1] += (base-sv)

    return stats


def plot_differences(dnd_dicts: dict, m1, m2, t=None, title='Differences in Detection (ours)'):

    c1 =  ['#a5d1fb', '#fcea98', '#f6c5c3', '#b7edb7']
    c2 =  ['#58a1ef', '#f8d04e', '#ef847e', '#7ad17a']
    labels = ['Matched GT:', 'Both', m1, m2, 'Neither']

    fig = plt.figure(figsize=(9, 2.5), dpi=100.0)
    for i, (k,v) in enumerate(list(dnd_dicts.items())[::-1]):

        ax = fig.add_subplot(10, 1, i+1)
        ax.set_aspect(aspect="equal")

        # Prepare data
        stats = np.array([len(v[_s]) for _s in ['A', 'B', 'C', 'D']])
        stats = discretize_stats(stats)

        Waffle.make_waffle(
            ax=ax,
            rows=1,
            columns=36,
            values=stats,
            colors=c2,
            #rounding_rule='ceil',  # we use our own rounding
            #title={"label": "Differences in Detection (ours)", "loc": "center", 'fontsize': 18 } if i == 0 else None
        )
        if i == 0:
            ax.set_title(title, fontsize=18, y=1.5)
        fontweight='bold' if t is not None and k == t else None
        ax.text(-0.015, 0.5, f"{k:.2f}", va='center', ha='right', fontsize=10, fontweight=fontweight, transform=ax.transAxes)

    handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor=c, markersize=16) for c in ['#ffffff',*c2]]
    fig.legend(handles, labels, ncol=5, fontsize=12, frameon=False,
               handletextpad=0.3, handlelength=1.5, columnspacing=1.5, bbox_to_anchor=(0.85, 0.1)) #,  bbox_to_anchor=(0.43, 0.03)) #  , 
    #fig.suptitle(f"Differences in Detection (ours)", fontsize=20, y=1.05)
    #plt.show()
    plt.show()


def to_counts(cat_dist_1, cat_dist_2, labels=['Cls', 'Loc', 'Both', 'Miss'], as_confusion_matrix=False):

    if as_confusion_matrix:
        return confusion_matrix(cat_dist_1, cat_dist_2, labels=labels)

    l1, c1 = np.unique(cat_dist_1, return_counts=True)
    l1_dict = { l:c for l,c in zip(l1, c1) }
    sorted_counts_1 = [l1_dict[l] for l in labels]

    l2, c2 = np.unique(cat_dist_2, return_counts=True)
    l2_dict = { l:c for l,c in zip(l2, c2) }
    sorted_counts_2 = [l2_dict[l] for l in labels]

    return sorted_counts_1, sorted_counts_2


def plot_dnd_details(dnd_dicts, m1, m2, t=0.5):

    c2 =  ['#58a1ef', '#f8d04e', '#ef847e', '#7ad17a']

    fig = plt.figure(figsize=(9, 3), dpi=100.0)
    fig.suptitle(f"Direct Comparison @ IoU={t}", fontsize=18, y=1.03)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], hspace=0.1, wspace=0.5)

    # TOP LEFT
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect(aspect="equal")
    stats = np.array([len(dnd_dicts[t][_s]) for _s in ['A', 'B', 'C', 'D']])

    Waffle.make_waffle(
            ax=ax,
            rows=3,
            columns=12,
            values=discretize_stats(stats),
            colors=c2,
            starting_location='NW',
            #rounding_rule='nearest',
            title={"label": "Matched GT", "loc": "center", 'fontsize': 12 }
        )
    #ax.text(-0.015, 0.5, f"{t:.2f}", va='center', ha='right', fontsize=10, transform=ax.transAxes)
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.0275, pos.y0, pos.width, pos.height])

    # BOTTOM LEFT
    ax = fig.add_subplot(gs[1, 0])
    len_gt = len(dnd_dicts[t]['GT'])
    cellText = [['', f'{stats[0]}',  f'{stats[0]/len_gt*100:.2f}%'],
                ['', f'+{stats[1]}', f'{stats[1]/len_gt*100:.2f}%'],
                ['', f'+{stats[2]}', f'{stats[2]/len_gt*100:.2f}%'],
                ['', f'{stats[3]}',  f'{stats[3]/len_gt*100:.2f}%']]

    ax.axis('off')
    table = ax.table(
            cellText=cellText,
            loc="center",
            cellLoc="right", 
            colLoc="center",
            colWidths=[0.15, 0.425, 0.425],
            bbox=[0.14, 0.3, 1, 1],
            )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.0)
        #cell.set_linewidth(0.5)
        if col == 0:
            cell.set_facecolor(c2[row])
            cell.set_linewidth(5)
            cell.set_edgecolor('white')

    # TOP and BOTTOM CENTER
    colors = ['#D6FFD6', '#7ad17a', '#2E6E2E']
    custom_cmap = LinearSegmentedColormap.from_list("my_cmap", colors)
    ExB = dnd_dicts[t]['categorical_distributions']['ExB']
    ExC = dnd_dicts[t]['categorical_distributions']['ExC']
    D1 = dnd_dicts[t]['categorical_distributions']['D1']
    D2 = dnd_dicts[t]['categorical_distributions']['D2']
    labels = ['Cls', 'Loc', 'Both', 'Miss']
    exb, exc = to_counts(ExB, ExC, labels=labels)
    cm = to_counts(D2, D1, labels=labels, as_confusion_matrix=True)

    ax = fig.add_subplot(gs[0, 1])
    plt.title(f"ExB Errors: {m1}")
    ax = sns.heatmap([exb], square=True, norm=LogNorm(), cmap=custom_cmap, annot=True, fmt='g', vmin=0, vmax=np.max(4),
                        xticklabels=labels, yticklabels=[], cbar=False, ax=ax)

    ax = fig.add_subplot(gs[1, 1])
    plt.title(f"ExC Errors: {m2}")
    ax = sns.heatmap([exc], square=True, norm=LogNorm(), cmap=custom_cmap, annot=True, fmt='g', vmin=0, vmax=np.max(4),
                        xticklabels=labels, yticklabels=[], cbar=False, ax=ax   )

    # RIGHT
    ax = fig.add_subplot(gs[:, 2])
    ax = sns.heatmap(cm, square=True, norm=LogNorm(), cmap=custom_cmap, annot=True, fmt='g', vmin=0, vmax=np.max(cm),
                        xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.tick_params(axis='both', which='both', length=0)
    ax.yaxis.tick_right()
    ax.set_ylabel(f"{m2}")
    ax.set_xlabel(f"{m1}")
    ax.xaxis.set_label_position('top')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')
    plt.show()

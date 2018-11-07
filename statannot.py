from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib import transforms, lines
import numpy as np
import pandas as pd
from scipy import stats



def pvalAnnotation_text(x, pvalueThresholds):
    singleValue = False
    if type(x) is np.array:
        x1 = x
    else:
        x1 = np.array([x])
        singleValue = True
    # Sort the threshold array
    pvalueThresholds = pd.DataFrame(pvalueThresholds).sort_values(by=0, ascending=False).values
    xAnnot = pd.Series(["" for _ in range(len(x1))])
    for i in range(0, len(pvalueThresholds)):
        if (i < len(pvalueThresholds)-1):
            condition = (x1 <= pvalueThresholds[i][0]) & (pvalueThresholds[i+1][0] < x1)
            xAnnot[condition] = pvalueThresholds[i][1]
        else:
            condition = x1 < pvalueThresholds[i][0]
            xAnnot[condition] = pvalueThresholds[i][1]

    return xAnnot if not singleValue else xAnnot.iloc[0]


def add_statistical_test_annotation(
        ax, df, catPairList, xlabel=None, ylabel=None, test='Mann-Whitney', order=None,
        textFormat='star', loc='inside',
        pvalueThresholds=[[1e9,"ns"], [0.05,"*"], [1e-2,"**"], [1e-3,"***"], [1e-4,"****"]],
        color='0.2', lineYOffsetAxesCoord=None, lineHeightAxesCoord=0.02, yTextOffsetPoints=1,
        linewidth=1.5, fontsize='medium', verbose=1):

    """
    """
    
    if loc not in ['inside', 'outside']:
        raise ValueError("loc value should be inside or outside.")
    if test not in ['t-test', 'Mann-Whitney']:
        raise ValueError("test name incorrect.")

    if verbose >= 1 and textFormat == 'star':
        print("pvalue annotation legend:")
        pvalueThresholds = pd.DataFrame(pvalueThresholds).sort_values(by=0, ascending=False).values
        for i in range(0, len(pvalueThresholds)):
            if (i < len(pvalueThresholds)-1):
                print('{}: {:.2e} < p <= {:.2e}'.format(pvalueThresholds[i][1], pvalueThresholds[i+1][0], pvalueThresholds[i][0]))
            else:
                print('{}: p <= {:.2e}'.format(pvalueThresholds[i][1], pvalueThresholds[i][0]))
        print()

    yStack = []
    
    if xlabel is None:
        # Guess the xlabel based on the xaxis label text
        xlabel = ax.xaxis.get_label().get_text()
    if ylabel is None:
        # Guess the xlabel based on the xaxis label text
        ylabel = ax.yaxis.get_label().get_text()
    
    xtickslabels = [t.get_text() for t in ax.xaxis.get_ticklabels()]
    
    g = df.groupby(xlabel)
    catValues = df[xlabel].unique()
    catValues = np.array(xtickslabels)
    if order is not None:
        catValues = order

    ylim = ax.get_ylim()
    yRange = ylim[1] - ylim[0]
    if loc == 'inside':
        lineYOffsetAxesCoord = 0.05
    elif loc == 'outside':
        lineYOffsetAxesCoord = 0.03
    yOffset = lineYOffsetAxesCoord*yRange

    annList = []
    for cat1, cat2 in catPairList:
        if cat1 in catValues and cat2 in catValues:
            # Get position of bars 1 and 2
            x1 = np.where(catValues == cat1)[0][0]
            x2 = np.where(catValues == cat2)[0][0]

            cat1YMax = g[ylabel].max().loc[cat1]
            cat2YMax = g[ylabel].max().loc[cat2]

            cat1Values = g.get_group(cat1)[ylabel].values
            cat2Values = g.get_group(cat2)[ylabel].values

            testShortName = ''
            if test == 'Mann-Whitney':
                u_stat, pval = stats.mannwhitneyu(cat1Values, cat2Values, alternative='two-sided')
                testShortName = 'M.W.W.'
                if verbose >= 2: print ("{} v.s. {}: MWW RankSum P_val={:.3e} U_stat={:.3e}".format(cat1, cat2, pval, u_stat))
            elif test == 't-test':
                stat, pval = stats.ttest_ind(a=cat1Values, b=cat2Values)
                testShortName = 't-test'
                if verbose >= 2: print ("{} v.s. {}: t-test independent samples, P_val=={:.3e} stat={:.3e}".format(cat1, cat2, pval, stat))

            if textFormat == 'full':
                text = "{} p < {:.2e}".format(testShortName, pval)
            elif textFormat is None:
                text = ''
            elif textFormat is 'star':
                text = pvalAnnotation_text(pval, pvalueThresholds)
            
            if loc == 'inside':
                yRef = max(cat1YMax, cat2YMax)
            elif loc == 'outside':
                yRef = ylim[1]

            if len(yStack) > 0:
                yRef2 = max(yRef, max(yStack))
            else:
                yRef2 = yRef
                
            y = yRef2 + yOffset
            h = lineHeightAxesCoord*yRange
            lineX, lineY = [x1, x1, x2, x2], [y, y + h, y + h, y]
            if loc == 'inside':
                ax.plot(lineX, lineY, lw=linewidth, c=color)
            elif loc == 'outside':        
                line = lines.Line2D(lineX, lineY, lw=linewidth, c=color, transform=ax.transData)
                line.set_clip_on(False)
                ax.add_line(line)
            
            ann = ax.annotate(text, xy=(np.mean([x1, x2]), y + h),
                              xytext=(0, yTextOffsetPoints), textcoords='offset points',
                              xycoords='data', ha='center', va='bottom', fontsize=fontsize,
                              clip_on=False, annotation_clip=False)
            annList.append(ann)
            ax.set_ylim((ylim[0], 1.1*(y + h)))
            plt.draw()
            bbox = ann.get_window_extent()
            bbox_data = bbox.transformed(ax.transData.inverted())
            yTopAnnot = bbox_data.ymax
            yStack.append(yTopAnnot)
            
        
    yStackMax = max(yStack)
    if loc == 'inside':
        ax.set_ylim((ylim[0], 1.03*yStackMax))
    elif loc == 'outside':
        ax.set_ylim((ylim[0], ylim[1]))
    
    return ax

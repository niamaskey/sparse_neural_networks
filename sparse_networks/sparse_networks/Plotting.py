import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import seaborn as sns
from itertools import cycle
import torch
from torchvision.utils import make_grid as make_grid

plt.style.use('seaborn-paper')
#plt.style.use('ggplot')

sns.set_style('ticks',
              {"font.family": "serif",
               'font.serif': ['computer modern roman'],
               }
               )
sns.set_context("paper", font_scale=2)

#Colour palletes
qualitative_colors = sns.color_palette("Set3", 10)
#colorblind = sns.color_palette("colorblind")[:5] + [sns.color_palette("colorblind")[-1]]
colorblind = sns.color_palette("colorblind")
pallete = colorblind

sns.set_palette(pallete)
sns.palplot(pallete)

thesis_textwidth_pts = 418.25368

def set_size(width = 'Thesis', fraction=2):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'Thesis':
        width = 418.25368
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    print(thesis_textwidth_pts, fig_dim)
    return fig_dim


### Plotting for thesis ###
def Figure(FigNo = 0, figsize = set_size(thesis_textwidth_pts, 2)):
    return plt.figure(FigNo, figsize = figsize)

def Plot(x, y, label, error = None, ax = plt, ls = cycle(['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]), lw = 1.4, c = 'black', **kwargs):
    sns.set_palette(pallete)

    if error is not None:
        ax.errorbar(x, y, error, capsize = 4, markeredgewidth = 1, label = label, ls = next(ls), lw = lw, color = c, **kwargs)
    elif error is None:
        ax.plot(x, y, label = label, ls = next(ls), lw = lw, c = c, **kwargs)
    
    sns.despine()
    
def Save(figure, figname, save = r'C:\Users\Niam\OneDrive - University of Tasmania\Documents\2021\Honours\Documents\Thesis\Plots'):
    figure.savefig(fname = f'{save}\{figname}.pdf', dpi = 300, bbox_inches = 'tight', transparent=True)
    figure.savefig(fname = f'{save}\{figname}.png', dpi = 300, bbox_inches = 'tight', transparent=True)
   
def weights_hist(network, ax = plt, bins = 30, title = "Histogram of Network Weights", **kwargs):
    """
    Grabs all the weights from a fully conected network and plots a histogram of their
    distribution!
    """
    sns.set_palette(pallete)
    
    weights = []
    for names, params in network.named_parameters():
      if 'weight' in names:
        parameters = params.flatten().detach().numpy()
        weights += list(parameters)     #Did not know you could concat lists like this....
    
    ax.hist(weights, bins, **kwargs)
    plt.title(title)
    
    sns.despine()

def plot_weights(network, layer, cmap = 'viridis', n_in = 28, n_nodes = (64), ncols = 8, vmin = -1, vmax = 1):
    """
    Gets weights from a specific layer in the network and plots the weights 
    leading INTO each node in that layer in a grid - i.e. plots the "Receptive fields"
    of each node in a layer. Assumes receptive fields are square - i.e. a square number of nodes
    (or inputs) from the previous layer (4, 9, 16, 25, 36, ...)
    """
    all_weights = []
    #torch.empty(n_nodes, 1, (np.sqrt(n_in)), int(np.sqrt(n_in)))
    rf_list = []
    
    #Get weights and biases from model
    l = 0
    for names, params in network.named_parameters():
        
        if 'weight' in names:
            params_in_len = params.shape[1]
            im_size = int(np.sqrt(params_in_len))
            params_out_len = int(params.shape[0])
            all_weights.append(params.reshape(params_out_len, im_size, im_size))
            l+=1
            
    layer_weights = all_weights[layer]  #Select the weights in the desired layer
    
    #Assuming square receptive fields
    im_size = layer_weights.shape[1] #Image size determined by previous layer
    nnodes = layer_weights.shape[0]     # number of nodes = number of weights in a column
    print(f'Images are size {im_size}x{im_size}, and there are {nnodes} nodes')
    
    for row in range(layer_weights.shape[0]):   #Grab receptive fields of each node
        rf = layer_weights[row, :]                      #Row = RF of node in next layer
        rf = torch.reshape(rf, (im_size, im_size))      #Reshape row into square
        rf_list.append(rf)                              #Store in a list
    #Plot
    nrows = int(np.sqrt(nnodes))
    fig = plt.figure(figsize = (10,10))
    for n in range(nnodes):
        plt.subplot(nrows, nrows, 1+n)
        plt.imshow(rf_list[n].detach())
        
        plt.setp(plt.gca().get_xticklabels(), visible=False)
        plt.setp(plt.gca().get_yticklabels(), visible=False)
        plt.gca().yaxis.set_ticks_position('none')
        plt.gca().xaxis.set_ticks_position('none')
        plt.subplots_adjust(hspace=0.05, wspace=0.04)
    
    #grid = make_grid(rf_list,
    #                 nrow = nnodes)
    #plt.imshow(grid.detach())
    #grid = ImageGrid(fig, 111,
    #             nrows_ncols = (nrows, ncols),
    #             axes_pad = 0.01)

    #for ax, im in zip(grid, rf_list):
        #ax.imshow(im.detach().numpy(), str(cmap), vmin = vmin, vmax = vmax)
        #for spine in ax.spines:
         #   ax.spines[str(spine)].set_visible(False)
 
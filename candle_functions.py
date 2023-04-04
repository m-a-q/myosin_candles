import numpy as np                 # This contains all our math functions we'll need
# This toolbox is what we'll use for reading and writing images
import skimage.io as io
# %matplotlib notebook
# This toolbox is to create our plots. Line above makes them interactive
import matplotlib.pyplot as plt
import matplotlib.patches as patches #for plotting rectanglular bounding box over features in images
import seaborn as sns              #plotting tool
import matplotlib.ticker as ticker
import colorcet as cc       #set of color maps that can be called as strings in plt
import cmasher as cmr       #set of color maps that can be called as strings in plt
from colormath.color_objects import *
from colormath.color_conversions import convert_color
from matplotlib import cm       #colormaps
from matplotlib.colors import ListedColormap      #for creating colormaps from a list of RGB values
import os       # This toolbox is a useful directory tool to see what files we have in our folder
import cv2                         # image processing toolbox
import glob as glob                # grabbing file names
import czifile                     # read in the czifile
from skimage import morphology, util, filters
from scipy import stats, optimize, ndimage               # for curve fitting 
from scipy.signal import medfilt, convolve   # Used to detect overlap
from scipy.spatial.distance import pdist, squareform
from skimage.measure import label, regionprops      # For labeling regions in thresholded images and calculating properties of labeled regions
from skimage.segmentation import clear_border       # Removes junk from the borders of thresholded images
from skimage.color import label2rgb  # Pretty display labeled images
from skimage.morphology import opening, disk, dilation, remove_small_objects, remove_small_holes     # morphology operations
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, generate_binary_structure                    # morphological operations
from skimage.feature import peak_local_max      #finds local peaks based on a local or absolute threshold
import pandas as pd  # For creating our dataframe which we will use for filtering
from pandas import DataFrame, Series #for convenience
import untangle   # for parsing the XML files
from image_plotting_tools import *




# Functions for candle analysis
def get_czifile_metadata(filename):
    '''
    Pull the metadata from the czi file. 
    '''

    # read the czifile
    czi = czifile.CziFile(filename)
    # read the metadata from the czifile
    metadata_xml = czi.metadata()
    # convert the metadata to an easily parsible structure
    try:
        XML_data = untangle.parse(metadata_xml)
    except ValueError:
        with open('temp.xml','w') as f:
            f.write(metadata_xml)
        XML_data=untangle.parse('temp.xml')
        os.remove('temp.xml')
    # pull the image data for number of rows, columns, planes, channels, timepoints
    N_rows = int(XML_data.ImageDocument.Metadata.Information.Image.SizeY.cdata)
    N_cols = int(XML_data.ImageDocument.Metadata.Information.Image.SizeX.cdata)
    # have to check whether it's a z stack first
    if hasattr(XML_data.ImageDocument.Metadata.Information.Image, 'SizeZ'):
        N_zplanes = int(XML_data.ImageDocument.Metadata.Information.Image.SizeZ.cdata)
    else:
        N_zplanes = 1
    # check whether it has multiple channels
    if hasattr(XML_data.ImageDocument.Metadata.Information.Image, 'SizeC'):
        N_channels = int(XML_data.ImageDocument.Metadata.Information.Image.SizeC.cdata)
    else:
        N_channels = 1
    # check whether it has multiple time points
    if hasattr(XML_data.ImageDocument.Metadata.Information.Image, 'SizeT'):
        N_timepoints = int(XML_data.ImageDocument.Metadata.Information.Image.SizeT.cdata)
        # pull the actual timestamps from the czi file
        for attachment in czi.attachments():
            if attachment.attachment_entry.name == 'TimeStamps':
                timestamps = attachment.data()
    else:
        N_timepoints = 1
        timestamps = []

    # get the x,y,z resolutions - multiply by 1e6 to convert it to microns
    x_micron_per_pix = float(XML_data.ImageDocument.Metadata.Scaling.Items.Distance[0].Value.cdata) * 1e6
    y_micron_per_pix = float(XML_data.ImageDocument.Metadata.Scaling.Items.Distance[1].Value.cdata) * 1e6
    z_micron_per_pix = float(XML_data.ImageDocument.Metadata.Scaling.Items.Distance[2].Value.cdata) * 1e6
    
    # create a dictionary with the data
    exp_details = {
        'N_rows' : N_rows,
        'N_cols' : N_cols,
        'N_zplanes' : N_zplanes,
        'N_channels' : N_channels,
        'N_timepoints' : N_timepoints,
        'timestamps' : timestamps,
        'x_micron_per_pix' : x_micron_per_pix,
        'y_micron_per_pix' : y_micron_per_pix,
        'z_micron_per_pix' : z_micron_per_pix
    }
    
    return exp_details

def prep_file(filename, show_results=False):
    # read in the file
    imstack = czifile.imread(filename)

    # keep only those dimensions that have the z-stack
    imstack = imstack[0, 0, 0, 0, :, :, :, 0]

    # make max projection and get the size of the image
    max_projection = np.amax(imstack, axis=0)

    # get the metadata
    exp_details = get_czifile_metadata(filename)

    if show_results:
        max_projection_fig, max_projection_axes = plt.subplots()
        max_projection_axes.imshow(max_projection, cmap='Greys', vmin=50, vmax=np.max(max_projection)*.4)
        max_projection_fig.show()

    return imstack, max_projection, exp_details

def candle_masks(imstack, show_results=False):
    im_sum = np.sum(imstack, axis=0)
    intensity_values = np.unique(im_sum.ravel())
    # reduce list of intensity values down to something manageable to speed up computation
    if len(intensity_values) > 300:
        slice_width = np.round(len(intensity_values)/300).astype('int')
        if slice_width == 0:
            slice_width = 1
        intensity_values = intensity_values[::slice_width]
    # Find the mean intensity value of the image
    intensity_mean = np.mean(im_sum)
    intensity_difference = []
    # create a zero matrix to hold our difference values
    for i,intensity in enumerate(intensity_values):
        # make a mask of pixels about a given intensity
        mask = im_sum > intensity
        intensity_difference.append(np.sum(im_sum[mask]) - intensity_mean*np.sum(mask))
    # find the maximum value of the intensity_difference and set it equal to the threshold
    max_intensity = np.argwhere(intensity_difference == np.max(intensity_difference))
    threshold = intensity_values[max_intensity[0][0]]
#     print(threshold)
    # make a mask at this threshold
    mask = im_sum > threshold * (2 / 3)
    small_object_size = 11 * 11
    # get rid of small objects
    mask = remove_small_objects(mask, small_object_size)
    mask = remove_small_holes(mask, 10000)
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    areas = []
    for region in props:
        areas.append(region.area)
    max_area_label = np.argwhere(areas == np.max(areas))
    cytoplasm_mask = labeled_mask == max_area_label[0][0] + 1
    SE = disk(5)
    cytoplasm_mask = binary_dilation(cytoplasm_mask, structure=SE)
    cell_mask = binary_fill_holes(cytoplasm_mask)
    extracellular_mask = cell_mask == False
    nuclear_mask = (cytoplasm_mask == False) * cell_mask
    if show_results:
        mask_fig, mask_ax = plt.subplots(nrows=2, ncols=2)
        mask_ax[0,0].imshow(cell_mask)
        mask_ax[0,0].set_title('Cell Mask')
        mask_ax[0,1].imshow(cytoplasm_mask)
        mask_ax[0,1].set_title('Cytoplasm Mask')
        mask_ax[1,0].imshow(extracellular_mask)
        mask_ax[1,0].set_title('Extracellular Mask')
        mask_ax[1,1].imshow(nuclear_mask)
        mask_ax[1,1].set_title('Nuclear Mask')
        for ax in mask_ax.ravel():
            ax.axis('off')
        mask_fig.show()
        
    counts, bins = np.histogram(im_sum[cytoplasm_mask], bins=150)
    bins = bins[:-1] + np.diff(bins/2)
    hist_max = np.argwhere(counts == np.max(counts))
    bg_pixel = bins[hist_max[0, 0]]
    print(bg_pixel)
    
    return cell_mask, cytoplasm_mask, extracellular_mask, nuclear_mask, bg_pixel

def find_candles(max_projection, cell_mask, show_results=False, min_value=1000):

    # Function returns local maxim
    peak_coords = peak_local_max(max_projection,
                                 min_distance=3,
                                 threshold_abs=min_value)
    

    # create a DataFrame for the peaks
    data_dict = {'labels': np.arange(len(peak_coords)).astype(int),
                 'crows': peak_coords[:, 0],
                 'ccols': peak_coords[:, 1],
                 'intensity': max_projection[peak_coords[:, 0], peak_coords[:, 1]]}
    df = pd.DataFrame(data_dict)
    
    inside = []
    for candle in range(len(df)):
        x = df.crows[candle]
        y = df.ccols[candle]
        if cell_mask[x, y] ==1:
            inside.append(1)
        else:
            inside.append(0)
    df['inside'] = inside
    df = df[df.inside == 1]
    df = df.reset_index(drop=True)

    # Find the number of total peaks
    Nobjects = len(df)
    print('Found %d Objects' % (Nobjects))
        
    if show_results:
        # Plot results
        overlay_fig, overlay_axes = plt.subplots()
        overlay_axes.imshow(max_projection, cmap='Greys', vmin=50, vmax=np.max(max_projection)*.4)
        overlay_axes.plot(df.ccols[:], df.crows[:], 'x', color='xkcd:scarlet')
        overlay_fig.show()

    
    return df, Nobjects


def find_filter_values(imstack, df, Nobjects, bg_pixel, volume_width=5):
    
    # volume width variables
    vw = volume_width
    vwl = volume_width - 1
    vwr = volume_width + 1
    z = imstack.shape[0]

    # Create empty holders for parameters
    candle_volume = []
    candle_sum_intensity = []
    # Create variable for the different faces of the cube filter (top and bottom faces = z_bg, side faces = xy_bg)
    z_bg = []
    xy_bg = []
    cell_bg = []
    bg_sub_intensity = []
    
    max_projection = np.amax(imstack, axis=0)
    nrows, ncols = max_projection.shape
    
    # Loop through each cand and create a substack with a defined volume around it (change with volume_width definition)
    # Get the intensities of the faces of the cube filter in order to empirically define a maximum intensity threshold for those faces for each data set
    # The top/bottom and side slices are separated in this analysis because larger z stacks would likely mean that there is very 
        # little background for the top and bottom z slice but the side slices likely have quite a bit (especially with an abberation)
    for candle in np.arange(0,Nobjects):
        #remove any candles that are on the edges of the image
        if df.crows[candle] > vwl and df.crows[candle] < (nrows - vwr) and df.ccols[candle] > vwl and df.ccols[candle] < (ncols - vwr):
            # select volume around the candle
            substack = imstack[:, df.crows[candle]-vw:df.crows[candle] + vwr,
                               df.ccols[candle]-vw:df.ccols[candle]+vwr]
            candle_volume.append(substack)
            bg_box = bg_pixel*(vw*2+1)**2
            cell_bg.append(bg_box)
            # Sum the intensity in that volume
            candle_sum_intensity.append(np.sum(substack))
            bg_sub_intensity.append(np.sum(substack) - bg_box)
            # Add the sum intensity of these faces to their new variables
            z_bg.append(np.sum(substack[0, :, :]) + np.sum(substack[-1, :, :]))
            xy_bg.append(np.sum(substack[:, 0, :]) + np.sum(substack[:, -1, :]) + np.sum(substack[:, :, 0]) + np.sum(substack[:, :, -1]))
        # If the candles were on the edge, their data values are set to zero (to keep the same shape for all variables)    
        else:
            candle_sum_intensity.append(0)
            candle_volume.append(0)
            z_bg.append(0)
            xy_bg.append(0)
            cell_bg.append(0)
            bg_sub_intensity.append(0)
            
    # Add columns to the dataframe
    df['sum_intensity'] = candle_sum_intensity
    df['volume'] = candle_volume
    df['z_bg'] = z_bg
    df['xy_bg'] = xy_bg
    df['cell_bg'] = cell_bg
    df['bg_sub_intensity'] = bg_sub_intensity

    
    return df, Nobjects


def save_prefilter_candle_data(df, filename):
    df.to_hdf(filename[:-4] + '_prefilterdata.hd5', key='Candles', mode='w')
    
    return


def fit_all_prefilter_distribution_act(all_prefilter_data_act, experiment_date, save_results=False):
    
    # make a histogram of the intensities for the sum intensity of z faces or xy faces
    all_prefilter_data_act= all_prefilter_data_act[all_prefilter_data_act['z_bg'].values >0]
    countsactz, binsactz = np.histogram(all_prefilter_data_act['z_bg'].values, bins=150)
    countsactxy, binsactxy = np.histogram(all_prefilter_data_act['xy_bg'].values, bins=150)

    # This gets the center of the bin instead of the edges
    binsactz = binsactz[:-1] + np.diff(binsactz/2)
    binsactxy = binsactxy[:-1] + np.diff(binsactxy/2)
    
    # Make initial guesses as to fitting parameters
    hist_maxactz = np.argwhere(countsactz == np.max(countsactz))

    p0_actz = [np.max(countsactz), binsactz[hist_maxactz[0, 0]],
             np.std(all_prefilter_data_act['z_bg'].values)]
    
    hist_maxactxy = np.argwhere(countsactxy == np.max(countsactxy))

    p0_actxy = [np.max(countsactxy), binsactxy[hist_maxactxy[0, 0]],
             np.std(all_prefilter_data_act['xy_bg'].values)]
    
    # Fit the curve
    prefilter_paramsactz, prefilter_paramsactz_covariance = optimize.curve_fit(
        gaussian_fit, binsactz, countsactz, p0_actz, bounds=(0, np.inf))

    prefilter_paramsactxy, prefilter_paramsactxy_covariance = optimize.curve_fit(
        gaussian_fit, binsactxy, countsactxy, p0_actxy, bounds=(0, np.inf))
                  
    # Create a fit line using the parameters from your fit and the original bins
    prefilter_bg_fitactz = gaussian_fit(binsactz, prefilter_paramsactz[0], prefilter_paramsactz[1], prefilter_paramsactz[2])
    prefilter_bg_fitactxy = gaussian_fit(binsactxy, prefilter_paramsactxy[0], prefilter_paramsactxy[1], prefilter_paramsactxy[2])

    # Plot result
    all_prefilter_z_distribution_fit_fig, all_prefilter_z_distribution_fit_axes = plt.subplots(figsize=(6,4))
    all_prefilter_z_distribution_fit_axes.hist(all_prefilter_data_act['z_bg'].values, bins=150, color='#d6cd0a')
    all_prefilter_z_distribution_fit_axes.plot(binsactz, prefilter_bg_fitactz, '-k')

    all_prefilter_z_distribution_fit_axes.set_xlabel('Summed Intensity of Z faces')
    all_prefilter_z_distribution_fit_axes.set_ylabel('Counts')
    all_prefilter_z_distribution_fit_axes.set_title('Combined Data')
    # all_distribution_fit_axes.text(0.95, 0.95, 'Average Value \nof a GFP fluorophore:\n' + str(np.round(params[1]/candle_size)),
    # verticalalignment='top', horizontalalignment='right',
    # transform=all_distribution_fit_axes.transAxes, fontsize=10)
    all_prefilter_z_distribution_fit_fig.show()
    # save the figure
    all_prefilter_z_distribution_fit_fig.savefig(experiment_date + '_all_prefilter_z_data_histograms.eps', dpi=150)
        
    all_prefilter_xy_distribution_fit_fig, all_prefilter_xy_distribution_fit_axes = plt.subplots(figsize=(6,4))
    all_prefilter_xy_distribution_fit_axes.hist(all_prefilter_data_act['xy_bg'].values, bins=150, color='#d6cd0a')
    all_prefilter_xy_distribution_fit_axes.plot(binsactxy, prefilter_bg_fitactxy, '-k')
    
    all_prefilter_xy_distribution_fit_axes.set_xlabel('Summed Intensity of XY faces')
    all_prefilter_xy_distribution_fit_axes.set_ylabel('Counts')
    all_prefilter_xy_distribution_fit_axes.set_title('Combined Data')
    # all_distribution_fit_axes.text(0.95, 0.95, 'Average Value \nof a GFP fluorophore:\n' + str(np.round(params[1]/candle_size)),
    # verticalalignment='top', horizontalalignment='right',
    # transform=all_distribution_fit_axes.transAxes, fontsize=10)
    all_prefilter_xy_distribution_fit_fig.show()
    
    if save_results:
        # save the figure
        all_prefilter_xy_distribution_fit_fig.savefig(experiment_date + '_all_prefilter_xy_data_histograms.eps', dpi=150)
    
    return prefilter_paramsactz,  prefilter_paramsactxy


def fit_all_prefilter_distribution(all_prefilter_data_60, all_prefilter_data_120, experiment_date, save_results=False):
    
    # make a histogram of the intensities for the sum intensity of z faces or xy faces
    counts60z, bins60z = np.histogram(all_prefilter_data_60['z_bg'].values, bins=150)
    counts60xy, bins60xy = np.histogram(all_prefilter_data_60['xy_bg'].values, bins=150)
    counts120z, bins120z = np.histogram(all_prefilter_data_120['z_bg'].values, bins=150)
    counts120xy, bins120xy = np.histogram(all_prefilter_data_120['xy_bg'].values, bins=150)

    # This gets the center of the bin instead of the edges
    bins60z = bins60z[:-1] + np.diff(bins60z/2)
    bins60xy = bins60xy[:-1] + np.diff(bins60xy/2)
    bins120z = bins120z[:-1] + np.diff(bins120z/2)
    bins120xy = bins120xy[:-1] + np.diff(bins120xy/2)

    # Make initial guesses as to fitting parameters
    hist_max60z = np.argwhere(counts60z == np.max(counts60z))
    hist_max120z = np.argwhere(counts120z == np.max(counts120z))

    p0_60z = [np.max(counts60z), bins60z[hist_max60z[0, 0]],
             np.std(all_prefilter_data_60['z_bg'].values)]
    p0_120z = [np.max(counts120z), bins120z[hist_max120z[0, 0]],
              np.std(all_prefilter_data_120['z_bg'].values)]

    hist_max60xy = np.argwhere(counts60xy == np.max(counts60xy))
    hist_max120xy = np.argwhere(counts120xy == np.max(counts120xy))

    p0_60xy = [np.max(counts60xy), bins60xy[hist_max60xy[0, 0]],
             np.std(all_prefilter_data_60['xy_bg'].values)]
    p0_120xy = [np.max(counts120xy), bins120xy[hist_max120xy[0, 0]],
              np.std(all_prefilter_data_120['xy_bg'].values)]
  
    # Fit the curve
    try:
        prefilter_params60z, prefilter_params60z_covariance = optimize.curve_fit(
        gaussian_fit, bins60z, counts60z, p0_60z, bounds=(0, np.inf))
    except RuntimeError:
        prefilter_params60z = p0_60z
    try:
        prefilter_params120z, prefilter_params120z_covariance = optimize.curve_fit(
        gaussian_fit, bins120z, counts120z, p0_120z, bounds=(0, np.inf))
    except RuntimeError:
        prefilter_params120z = p0_120z

    try:
        prefilter_params60xy, prefilter_params60xy_covariance = optimize.curve_fit(
        gaussian_fit, bins60xy, counts60xy, p0_60xy, bounds=(0, np.inf))
    except RuntimeError:
        prefilter_params60xy = p0_60xy
    
    try:
        prefilter_params120xy, prefilter_params120xy_covariance = optimize.curve_fit(
        gaussian_fit, bins120xy, counts120xy, p0_120xy, bounds=(0, np.inf))      
    except RuntimeError:
        prefilter_params120xy = p0_120xy

    # Create a fit line using the parameters from your fit and the original bins
#     prefilter_bg_fit60z = gaussian_fit(bins60z, prefilter_params60z[0], prefilter_params60z[1], prefilter_params60z[2])
    prefilter_bg_fit120z = gaussian_fit(bins120z, prefilter_params120z[0], prefilter_params120z[1], prefilter_params120z[2])
    prefilter_bg_fit60xy = gaussian_fit(bins60xy, prefilter_params60xy[0], prefilter_params60xy[1], prefilter_params60xy[2])
    prefilter_bg_fit120xy = gaussian_fit(bins120xy, prefilter_params120xy[0], prefilter_params120xy[1], prefilter_params120xy[2])
    
    # Plot result
    all_prefilter_z_distribution_fit_fig, all_prefilter_z_distribution_fit_axes = plt.subplots()
    
#     all_prefilter_z_distribution_fit_axes.plot(bins60z, prefilter_bg_fit60z, '-k')
    all_prefilter_z_distribution_fit_axes.hist(all_prefilter_data_120['z_bg'].values, bins=150, color='#270563')
    all_prefilter_z_distribution_fit_axes.hist(all_prefilter_data_60['z_bg'].values, bins=150, color='#067d54')
    all_prefilter_z_distribution_fit_axes.plot(bins120z, prefilter_bg_fit120z, '-k')

    all_prefilter_z_distribution_fit_axes.set_xlabel('Summed Intensity of Z faces')
    all_prefilter_z_distribution_fit_axes.set_ylabel('Counts')
    all_prefilter_z_distribution_fit_axes.set_title('Combined Data')
    # all_distribution_fit_axes.text(0.95, 0.95, 'Average Value \nof a GFP fluorophore:\n' + str(np.round(params[1]/candle_size)),
    # verticalalignment='top', horizontalalignment='right',
    # transform=all_distribution_fit_axes.transAxes, fontsize=10)
    all_prefilter_z_distribution_fit_fig.show()
    # save the figure
    all_prefilter_z_distribution_fit_fig.savefig(experiment_date + '_all_prefilter_z_data_histograms.eps', dpi=150)
        
    all_prefilter_xy_distribution_fit_fig, all_prefilter_xy_distribution_fit_axes = plt.subplots()

    all_prefilter_xy_distribution_fit_axes.hist(all_prefilter_data_120['xy_bg'].values, bins=150, color='#260566')
    all_prefilter_xy_distribution_fit_axes.plot(bins120xy, prefilter_bg_fit120xy, '-k')
    all_prefilter_xy_distribution_fit_axes.hist(all_prefilter_data_60['xy_bg'].values, bins=150, color='#067d54')
    all_prefilter_xy_distribution_fit_axes.plot(bins60xy, prefilter_bg_fit60xy, '-k')
 
    all_prefilter_xy_distribution_fit_axes.set_xlabel('Summed Intensity of XY faces')
    all_prefilter_xy_distribution_fit_axes.set_ylabel('Counts')
    all_prefilter_xy_distribution_fit_axes.set_title('Combined Data')
    # all_distribution_fit_axes.text(0.95, 0.95, 'Average Value \nof a GFP fluorophore:\n' + str(np.round(params[1]/candle_size)),
    # verticalalignment='top', horizontalalignment='right',
    # transform=all_distribution_fit_axes.transAxes, fontsize=10)
    all_prefilter_xy_distribution_fit_fig.show()
    
    if save_results:
        # save the figure
        all_prefilter_xy_distribution_fit_fig.savefig(experiment_date + '_all_prefilter_xy_data_histograms.eps', dpi=150)
        all_prefilter_xy_distribution_fit_fig.savefig(experiment_date + '_all_prefilter_xy_data_histograms.png', dpi=150)
    
    return prefilter_params60z, prefilter_params120z,  prefilter_params60xy, prefilter_params120xy
    
def identify_good_candles_act(imstack, df, Nobjects, intensity_max, 
                          z_max, xy_max, volume_width=5, filename=None, show_results=False):

    # volume width variables
    vw = volume_width
    vwl = volume_width - 1
    vwr = volume_width + 1
    z = imstack.shape[0]

    # Create empty holders for parameters
    candle_good_border = []
    good_intensity = []
    local_bg = []

    max_projection = np.amax(imstack, axis=0)
    nrows, ncols = max_projection.shape

    # Loop through each candle
    for candle in range(len(df)):

        # select volume around the candle
        substack = imstack[:, df.crows[candle]-vw:df.crows[candle] + vwr,
                           df.ccols[candle]-vw:df.ccols[candle]+vwr]

        # if intensity is below a threshold value mark it as good
        if np.sum(substack) < intensity_max:
            good_intensity.append(1)
        else:
            good_intensity.append(0)
            
        # If the edges of the candle volume are below a threshold intensity, mark it as good
        if df.z_bg[candle] < z_max \
                and df.xy_bg[candle] < xy_max:
            candle_good_border.append(1)
        else:
            candle_good_border.append(0)

        # sum the intensity values from the edge-pixels of the candle volume
        edge_mean = substack.copy()
        edge_mean[1:-1, 1:-1, 1:-1] = 0
        edge_mean = edge_mean.sum()
        edge_mean = edge_mean / (2*substack.shape[1]*substack.shape[2]
                                + 2*(substack.shape[0]-2)*(substack.shape[1])
                                + 2*(substack.shape[0]-2)*(substack.shape[2]-2))
        local_bg.append(edge_mean * substack.shape[0] * substack.shape[1] * substack.shape[2])


    # Add columns to the dataframe
    df['good_border'] = candle_good_border
    df['good_intensity'] = good_intensity
    df['local_bg'] = local_bg
    df['local_bg_sub_intensity'] = df.sum_intensity - df.local_bg


    # select subset of candles that are marked as good at the border and intensity
    good_candles_df = df[(df.good_intensity == 1) & (df.good_border == 1)]
    good_candles_df = good_candles_df[np.isfinite(good_candles_df['bg_sub_intensity'])]
    print('Selected %d good Objects' % (len(good_candles_df)))
    good_candles_df.head(10)

    if show_results:
        # plot where the good candles are in the image
        good_overlay_fig, good_overlay_axes = plt.subplots()
        good_overlay_axes.imshow(max_projection, cmap='Greys', vmin=50, vmax=np.max(max_projection)*.4)
        good_overlay_axes.plot(
            good_candles_df.ccols, good_candles_df.crows, 'x', color='xkcd:jungle green')
        good_overlay_fig.show()

        # plot histogram of intensities
        hist_inten_fig, hist_inten_axes = plt.subplots(figsize=(6,4))
        hist_inten_axes.hist(good_candles_df.sum_intensity -
                             good_candles_df.local_bg, bins=150)
        hist_inten_axes.set_xlim(0, 1200000)
        hist_inten_axes.set_xlabel('Candle Intensity')
        hist_inten_axes.set_ylim(0, 40)
        hist_inten_axes.set_ylabel('Counts')
        if filename is not None:
            hist_inten_axes.set_title(filename)
        hist_inten_fig.show()

#     # plot histograms of bg values
#     bg_fig, bg_axes = plt.subplots()
#     bg_axes.hist(good_candles_df.bg, bins=150)
#     bg_axes.set_title('Average background around candle')
#     bg_fig.show()

    return good_candles_df


def identify_good_candles(imstack, df, Nobjects, intensity_minimum, 
                          z_max, xy_max, volume_width=5, filename=None, show_results=True):

    # volume width variables
    vw = volume_width
    vwl = volume_width - 1
    vwr = volume_width + 1
    z = imstack.shape[0]

    # Create empty holders for parameters
    candle_good_border = []
    good_intensity = []
    bg = []

    max_projection = np.amax(imstack, axis=0)
    nrows, ncols = max_projection.shape

    # Loop through each candle
    for candle in np.arange(0, Nobjects):
        # Check that the candle isn't on the edge of the image
        if df.crows[candle] > vwl and df.crows[candle] < (nrows - vwr) and df.ccols[candle] > vwl and df.ccols[candle] < (ncols - vwr):
            # select volume around the candle
            substack = imstack[:, df.crows[candle]-vw:df.crows[candle] + vwr,
                               df.ccols[candle]-vw:df.ccols[candle]+vwr]

            # if intensity is above a threshold value mark it as good
            if np.sum(substack) > intensity_minimum:
                good_intensity.append(1)
            else:
                good_intensity.append(0)
            # If the edges of the candle volume are below a threshold intensity, mark it as good
            if (np.sum(substack[0, :, :]) + np.sum(substack[-1, :, :])) < z_max \
                    and (np.sum(substack[:, 0, :]) + np.sum(substack[:, -1, :]) + np.sum(substack[:, :, 0]) + np.sum(substack[:, :, -1])) < xy_max:
                candle_good_border.append(1)
            else:
                candle_good_border.append(0)

            # sum the intensity values from the edge-pixels of the candle volume
            edge_mean = substack.copy()
            edge_mean[1:-1, 1:-1, 1:-1] = 0
            edge_mean = edge_mean.sum()
            edge_mean = edge_mean / (2*substack.shape[1]*substack.shape[2]
                                    + 2*(substack.shape[0]-2)*(substack.shape[1])
                                    + 2*(substack.shape[0]-2)*(substack.shape[2]-2))
            bg.append(edge_mean * substack.shape[0] * substack.shape[1] * substack.shape[2])
        else:
            candle_good_border.append(0)
            good_intensity.append(0)
            bg.append(0)

    # Add columns to the dataframe
    df['good_border'] = candle_good_border
    df['good_intensity'] = good_intensity
    df['bg'] = bg

    # select subset of candles that are marked as good at the border and intensity
    good_candles_df = df[(df.good_intensity == 1) & (df.good_border == 1)]
    print('Selected %d good Objects' % (len(good_candles_df)))
    good_candles_df.head(10)

    if show_results:
        # plot where the good candles are in the image
        good_overlay_fig, good_overlay_axes = plt.subplots()
        good_overlay_axes.imshow(max_projection, cmap='Greys', vmin=50, vmax=np.max(max_projection)*.4)
        good_overlay_axes.plot(
            good_candles_df.ccols, good_candles_df.crows, 'x', color='xkcd:jungle green')
        good_overlay_fig.show()

        # plot histogram of intensities
        hist_inten_fig, hist_inten_axes = plt.subplots()
        hist_inten_axes.hist(good_candles_df.sum_intensity -
                             good_candles_df.bg, bins=150)
        hist_inten_axes.set_xlim(0, 1200000)
        hist_inten_axes.set_xlabel('Candle Intensity')
        hist_inten_axes.set_ylim(0, 40)
        hist_inten_axes.set_ylabel('Counts')
        if filename is not None:
            hist_inten_axes.set_title(filename)
        hist_inten_fig.show()

#     # plot histograms of bg values
#     bg_fig, bg_axes = plt.subplots()
#     bg_axes.hist(good_candles_df.bg, bins=150)
#     bg_axes.set_title('Average background around candle')
#     bg_fig.show()

    return good_candles_df


# Fit a double gaussian curve
def doublegaussian_fit(x, amp, x0, sig, amp_2, x0_2, sig_2):
    return amp * np.exp(-1/2 * ((x - x0) / sig) ** 2) + \
           amp_2 * np.exp(-1/2 * ((x - x0_2) / sig_2) ** 2)


# Fit a gaussian curve
def gaussian_fit(x, amp, x0, sig):
    return amp * np.exp(-1/2 * ((x - x0) / sig) ** 2)


def fit_candle_distribution(good_candles_df, candle_size, filename):
    # make a histogram of the intensities
    counts, bins = np.histogram(good_candles_df.sum_intensity - good_candles_df.bg, bins=150)

    # This gets the center of the bin instead of the edges
    bins = bins[:-1] + np.diff(bins/2)

    # Make initial guesses as to fitting parameters
    hist_max = np.argwhere(counts == np.max(counts))
    p0 = [np.max(counts), bins[hist_max[0, 0]], np.std(
        good_candles_df.sum_intensity - good_candles_df.bg)]

    # Fit the curve
    params, params_covariance = optimize.curve_fit(
        gaussian_fit, bins, counts, p0)

    # Create a fit line using the parameters from your fit and the original bins
    bg_fit = gaussian_fit(bins, params[0], params[1], params[2])
    print('Fit Parameters : ', params)
    print('Average Value of a GFP fluorophore : ', params[1]/candle_size)

    # Plot result
    distribution_fit_fig, distribution_fit_axes = plt.subplots()
    distribution_fit_axes.plot(bins, counts, '-ok')
    distribution_fit_axes.plot(bins, bg_fit, '-r')
    distribution_fit_axes.set_xlabel('Summed Intensity Candle - Background Subtracted')
    distribution_fit_axes.set_ylabel('Counts')
    distribution_fit_axes.set_title(filename)
    distribution_fit_axes.text(0.95, 0.95, 'Average Value \nof a GFP fluorophore:\n' + str(np.round(params[1]/candle_size)),
                               verticalalignment='top', horizontalalignment='right',
                               transform=distribution_fit_axes.transAxes, fontsize=10)
    distribution_fit_fig.show()
    # save the figure
    distribution_fit_fig.savefig(filename[:-4] + '_fit.eps', dpi=150)
    distribution_fit_fig.savefig(filename[:-4] + '_fit.png', dpi=150)

    return params


def save_candle_properties(good_candles_df, filename):
    good_candles_df.to_hdf(filename[:-4] + '_data.hd5', key='Candles', mode='w')
                           
    return
                           
def fit_allcandles_distribution_act(all_data_act, all_data_60, all_data_120, experiment_date, save_results=True):
    # make a histogram of the intensities
#     bg_subtract_intensity_60 = all_data_60['sum_intensity'].values - all_data_60['bg'].values
#     bg_subtract_intensity_120 = all_data_120['sum_intensity'].values - all_data_120['bg'].values
    
    countsact, binsact = np.histogram(
        all_data_act['bg_sub_intensity'].values, bins=150)    
    counts60, bins60 = np.histogram(
        all_data_60['bg_sub_intensity'].values, bins=150)
    counts120, bins120 = np.histogram(
        all_data_120['bg_sub_intensity'].values, bins=150)

    
    # This gets the center of the bin instead of the edges
    binsact = binsact[:-1] + np.diff(binsact/2)
    bins60 = bins60[:-1] + np.diff(bins60/2)
    bins120 = bins120[:-1] + np.diff(bins120/2)


    # Make initial guesses as to fitting parameters
    hist_maxact = np.argwhere(countsact == np.max(countsact))
    hist_max60 = np.argwhere(counts60 == np.max(counts60))
    hist_max120 = np.argwhere(counts120 == np.max(counts120))

    p0_act = [np.max(countsact), binsact[hist_maxact[0, 0]],
             np.std(all_data_act['bg_sub_intensity'].values)]
    p0_60 = [np.max(counts60), bins60[hist_max60[0, 0]],
             np.std(all_data_60['bg_sub_intensity'].values)]
    p0_120 = [np.max(counts120), bins120[hist_max120[0, 0]],
              np.std(all_data_120['bg_sub_intensity'].values)]


    # Fit the curve
    try:
        paramsact, paramsact_covariance = optimize.curve_fit(
        gaussian_fit, binsact, countsact, p0_act, bounds=(0, np.inf))
    except RuntimeError:
        paramsact = p0_act
    paramsact = p0_act
    params60, params60_covariance = optimize.curve_fit(
        gaussian_fit, bins60, counts60, p0_60, bounds=(0, np.inf))
    params120, params120_covariance = optimize.curve_fit(
        gaussian_fit, bins120, counts120, p0_120, bounds=(0, np.inf))
        

    
    # Create a fit line using the parameters from your fit and the original bins
    bg_fitact = gaussian_fit(binsact, paramsact[0], paramsact[1], paramsact[2])
    bg_fit60 = gaussian_fit(bins60, params60[0], params60[1], params60[2])
    bg_fit120 = gaussian_fit(bins120, params120[0], params120[1], params120[2])

    
    # Plot result
    all_distribution_fit_fig, all_distribution_fit_axes = plt.subplots(figsize=(15,10))
    all_distribution_fit_axes.hist(all_data_120['bg_sub_intensity'].values, bins=150, color='#270563', )
    all_distribution_fit_axes.plot(bins120, bg_fit120, '-k')
    all_distribution_fit_axes.hist(all_data_60['bg_sub_intensity'].values, bins=150, color='#067d54')
    all_distribution_fit_axes.plot(bins60, bg_fit60, '-k')
    all_distribution_fit_axes.hist(all_data_act['bg_sub_intensity'].values, bins=150, color='#d6cd0a')
    all_distribution_fit_axes.plot(binsact, bg_fitact, '-k')
    all_distribution_fit_axes.hist(all_data_120['bg_sub_intensity'].values, bins=150, color='#270563')
    all_distribution_fit_axes.plot(bins120, bg_fit120, '-k')
    plt.xlim(-100,500000)
    
    
    act_distribution_fit_fig, act_distribution_fit_axes = plt.subplots(figsize=(6,4))
    act_distribution_fit_axes.hist(all_data_act['bg_sub_intensity'].values, bins=150, color='#d6cd0a')
    act_distribution_fit_axes.plot(binsact, bg_fitact, '-k')
    plt.xlim(0,10000)
    plt.ylim(0,10)

    all_distribution_fit_axes.set_xlabel('Summed Intensity Candle - Background Subtracted')
    all_distribution_fit_axes.set_ylabel('Counts')
    all_distribution_fit_axes.set_title('Combined Data')
    # all_distribution_fit_axes.text(0.95, 0.95, 'Average Value \nof a GFP fluorophore:\n' + str(np.round(params[1]/candle_size)),
    # verticalalignment='top', horizontalalignment='right',
    # transform=all_distribution_fit_axes.transAxes, fontsize=10)
    all_distribution_fit_fig.show()
    


    x = np.array([1, 60, 120])
    y = np.array([paramsact[1], params60[1], params120[1]])
    yerr = np.array([paramsact[2], params60[2], params120[2]])

    # Generated linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope*x+intercept

    #The slope is the value of a GFP so define the single_GFP_value
    single_GFP_value = np.round(slope)

    candle_calibration_fig, candle_calibration_axes = plt.subplots(figsize=(6,4))
    candle_calibration_axes.errorbar(x, y, yerr=yerr, fmt='s', mec='#067d54', mfc='#d6cd0a', mew=2, ecolor='#270563')
    candle_calibration_axes.plot(x, line, 'k')
    candle_calibration_axes.set_title(
        'Candle Calibration\n' + 'y = ' + str(np.round(slope)) + 'X + ' + str(np.round(intercept)))
    candle_calibration_axes.text(0.05, 0.95, 'R-squared:' + str(r_value**2),
                                 verticalalignment='top', horizontalalignment='left',
                                 transform=candle_calibration_axes.transAxes, fontsize=12)
    candle_calibration_axes.set_xlabel('# of GFP')
    candle_calibration_axes.set_ylabel('Intensity')
    candle_calibration_fig.show()
    
    if save_results:
        # save the figure
        all_distribution_fit_fig.savefig(experiment_date + '_all_candle_histograms.eps', dpi=150)
        all_distribution_fit_fig.savefig(experiment_date + '_all_candle_histograms.png', dpi=150)
        # save the figure
        candle_calibration_fig.savefig(experiment_date + '_all_candle_calibration.eps', dpi=150)
        candle_calibration_fig.savefig(experiment_date + '_all_candle_calibration.png', dpi=150)

    return paramsact, params60, params120, single_GFP_value


def fit_allcandles_distribution(all_data_60, all_data_120, experiment_date, save_results=True):
    # make a histogram of the intensities
#     bg_subtract_intensity_60 = all_data_60['sum_intensity'].values - all_data_60['bg'].values
#     bg_subtract_intensity_120 = all_data_120['sum_intensity'].values - all_data_120['bg'].values
    counts60, bins60 = np.histogram(
        all_data_60['sum_intensity'].values, bins=150)
    counts120, bins120 = np.histogram(
        all_data_120['sum_intensity'].values, bins=150)

    
    # This gets the center of the bin instead of the edges
    bins60 = bins60[:-1] + np.diff(bins60/2)
    bins120 = bins120[:-1] + np.diff(bins120/2)


    # Make initial guesses as to fitting parameters
    hist_max60 = np.argwhere(counts60 == np.max(counts60))
    hist_max120 = np.argwhere(counts120 == np.max(counts120))

    p0_60 = [np.max(counts60), bins60[hist_max60[0, 0]],
             np.std(all_data_60['sum_intensity'].values)]
    p0_120 = [np.max(counts120), bins120[hist_max120[0, 0]],
              np.std(all_data_120['sum_intensity'].values)]


    # Fit the curve
    params60, params60_covariance = optimize.curve_fit(
        gaussian_fit, bins60, counts60, p0_60, bounds=(0, inf))
    params120, params120_covariance = optimize.curve_fit(
        gaussian_fit, bins120, counts120, p0_120, bounds=(0, inf))

    
    # Create a fit line using the parameters from your fit and the original bins
    bg_fit60 = gaussian_fit(bins60, params60[0], params60[1], params60[2])
    bg_fit120 = gaussian_fit(bins120, params120[0], params120[1], params120[2])

    
    # Plot result
    all_distribution_fit_fig, all_distribution_fit_axes = plt.subplots()
    all_distribution_fit_axes.hist(all_data_60['sum_intensity'].values, bins=150, color='xkcd:light navy')
    all_distribution_fit_axes.plot(bins60, bg_fit60, '-k')
    all_distribution_fit_axes.hist(all_data_120['sum_intensity'].values, bins=150, color='xkcd:violet red')
    all_distribution_fit_axes.plot(bins120, bg_fit120, '-k')

    all_distribution_fit_axes.set_xlabel('Summed Intensity Candle - Background Subtracted')
    all_distribution_fit_axes.set_ylabel('Counts')
    all_distribution_fit_axes.set_title('Combined Data')
    # all_distribution_fit_axes.text(0.95, 0.95, 'Average Value \nof a GFP fluorophore:\n' + str(np.round(params[1]/candle_size)),
    # verticalalignment='top', horizontalalignment='right',
    # transform=all_distribution_fit_axes.transAxes, fontsize=10)
    all_distribution_fit_fig.show()


    x = np.array([0, 60, 120])
    y = np.array([0, params60[1], params120[1]])
    yerr = np.array([0, params60[2], params120[2]])

    # Generated linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope*x+intercept

    #The slope is the value of a GFP so define the single_GFP_value
    single_GFP_value = np.round(slope)

    candle_calibration_fig, candle_calibration_axes = plt.subplots()
    candle_calibration_axes.errorbar(x, y, yerr=yerr, fmt='s', ecolor='xkcd:darkish purple')
    candle_calibration_axes.plot(x, line, 'k')
    candle_calibration_axes.set_title(
        'Candle Calibration\n' + 'y = ' + str(np.round(slope)) + 'X + ' + str(np.round(intercept)))
    candle_calibration_axes.text(0.05, 0.95, 'R-squared:' + str(r_value**2),
                                 verticalalignment='top', horizontalalignment='left',
                                 transform=candle_calibration_axes.transAxes, fontsize=10)
    candle_calibration_axes.set_xlabel('# of GFP')
    candle_calibration_axes.set_ylabel('Intensity')
    candle_calibration_fig.show()

    if save_results:
        # save the figure
        all_distribution_fit_fig.savefig(experiment_date + '_all_candle_histograms.eps', dpi=150)
        all_distribution_fit_fig.savefig(experiment_date + '_all_candle_histograms.png', dpi=150)
        # save the figure
        candle_calibration_fig.savefig(experiment_date + '_all_candle_calibration.eps', dpi=150)
        candle_calibration_fig.savefig(experiment_date + '_all_candle_calibration.png', dpi=150)

    return params60, params120, single_GFP_value
    
def identify_good_maxima(imstack, df, Nobjects, intensity_minimum, 
                          volume_width=5, z_max = 2500, filename=None):

    # volume width variables
    vw = volume_width
    vwl = volume_width - 1
    vwr = volume_width + 1
    z = imstack.shape[0]

    # Create empty holders for parameters
    good_border = []
    good_intensity = []
    bg = []

    max_projection = np.amax(imstack, axis=0)
    nrows, ncols = max_projection.shape

    # Loop through each candle
    for maxima in np.arange(0, Nobjects):
        # Check that the candle isn't on the edge of the image
        if df.crows[maxima] > vwl and df.crows[maxima] < (nrows - vwr) and df.ccols[maxima] > vwl and df.ccols[maxima] < (ncols - vwr):
            # select volume around the candle
            substack = imstack[:, df.crows[maxima]-vw:df.crows[maxima] + vwr,
                               df.ccols[maxima]-vw:df.ccols[maxima]+vwr]

            # if intensity is above a threshold value mark it as good
            if np.sum(substack) > intensity_minimum:
                good_intensity.append(1)
            else:
                good_intensity.append(0)
            # If the edges of the candle volume are below a threshold intensity, mark it as good
            if (np.sum(substack[0, :, :]) + np.sum(substack[-1, :, :])) < z_max:
                good_border.append(1)
            else:
                good_border.append(0)

            # sum the intensity values from the edge-pixels of the candle volume
            edge_mean = substack.copy()
            edge_mean[1:-1, 1:-1, 1:-1] = 0
            edge_mean = edge_mean.sum()
            edge_mean = edge_mean / (2*substack.shape[1]*substack.shape[2]
                                    + 2*(substack.shape[0]-2)*(substack.shape[1])
                                    + 2*(substack.shape[0]-2)*(substack.shape[2]-2))
            bg.append(edge_mean * substack.shape[0] * substack.shape[1] * substack.shape[2])
        else:
            good_border.append(0)
            good_intensity.append(0)
            bg.append(0)

    # Add columns to the dataframe
    df['good_border'] = good_border
    df['good_intensity'] = good_intensity
    df['bg'] = bg

def identify_good_maxima(imstack, df, Nobjects, intensity_minimum, volume_width=5, z_max = 2500, filename=None, show_results=True):
    # volume width variables
    vw = volume_width
    vwl = volume_width - 1
    vwr = volume_width + 1
    z = imstack.shape[0]

    # Create empty holders for parameters
    good_border = []
    good_intensity = []
    bg = []

    max_projection = np.amax(imstack, axis=0)
    nrows, ncols = max_projection.shape

    # Loop through each candle
    for maxima in np.arange(0, Nobjects):
        # Check that the candle isn't on the edge of the image
        if df.crows[maxima] > vwl and df.crows[maxima] < (nrows - vwr) and df.ccols[maxima] > vwl and df.ccols[maxima] < (ncols - vwr):
            # select volume around the candle
            substack = imstack[:, df.crows[maxima]-vw:df.crows[maxima] + vwr,
                               df.ccols[maxima]-vw:df.ccols[maxima]+vwr]

            # if intensity is above a threshold value mark it as good
            if np.sum(substack) > intensity_minimum:
                good_intensity.append(1)
            else:
                good_intensity.append(0)
            # If the edges of the candle volume are below a threshold intensity, mark it as good
            if (np.sum(substack[0, :, :]) + np.sum(substack[-1, :, :])) < z_max:
                good_border.append(1)
            else:
                good_border.append(0)

            # sum the intensity values from the edge-pixels of the candle volume
            edge_mean = substack.copy()
            edge_mean[1:-1, 1:-1, 1:-1] = 0
            edge_mean = edge_mean.sum()
            edge_mean = edge_mean / (2*substack.shape[1]*substack.shape[2]
                                    + 2*(substack.shape[0]-2)*(substack.shape[1])
                                    + 2*(substack.shape[0]-2)*(substack.shape[2]-2))
            bg.append(edge_mean * substack.shape[0] * substack.shape[1] * substack.shape[2])
        else:
            good_border.append(0)
            good_intensity.append(0)
            bg.append(0)

    # Add columns to the dataframe
    df['good_border'] = good_border
    df['good_intensity'] = good_intensity
    df['bg'] = bg


    # select subset of candles that are marked as good at the border and intensity
    good_maxima_df = df[(df.good_intensity == 1) & (df.good_border == 1)]
    print('Selected %d good Objects' % (len(good_maxima_df)))
    good_maxima_df.head(10)

    if show_results:
        # plot where the good candles are in the image
        good_overlay_fig, good_overlay_axes = plt.subplots()
        good_overlay_axes.imshow(max_projection, cmap='Greys', vmin=50, vmax=800)
        good_overlay_axes.plot(
            good_maxima_df.ccols, good_maxima_df.crows, 'x', color='xkcd:bright purple')
        good_overlay_fig.show()

        # plot histogram of intensities
        hist_inten_fig, hist_inten_axes = plt.subplots()
        hist_inten_axes.hist(good_maxima_df.sum_intensity -
                             good_maxima_df.bg, bins=150)
        hist_inten_axes.set_xlim(0, 1200000)
        hist_inten_axes.set_xlabel('Intensity')
        hist_inten_axes.set_ylim(0, 40)
        hist_inten_axes.set_ylabel('Counts')
        if filename is not None:
            hist_inten_axes.set_title(filename)
        hist_inten_fig.show()

#     # plot histograms of bg values
#     bg_fig, bg_axes = plt.subplots()
#     bg_axes.hist(good_candles_df.bg, bins=150)
#     bg_axes.set_title('Average background around candle')
#     bg_fig.show()

    return good_maxima_df


def filament_finder(good_maxima_df, micron_per_pixel=0.043, min_dist=0.2, max_dist=0.4):
    
    # calculate pairwise distance between points
    pairwise_distances = squareform(pdist(good_maxima_df[['crows', 'ccols']]))
    pairwise_distances *= micron_per_pixel

    # potential neighbors are within min_dist and max_dist (in micrometer)
    neighbor_condition = np.logical_and(min_dist < pairwise_distances, pairwise_distances < max_dist)

    # assign neighbors
    neighbors = []
    for i in range(len(good_maxima_df)):
        neighbors.append(np.where(neighbor_condition[i])[0])
    good_maxima_df['neighbors'] = neighbors

    # select all peaks with neighbors
    good_maxima_df = good_maxima_df.reset_index(drop=True)
    peak_labels = good_maxima_df[good_maxima_df['neighbors'].apply(len) > 0].index
    
    # Assign labels to the individual filaments by iterating over the points and their respective neighbors
    good_maxima_df['filament'] = 0

    def set_filament_iteratively(df, label, filament_nr):
        '''Helper function to iteratively set filament number of label and neighbors'''
        if df.loc[label, 'filament'] == 0:
            df.loc[label, 'filament'] = filament_nr

            for neighbor in df.loc[label, 'neighbors']:
                set_filament_iteratively(df, neighbor, filament_nr)
    
    filament_nr = 0
    for label in peak_labels:
        if good_maxima_df.loc[label, 'filament'] == 0:
            filament_nr += 1
            set_filament_iteratively(good_maxima_df, label, filament_nr)

    print(f'Found {filament_nr} individual filaments')

    return good_maxima_df


def good_filament_finder(filename, good_maxima_df, imstack, gfp=1731, pad = 5):
 
    # Group good_maxima_df by the individual filament labels
    unique_filaments = good_maxima_df[good_maxima_df['filament'] != 0.].groupby('filament')

    # Create DataFrame for results
    good_filament_df = pd.DataFrame(index = np.sort(good_maxima_df['filament'].unique())[1:])

    # assign filament label, number of maxima and center coordinates
    good_filament_df['filament'] = np.sort(good_maxima_df['filament'].unique())[1:]
    good_filament_df['n_maxima'] = unique_filaments['labels'].count()
    good_filament_df[['centrows', 'centcols']] = unique_filaments[['crows', 'ccols']].mean()

    # find individual row and col position of the peaks
    maxrows, maxcols = [], []
    for i, filament in unique_filaments:
        maxrows.append(filament['crows'].values)
        maxcols.append(filament['ccols'].values)
    good_filament_df['maxrows'] = maxrows
    good_filament_df['maxcols'] = maxcols

    # Find edge points, the bounding box and volume
    good_filament_df['min_row'] = (good_filament_df['maxrows'].apply(np.min) - pad)
    good_filament_df['max_row'] = (good_filament_df['maxrows'].apply(np.max) + pad)
    good_filament_df['min_col'] = (good_filament_df['maxcols'].apply(np.min) - pad)
    good_filament_df['max_col'] = (good_filament_df['maxcols'].apply(np.max) + pad)

    bboxs = []
    volumes = []
    for i, filament in good_filament_df.iterrows():
        bboxs.append(np.array([[filament['min_col'], filament['min_row']],
                               [filament['min_col'], filament['max_row']],
                               [filament['max_col'], filament['max_row']],
                               [filament['max_col'], filament['min_row']],
                               [filament['min_col'], filament['min_row']]]))
        volumes.append(imstack[:, 
                               int(filament['min_row']):int(filament['max_row'])+1, 
                               int(filament['min_col']):int(filament['max_col'])+1])
    good_filament_df['bbox'] = bboxs
    good_filament_df['volumes'] = volumes

    # calculate intensity
    good_filament_df['intensity'] = good_filament_df['volumes'].apply(np.sum)
    good_filament_df['ngfp'] = good_filament_df['intensity'] / gfp
    good_filament_df['monomers'] = good_filament_df['intensity'] / gfp / 2

    # Select filaments with 1 < N < 5 peaks
    good_filament_df = good_filament_df[good_filament_df.n_maxima > 1]
    good_filament_df = good_filament_df[good_filament_df.n_maxima < 5]
    good_filament_df = good_filament_df.reset_index(drop=True)


    # Plot results color-coded for number of maxima in each structure
    max_projection = np.amax(imstack, axis=0)
    cluster_maxima_fig, cluster_maxima_axes = plt.subplots(figsize=(6,6))
    cluster_maxima_axes.imshow(max_projection, cmap='Greys', vmin=50, vmax=1000)

    for i, filament in good_filament_df.iterrows():
        if filament['n_maxima'] == 2:
            cluster_maxima_axes.plot(filament['maxcols'],
                          filament['maxrows'],
                          color='xkcd:french blue', marker='o')
        elif filament['n_maxima'] == 3:
            cluster_maxima_axes.plot(filament['maxcols'],
                          filament['maxrows'],
                          color='xkcd:amethyst', marker='o')
        elif filament['n_maxima'] == 4:
            cluster_maxima_axes.plot(filament['maxcols'],
                          filament['maxrows'],
                          color='xkcd:violet red', marker='o')
        cluster_maxima_axes.plot(filament['bbox'][:, 0],
                          filament['bbox'][:, 1], 
                          color='xkcd:very dark blue')
#     cluster_maxima_fig.show()
    
    # save the figure
    cluster_maxima_fig.savefig(filename[:-4] +'_cluster_maxima.eps', dpi=150)

    # Plot results color-coded 'heat map' for number of monomers in each structure
    max_projection = np.amax(imstack, axis=0)
    cluster_monomers_fig, cluster_monomers_axes = plt.subplots(figsize=(6,6))
    cluster_monomers_axes.imshow(max_projection, cmap='Greys', vmin=50, vmax=1000)

    for i, filament in good_filament_df.iterrows():
        if (filament['monomers'] < 30):
            cluster_monomers_axes.plot(filament['maxcols'],
                          filament['maxrows'],
                          color='xkcd:sunshine yellow', marker='o')
        elif (filament['monomers'] >= 30 and filament['monomers'] < 60):
            cluster_monomers_axes.plot(filament['maxcols'],
                          filament['maxrows'],
                          color='xkcd:yellowish orange', marker='o')
        elif (filament['monomers'] >= 60 and filament['monomers'] < 120):
            cluster_monomers_axes.plot(filament['maxcols'],
                          filament['maxrows'],
                          color='xkcd:bright orange', marker='o')
        elif (filament['monomers'] > 120):
            cluster_monomers_axes.plot(filament['maxcols'],
                          filament['maxrows'],
                          color='xkcd:scarlet', marker='o')
        cluster_monomers_axes.plot(filament['bbox'][:, 0],
                          filament['bbox'][:, 1], 
                          color='xkcd:claret')
#     cluster_monomers_fig.show()
    
#     # save the figure
    cluster_monomers_fig.savefig(filename[:-4] +'_cluster_monomers.eps', dpi=150)


    return good_filament_df


def save_filament_properties(good_filament_df, filename):
    good_filament_df.to_hdf(filename[:-4] + '_data.hd5', key='Filaments', mode='w')
                           
    return


def save_filament_excel(good_filament_df, filename):
    # writing to Excel 
    datatoexcel = pd.ExcelWriter(filename[:-4] + '_data.xlsx') 
  
    # write DataFrame to excel 
    good_filament_df.to_excel(datatoexcel) 
  
    # save the excel 
    datatoexcel.save() 
    
    return

def plot_filament_data(good_filament_df, filename):
    good_filament_df = good_filament_df.astype({"n_maxima":'category'})
#     ax = sns.boxplot(x="monomers", y="n_maxima", data=good_filament_df, whis=np.inf)
#     ax = sns.swarmplot(x="monomers", y="n_maxima", data=good_filament_df, color=".2")
    plt.figure()
    ax = sns.swarmplot(x=good_filament_df['n_maxima'], y=good_filament_df['monomers'], color='black', edgecolor="gray", alpha = 0.5, size = 5)
    ax = sns.boxplot(x=good_filament_df['n_maxima'], y=good_filament_df['monomers'], palette = "cool", showfliers = False)
    
    return

def save_all_filament_properties(all_filament_data, experiment_date):
    all_filament_data.to_hdf(experiment_date + '_all_filament_data.hd5', key='Filaments', mode='w')
                           
    return



def plot_all_data(all_filament_data, experiment_date):
    all_filament_data = all_filament_data.astype({"n_maxima":'category'})
    sns.set(rc={'figure.figsize':(7,7)})
    plt.figure()
    ax = sns.swarmplot(x=all_filament_data['n_maxima'], y=all_filament_data['monomers'], color='black', edgecolor="gray", alpha = 0.5, size = 5)
    ax = sns.boxplot(x=all_filament_data['n_maxima'], y=all_filament_data['monomers'], palette = "cool", showfliers = False)
    ax.set(xlabel='Number of Maxima', ylabel='Monomers')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    # save the figure
    fig = ax.get_figure()
    fig.savefig(experiment_date + 'all_filament_data.png', dpi=150)
    return

def save_all_experiment_properties(all_experiment_data):
    all_experiment_data.to_hdf('all_experiment_data.hd5', key='Filaments', mode='w')
                           
    return

def plot_all_experiments(all_experiment_data):
    all_experiment_data = all_experiment_data.astype({"n_maxima":'category'})
#     sns.set(rc={'figure.figsize':(10,7)})
    sns.set_context("talk", font_scale=1)
    plt.figure(figsize=(10,7))
    ax = sns.swarmplot(x=all_experiment_data['n_maxima'], y=all_experiment_data['monomers'], hue=all_experiment_data['date'], dodge=True, palette = "gray_r", edgecolor="gray", alpha = 0.5, size = 5)
    ax = sns.boxplot(x=all_experiment_data['n_maxima'], y=all_experiment_data['monomers'], hue=all_experiment_data['date'], dodge=True,palette = "ocean_r", showfliers = False)
    ax.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
    plt.tight_layout()
    ax.set(xlabel='Number of Maxima', ylabel='Monomers')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    # save the figure
    fig = ax.get_figure()
    fig.savefig('all_experiment_data.png', dpi=150)
    return

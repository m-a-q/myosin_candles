import numpy as np                 # This contains all our math functions we'll need
# This toolbox is what we'll use for reading and writing images
import skimage.io as io
# %matplotlib notebook
# This toolbox is to create our plots. Line above makes them interactive
import matplotlib.pyplot as plt
# This toolbox is a useful directory tool to see what files we have in our folder
import os
import cv2                         # image processing toolbox
import glob as glob                # grabbing file names
from mpl_toolkits import mplot3d   # for making a 3D surface plot
import czifile                     # read in the czifile
from scipy import stats
from scipy import optimize, ndimage               # for curve fitting
from scipy.signal import convolve   # Used to detect overlap
from scipy.spatial.distance import pdist, squareform
from skimage.measure import label  # For labeling regions in thresholded images
# For calculating properties of labeled regions
from skimage.measure import regionprops
# Removes junk from the borders of thresholded images
from skimage.segmentation import clear_border
from skimage.color import label2rgb  # Pretty display labeled images
# Clean up small objects or holes in our image
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.feature import peak_local_max
import pandas as pd  # For creating our dataframe which we will use for filtering


# Functions for candle analysis
def prep_file(filename):
    # read in the file
    imstack = czifile.imread(filename)

    # keep only those dimensions that have the z-stack
    imstack = imstack[0, 0, 0, 0, :, :, :, 0]

    # make max projection and get the size of the image
    max_projection = np.amax(imstack, axis=0)
    nrows, ncols = max_projection.shape
    max_projection_fig, max_projection_axes = plt.subplots()
    max_projection_axes.imshow(max_projection, cmap='Greys', vmin=50, vmax=800)
    max_projection_fig.show()

    return imstack, max_projection, nrows, ncols


def find_candles(max_projection, min_value=1000):

    # Function returns local maxim
    peak_coords = peak_local_max(max_projection,
                                 min_distance=3,
                                 threshold_abs=min_value)

    # Find the number of total peaks
    Nobjects = len(peak_coords)
    print('Found %d Objects' % (Nobjects))

    # create a DataFrame for the peaks
    data_dict = {'labels': np.arange(len(peak_coords)).astype(int),
                 'crows': peak_coords[:, 0],
                 'ccols': peak_coords[:, 1],
                 'intensity': max_projection[peak_coords[:, 0], peak_coords[:, 1]]}
    df = pd.DataFrame(data_dict)

    # Plot results
    overlay_fig, overlay_axes = plt.subplots()
    overlay_axes.imshow(max_projection, cmap='Greys', vmin=50, vmax=800)
    overlay_axes.plot(peak_coords[:, 1], peak_coords[:, 0], 'xr')
    overlay_fig.show()

    return df, Nobjects


def identify_good_candles(imstack, df, Nobjects, intensity_minimum, 
                          volume_width=5, min_val=2500, filename=None):

    # volume width variables
    vw = volume_width
    vwl = volume_width - 1
    vwr = volume_width + 1
    z = imstack.shape[0]

    # Create empty holders for parameters
    candle_good_border = []
    candle_sum_intensity = []
    good_intensity = []
    candle_volume = []
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
            candle_volume.append(substack)
            # Sum the intensity in that volume
            candle_sum_intensity.append(np.sum(substack))
            # if intensity is above a threshold value mark it as good
            if np.sum(substack) > intensity_minimum:
                good_intensity.append(1)
            else:
                good_intensity.append(0)
            # If the edges of the candle volume are below a threshold intensity, mark it as good
            if np.sum(substack[0, :, :]) < min_val and np.sum(substack[-1, :, :]) < min_val \
                    and np.sum(substack[:, 0, :]) < min_val and np.sum(substack[:, -1, :]) < min_val \
                    and np.sum(substack[:, :, 0]) < min_val and np.sum(substack[:, :, -1]) < min_val:
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
            candle_sum_intensity.append(0)
            good_intensity.append(0)
            candle_volume.append(0)
            bg.append(0)

    # Add columns to the dataframe
    df['sum_intensity'] = candle_sum_intensity
    df['good_border'] = candle_good_border
    df['good_intensity'] = good_intensity
    df['volume'] = candle_volume
    df['bg'] = bg

    # select subset of candles that are marked as good at the border and intensity
    good_candles_df = df[(df.good_intensity == 1) & (df.good_border == 1)]
    print('Selected %d good Objects' % (len(good_candles_df)))

    # plot where the good candles are in the image
    good_overlay_fig, good_overlay_axes = plt.subplots()
    good_overlay_axes.imshow(max_projection, cmap='Greys', vmin=50, vmax=800)
    good_overlay_axes.plot(
        good_candles_df.ccols, good_candles_df.crows, 'x', color='xkcd:bright purple')
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

    # plot histograms of bg values
    bg_fig, bg_axes = plt.subplots()
    bg_axes.hist(good_candles_df.bg, bins=150)
    bg_axes.set_title('Average background around candle')
    bg_fig.show()

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

    return params


def save_candle_intensity(good_candles_df, filename):
    output = np.array((good_candles_df.sum_intensity, good_candles_df.bg))
    # save the output array to a text file
    np.savetxt(filename[:-4] + '_data.txt',
               output.T, fmt='%8.4f', delimiter='\t')
    return


def fit_allcandles_distribution(all_data_60, all_data_120):
    # make a histogram of the intensities
    counts60, bins60 = np.histogram(
        all_data_60[:, 0] - all_data_60[:, 1], bins=150)
    counts120, bins120 = np.histogram(
        all_data_120[:, 0] - all_data_120[:, 1], bins=150)

    # This gets the center of the bin instead of the edges
    bins60 = bins60[:-1] + np.diff(bins60/2)
    bins120 = bins120[:-1] + np.diff(bins120/2)

    # Make initial guesses as to fitting parameters
    hist_max60 = np.argwhere(counts60 == np.max(counts60))
    hist_max120 = np.argwhere(counts120 == np.max(counts120))
    p0_60 = [np.max(counts60), bins60[hist_max60[0, 0]],
             np.std(all_data_60[:, 0] - all_data_60[:, 1])]
    p0_120 = [np.max(counts120), bins120[hist_max120[0, 0]],
              np.std(all_data_120[:, 0] - all_data_120[:, 1])]

    # Fit the curve
    params60, params60_covariance = optimize.curve_fit(
        gaussian_fit, bins60, counts60, p0_60)
    params120, params120_covariance = optimize.curve_fit(
        gaussian_fit, bins120, counts120, p0_120)

    # Create a fit line using the parameters from your fit and the original bins
    bg_fit60 = gaussian_fit(bins60, params60[0], params60[1], params60[2])
    bg_fit120 = gaussian_fit(bins120, params120[0], params120[1], params120[2])

    # Plot result
    all_distribution_fit_fig, all_distribution_fit_axes = plt.subplots()
    all_distribution_fit_axes.hist(all_data_60[:, 0] - all_data_60[:, 1], bins=150, color="navy")
    all_distribution_fit_axes.plot(bins60, bg_fit60, '-k')
    all_distribution_fit_axes.hist(all_data_120[:, 0] - all_data_120[:, 1], bins=150, color="purple")
    all_distribution_fit_axes.plot(bins120, bg_fit120, '-k')
    all_distribution_fit_axes.set_xlabel('Summed Intensity Candle - Background Subtracted')
    all_distribution_fit_axes.set_ylabel('Counts')
    all_distribution_fit_axes.set_title('Combined Data')
    # all_distribution_fit_axes.text(0.95, 0.95, 'Average Value \nof a GFP fluorophore:\n' + str(np.round(params[1]/candle_size)),
    # verticalalignment='top', horizontalalignment='right',
    # transform=all_distribution_fit_axes.transAxes, fontsize=10)
    all_distribution_fit_fig.show()
    # save the figure
    all_distribution_fit_fig.savefig('all_data_histograms.eps', dpi=150)

    x = np.array([0, 60, 120])
    y = np.array([0, params60[1], params120[1]])
    yerr = np.array([0, params60[2], params120[2]])

    # Generated linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope*x+intercept

    candle_calibration_fig, candle_calibration_axes = plt.subplots()
    candle_calibration_axes.errorbar(x, y, yerr=yerr, max_color='purple', fmt='s')
    candle_calibration_axes.plot(x, line, 'k')
    candle_calibration_axes.set_title(
        'Candle Calibration\n' + 'y = ' + str(np.round(slope)) + 'X + ' + str(np.round(intercept)))
    candle_calibration_axes.text(0.05, 0.95, 'R-squared:' + str(r_value**2),
                                 verticalalignment='top', horizontalalignment='left',
                                 transform=candle_calibration_axes.transAxes, fontsize=10)
    candle_calibration_axes.set_xlabel('# of GFP')
    candle_calibration_axes.set_ylabel('Intensity')
    candle_calibration_fig.show()

    # save the figure
    candle_calibration_fig.savefig('all_data_calibration.eps', dpi=150)

    return params60, params120


def filament_finder(good_maxima_df, micron_per_pixel=0.043, min_dist=0.3, max_dist=0.5):
    
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


def good_filament_finder(good_maxima_df, imstack, gfp=1731, pad = 5):
 
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

    # Plot results
    max_projection = np.amax(imstack, axis=0)
    cluster_fig, cluster_axes = plt.subplots()
    cluster_axes.imshow(max_projection, cmap='Greys', vmin=50, vmax=800)

    for i, filament in good_filament_df.iterrows():
        cluster_axes.plot(filament['maxcols'],
                          filament['maxrows'],
                          color='xkcd:lightish blue', marker='o')
        cluster_axes.plot(filament['bbox'][:, 0],
                          filament['bbox'][:, 1], 
                          color='xkcd:bright purple')
    cluster_fig.show()

    return good_filament_df

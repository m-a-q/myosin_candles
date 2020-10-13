import numpy as np                 # This contains all our math functions we'll need
import skimage.io as io            # This toolbox is what we'll use for reading and writing images 
%matplotlib notebook 
import matplotlib.pyplot as plt    # This toolbox is to create our plots. Line above makes them interactive
import os                          # This toolbox is a useful directory tool to see what files we have in our folder
import cv2                         # image processing toolbox
import glob as glob                # grabbing file names
from mpl_toolkits import mplot3d   # for making a 3D surface plot
import czifile                     # read in the czifile
from scipy import stats
from scipy import optimize, ndimage               # for curve fitting
from skimage.measure import label  # For labeling regions in thresholded images
from skimage.measure import regionprops  # For calculating properties of labeled regions
from skimage.segmentation import clear_border  # Removes junk from the borders of thresholded images
from skimage.color import label2rgb  # Pretty display labeled images
from skimage.morphology import remove_small_objects, remove_small_holes  # Clean up small objects or holes in our image
import pandas as pd  # For creating our dataframe which we will use for filtering

# Functions for candle analysis
def prep_file(filename):
    # read in the file
    imstack = czifile.imread(filename)
    
    # keep only those dimensions that have the z-stack
    imstack = imstack[0,0,0,0,:,:,:,0]
    
    # make max projection and get the size of the image
    max_projection = np.amax(imstack, axis=0)
    nrows, ncols = max_projection.shape
    max_projection_fig, max_projection_axes = plt.subplots()
    max_projection_axes.imshow(max_projection, cmap='Greys', vmin=50, vmax=800)
    max_projection_fig.show()
    
    return imstack, max_projection, nrows, ncols

def find_candles(max_projection, nrows, ncols):
    
    # Create a mask to hold local peaks
    mask = np.zeros_like(max_projection)
    for i in np.arange(1,nrows-1):
        for j in np.arange(1,ncols-1):
            if max_projection[i,j] > 1000:
                if max_projection[i,j] > max_projection[i-1, j-1]:
                    if max_projection[i,j] > max_projection[i-1,j]:
                        if max_projection[i,j] > max_projection[i-1,j+1]:
                            if max_projection[i,j] > max_projection[i, j-1]:
                                if max_projection[i,j] > max_projection[i,j+1]:
                                    if max_projection[i,j] > max_projection[i+1,j-1]:
                                        if max_projection[i,j] > max_projection[i+1, j]:
                                            if max_projection[i,j] > max_projection[i+1, j+1]:
                                                mask[i,j] = 1
    # Find the number of total peaks
    Nobjects = np.sum(mask.ravel())
    print('Found %d Objects' % (Nobjects))

    # label individual peaks
    label_mask = label(mask)
    # get properties of each peak
    candle_props = regionprops(label_mask, max_projection)

    # create empty holders for properties of interest
    candle_ids = []
    candle_centroid_rows = []
    candle_centroid_cols = []
    candle_intensity = []

    # extract properties from each peak
    for candle in candle_props:
        candle_ids.append(candle.label)
        row, col = candle.centroid
        candle_centroid_rows.append(row.astype('int'))
        candle_centroid_cols.append(col.astype('int'))
        candle_intensity.append(candle.max_intensity)

    # Create a dictionary with data
    data_dict = {'labels': candle_ids,
            'crows': candle_centroid_rows,
            'ccols': candle_centroid_cols,
            'intensity': candle_intensity}

    # create a dataframe from the dictionary
    df = pd.DataFrame(data_dict)


    overlay_fig, overlay_axes = plt.subplots()
    overlay_axes.imshow(max_projection, cmap='Greys', vmin=50, vmax=800)
    overlay_axes.plot(df.ccols,df.crows,'xr')
    overlay_fig.show()
    
    
    return df, Nobjects

def identify_good_candles(imstack, df, Nobjects, intensity_minimum, volume_width=5, min_val=2500):
    
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

    # Loop through each candle
    for candle in np.arange(0,Nobjects):
        # Check that the candle isn't on the edge of the image
        if df.crows[candle] > vwl and df.crows[candle] < (nrows - vwr) and df.ccols[candle] > vwl and df.ccols[candle] < (ncols - vwr):
            # select volume around the candle
            substack = imstack[:,df.crows[candle]-vw:df.crows[candle]+vwr,df.ccols[candle]-vw:df.ccols[candle]+vwr]
            candle_volume.append(substack)
            # Sum the intensity in that volume
            candle_sum_intensity.append(np.sum(substack))
            # if intensity is above a threshold value mark it as good
            if np.sum(substack) > intensity_minimum:
                good_intensity.append(1)
            else:
                good_intensity.append(0)
            # If the edges of the candle volume are below a threshold intensity, mark it as good
            if np.sum(substack[0,:,:]) < min_val and np.sum(substack[-1,:,:]) < min_val \
                and np.sum(substack[:,0,:]) < min_val and np.sum(substack[:,-1,:]) < min_val \
                and np.sum(substack[:,:,0]) < min_val and np.sum(substack[:,:,-1]) < min_val:
                    candle_good_border.append(1)
            else:
                candle_good_border.append(0)
            
            # pull the intensity values from the pixels around edge of the candle volume
            edge_pts = np.zeros(2*substack.shape[1]*substack.shape[2] \
                                + 2*(substack.shape[0]-2)*(substack.shape[1]) \
                                + 2*(substack.shape[0]-2)*(substack.shape[2]-2))
            edge_pts[0:(2*vw + 1)**2] = substack[0,:,:].ravel()
            edge_pts[(2*vw + 1)**2:((2*vw + 1)**2)*2] = substack[-1,:,:].ravel()
            edge_pts[((2*vw + 1)**2)*2:(((2*vw + 1)**2)*2 + (z-2)*(2*vw + 1))] = substack[1:-1,0,:].ravel()
            edge_pts[(((2*vw + 1)**2)*2 + (z-2)*(2*vw + 1)):(((2*vw + 1)**2)*2 + 2*(z-2)*(2*vw + 1))] \
                = substack[1:-1,-1,:].ravel()
            edge_pts[(((2*vw + 1)**2)*2 + 2*(z-2)*(2*vw + 1)):(((2*vw + 1)**2)*2 + 2*(z-2)*(2*vw + 1) + (z-2)*(2*vw - 1))] \
                = substack[1:-1,1:-1,0].ravel()
            edge_pts[(((2*vw + 1)**2)*2 + 2*(z-2)*(2*vw + 1) + (z-2)*(2*vw - 1)): \
                    (((2*vw + 1)**2)*2 + 2*(z-2)*(2*vw + 1) + 2*(z-2)*(2*vw - 1))] \
                = substack[1:-1,1:-1,-1].ravel()
            
            bg.append(np.mean(edge_pts) * substack.shape[0] * substack.shape[1] * substack.shape[2])
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
    
    # plot where the good candles are in the image
    good_overlay_fig, good_overlay_axes = plt.subplots()
    good_overlay_axes.imshow(max_projection, cmap='Greys', vmin=50, vmax=800)
    good_overlay_axes.plot(good_candles_df.ccols, good_candles_df.crows,'x', color ='xkcd:bright purple')
    good_overlay_fig.show()

    # plot histogram of intensities
    hist_inten_fig, hist_inten_axes = plt.subplots()
    hist_inten_axes.hist(good_candles_df.sum_intensity-good_candles_df.bg, bins = 150)
    hist_inten_axes.set_xlim(0,1200000)
    hist_inten_axes.set_xlabel('Candle Intensity')
    hist_inten_axes.set_ylim(0,40)
    hist_inten_axes.set_ylabel('Counts')
    hist_inten_axes.set_title(filename)
    hist_inten_fig.show()

    # plot histograms of bg values
    bg_fig, bg_axes = plt.subplots()
    bg_axes.hist(good_candles_df.bg, bins = 150)
    bg_axes.set_title('Average background around candle')
    bg_fig.show()
    
    return good_candles_df

# Fit a double gaussian curve
def doublegaussian_fit(x, A, B, C, D, E, F):
    return A * np.exp(-1/2 * ((x - B) / C) ** 2) + D * np.exp(-1/2 * ((x - E) / F) ** 2)

# Fit a gaussian curve
def gaussian_fit(x, A, B, C):
    return A * np.exp(-1/2 * ((x - B) / C) ** 2)

def fit_candle_distribution(good_candles_df, candle_size):
    # make a histogram of the intensities
    counts, bins = np.histogram(good_candles_df.sum_intensity - good_candles_df.bg, bins=150)

    # This gets the center of the bin instead of the edges
    bins = bins[:-1] + np.diff(bins/2)

    # Make initial guesses as to fitting parameters
    hist_max = np.argwhere(counts == np.max(counts))
    p0 = [np.max(counts), bins[hist_max[0,0]], np.std(good_candles_df.sum_intensity - good_candles_df.bg)]

    # Fit the curve
    params, params_covariance = optimize.curve_fit(gaussian_fit, bins, counts, p0)

    # Create a fit line using the parameters from your fit and the original bins
    bg_fit = gaussian_fit(bins, params[0], params[1], params[2])
    print('Fit Parameters : ',params)
    print('Average Value of a GFP fluorophore : ', params[1]/candle_size)

    # Plot result
    distribution_fit_fig, distribution_fit_axes = plt.subplots()
    distribution_fit_axes.plot(bins, counts,'-ok')
    distribution_fit_axes.plot(bins,bg_fit,'-r')
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

def save_candle_intensity(good_candles_df):
    output = np.array((good_candles_df.sum_intensity,good_candles_df.bg))
    # save the output array to a text file
    np.savetxt( filename[:-4] + '_data.txt', output.T, fmt='%8.4f', delimiter='\t')
    return

def fit_allcandles_distribution(all_data_60, all_data_120):
    # make a histogram of the intensities
    counts60, bins60 = np.histogram(all_data_60[:,0] - all_data_60[:,1], bins=150)
    counts120, bins120 = np.histogram(all_data_120[:,0] - all_data_120[:,1], bins=150)

    # This gets the center of the bin instead of the edges
    bins60 = bins60[:-1] + np.diff(bins60/2)
    bins120 = bins120[:-1] + np.diff(bins120/2)
    
    # Make initial guesses as to fitting parameters
    hist_max60 = np.argwhere(counts60 == np.max(counts60))
    hist_max120 = np.argwhere(counts120 == np.max(counts120))
    p0_60 = [np.max(counts60), bins60[hist_max60[0,0]], np.std(all_data_60[:,0] - all_data_60[:,1])]
    p0_120 = [np.max(counts120), bins120[hist_max120[0,0]], np.std(all_data_120[:,0] - all_data_120[:,1])]

    # Fit the curve
    params60, params60_covariance = optimize.curve_fit(gaussian_fit, bins60, counts60, p0_60)
    params120, params120_covariance = optimize.curve_fit(gaussian_fit, bins120, counts120, p0_120)
    
    # Create a fit line using the parameters from your fit and the original bins
    bg_fit60 = gaussian_fit(bins60, params60[0], params60[1], params60[2])
    bg_fit120 = gaussian_fit(bins120, params120[0], params120[1], params120[2]) 
    
    # Plot result
    all_distribution_fit_fig, all_distribution_fit_axes = plt.subplots()
    all_distribution_fit_axes.hist(all_data_60[:,0] - all_data_60[:,1], bins=150, color="navy")
    all_distribution_fit_axes.plot(bins60,bg_fit60,'-k')
    all_distribution_fit_axes.hist(all_data_120[:,0] - all_data_120[:,1], bins=150, color="purple")
    all_distribution_fit_axes.plot(bins120,bg_fit120,'-k')
    all_distribution_fit_axes.set_xlabel('Summed Intensity Candle - Background Subtracted')
    all_distribution_fit_axes.set_ylabel('Counts')
    all_distribution_fit_axes.set_title('Combined Data')
    #all_distribution_fit_axes.text(0.95, 0.95, 'Average Value \nof a GFP fluorophore:\n' + str(np.round(params[1]/candle_size)),
        #verticalalignment='top', horizontalalignment='right',
        #transform=all_distribution_fit_axes.transAxes, fontsize=10)
    all_distribution_fit_fig.show()
    # save the figure
    all_distribution_fit_fig.savefig('all_data_histograms.eps', dpi=150)
    
    x = np.array([0, 60, 120])
    y = np.array([0, params60[1], params120[1]])
    yerr = np.array([0, params60[2], params120[2]])
    
    # Generated linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    line = slope*x+intercept
    
    candle_calibration_fig, candle_calibration_axes = plt.subplots()
    candle_calibration_axes.errorbar(x,y,yerr=yerr, ecolor='purple', fmt='s')
    candle_calibration_axes.plot(x,line,'k')
    candle_calibration_axes.set_title('Candle Calibration\n' + 'y = ' + str(np.round(slope)) + 'X + ' + str(np.round(intercept)))
    candle_calibration_axes.text(0.05, 0.95, 'R-squared:' + str(r_value**2),
        verticalalignment='top', horizontalalignment='left',
        transform=candle_calibration_axes.transAxes, fontsize=10)
    candle_calibration_axes.set_xlabel('# of GFP')
    candle_calibration_axes.set_ylabel('Intensity')
    candle_calibration_fig.show()
    
    # save the figure
    candle_calibration_fig.savefig('all_data_calibration.eps', dpi=150)
   
    return params60, params120
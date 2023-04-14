import numpy as np                 # This contains all our math functions we'll need
# This toolbox is what we'll use for reading and writing images
import skimage.io as io
import matplotlib as mpl
import pims #python image sequence - allows for easy opening and processing of movies/movie frames
import trackpy as tp #tracks features in a movie to identify particle trajectories
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
import os               # This toolbox is a useful directory tool to see what files we have in our folder
import cv2                         # image processing toolbox
import glob as glob                # grabbing file names
import czifile                     # read in the czifile
from skimage import morphology, util, filters
from celluloid import Camera  #animates figures for movies with overlays plotted on them
from scipy import stats, optimize, ndimage               # for curve fitting  
from scipy.signal import medfilt, convolve   # Used to detect overlap
from scipy.spatial.distance import pdist, squareform
from skimage.measure import label, regionprops      # For labeling regions in thresholded images and calculating properties of labeled regions
from skimage.segmentation import clear_border       # Removes junk from the borders of thresholded images
from skimage.color import label2rgb  # Pretty display labeled images
from skimage.morphology import remove_small_objects, remove_small_holes     # Clean up small objects or holes in our image
from skimage.feature import peak_local_max      #finds local peaks based on a local or absolute threshold
import pandas as pd  # For creating our dataframe which we will use for filtering
from pandas import DataFrame, Series #for convenience
from candle_functions import *
from image_plotting_tools import *

#creates the truncated, inverted Kindlmann heatmap
def create_qbk_cmap():
    qbk_list = np.genfromtxt('C:/Users/mquintanilla1/Documents/Python_Code/Molecular_Counting/quarter_black_kindlmann-no-white-table-float-0256.csv', delimiter=',')
    qbk_cmap = ListedColormap(qbk_list)
    return qbk_cmap

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

def prep_file_zproject(czifilename):
    '''
    Reading in the file, reducing the dimensions to just XYZ, creating max projections and reading the metadata. 
    '''
    # read in the file
    imstack = czifile.imread(czifilename)

#     # keep only those dimensions that have the z-stack
    imstack = imstack[0, 0, 0, :, :, :, :, 0]
    #third dimension is channel
    #fourth dimension is time
    #fifth dimension is z
    #sixth and seventh are x and y

    # make max projection 
    imstack_max = np.max(imstack, axis=1)
    #make sum projection
    imstack_sum = np.sum(imstack, axis=1)

    # get the metadata
    exp_details = get_czifile_metadata(czifilename)


    return imstack, imstack_max, imstack_sum, exp_details


def save_z_projections(czifilename, imstack_max, imstack_sum):
    #save the z projections if they aren't already saved
    io.imsave(czifilename[:-4] + '_max.tif', imstack_max)
    io.imsave(czifilename[:-4] + '_sum.tif', imstack_sum)
    return


'''Creates a function to process each frame in the movie (in this case thresholding each frame)\
Then, using the pims pipeline we read in the frames and preprocess them into a binary image'''
@pims.pipeline
#this 'decorates' the function and allows a lazy evaluation of full movies

def preprocess(img, threshold_val = 1991):
    """
    Apply image processing functions to return a binary image
    """
    # Crop the pictures as for raw images.
#     img = img[80]
    # Apply thresholds, specify what image (img,  ) and the threshold value (  ,1991)\
    '''NEED TO: explore whether 1991 value is uniform or if there is a way to automate, change to variable to make easier to change'''
    #The threshold value needs to be odd and the goal here was to increase until myosin clusters were grouped together
    adaptive_thresh = filters.threshold_local(img,threshold_val)
    idx = img > adaptive_thresh
    idx2 = img < adaptive_thresh
    img[idx] = 0
    img[idx2] = 255
    img = ndimage.binary_dilation(img)
    img = ndimage.binary_dilation(img)
    return util.img_as_int(img)


'''Proccesses each frame of the time series with the parameters selected in the above cell'''
def find_features(frames,qbk_cmap):
    features = pd.DataFrame()
    img_example = frames[100]
    for num, img in enumerate(frames):
        white = 255
        label_image = label(img, background=white)
        for region in regionprops(label_image, intensity_image=img):
            # Everywhere, skip small and large areas
            if region.area < 20 or region.area > 1000:
                continue
            # Only black areas
            if region.mean_intensity > 1:
                continue

            # Store features which survived to the criterions
            features = features.append([{'y': region.centroid[0],
                                         'x': region.centroid[1],
                                         'frame': num,
                                         },])

    '''Plots the identified features on the example image'''
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))
    ax.imshow(img_example, cmap = qbk_cmap, vmin=50, vmax = np.max(img_example)*.4)

    tp.annotate(features[features.frame==(100)], img_example);
    
    return features, img_example


'''Links features across frames to identify tracked particles and then plots the trajectories on example image'''
def track_particles(features, img_example,qbk_cmap, tp_search_range = 5, tp_min_track_length = 10, tp_max_gap = 3):

    t = tp.link_df(features, tp_search_range, memory=tp_max_gap)
    t1 = tp.filter_stubs(t, tp_min_track_length)#make track length another variable
    # Compare the number of particles in the unfiltered and filtered data.
    print('Before:', t['particle'].nunique())
    print('After:', t1['particle'].nunique())
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))
    ax.imshow(img_example, cmap = qbk_cmap, vmin=50, vmax = np.max(img_example)*.4)
    tp.plot_traj(t1, superimpose=img_example)


    '''Sort the dataframe of trackpy output by particle id and then frame and make the x and y coordinates integers'''
    t1 = t1.reset_index(drop=True)  #This reset the index and drops the original index since this is named 'frame' and conflates with the 'frame' column

    t1_sort = t1.sort_values(by=['particle','frame'])#sort by particle id and then by frame 
    t1_sort = t1_sort.reset_index() #reset index now that things are sorted

    # change x and y coords to integers so that they are easier to call in the other functions
    t1_sort = t1_sort.astype({'x': 'int32'})
    t1_sort = t1_sort.astype({'y': 'int32'})
    
    return t1_sort


'''This is making a dataframe with a list of the unique particle ids to loop through and identify peaks etc on individual particle tracks. 
I also needed to add some filters before I process individual particles. I want to make sure that the particles (tracked myosin clusters) 
are appearing after ten frames because I am using the previous ten frames before appearance to calculate background intensity. 
I also want to make sure that the tracked particle never touches the edges of the imaging area 
(checking the first and last frame of appearance is sufficient for this).'''

def find_good_particles(t1_sort, imstack_max, bbox_pad=17):

    nrows = imstack_max.shape[2]
    ncols = imstack_max.shape[1]
    vw = bbox_pad
    vwl = vw - 1
    vwr = vw + 1

    #get list of unique particle IDs
    particles = t1_sort[t1_sort['particle'] != 0.].groupby('particle')
    particles_df = pd.DataFrame(np.sort(t1_sort['particle'].unique())[0:])
    # print(len(particles))
    particles_df.columns = ['cluster_id']

    #only want to look at particles that show up after 10 frames so that I can use the previous ten frames to calculate background
    good_particles_list = []
    for c in range(len(particles_df)):
        cluster_id = particles_df.cluster_id[c]
        frame_test = t1_sort.loc[t1_sort['particle'] == cluster_id].reset_index()
        if frame_test.frame[0]>10\
        and int(frame_test.y[0]) > vwl\
        and int(frame_test.y[0]) < (nrows - vwr)\
        and int(frame_test.x[0]) > vwl\
        and int(frame_test.x[0]) < (ncols - vwr)\
        and int(frame_test.y[frame_test.index[-1]]) > vwl\
        and int(frame_test.y[frame_test.index[-1]]) < (nrows - vwr)\
        and int(frame_test.x[frame_test.index[-1]]) > vwl\
        and int(frame_test.x[frame_test.index[-1]]) < (ncols - vwr):
            good_particles_list.append(cluster_id)

    print(len(good_particles_list))
    
    return good_particles_list


'''I picked out good potential myosin clusters in Fiji, then saved the coordinates and frame where I picked out the cluster as a csv file.
I want to use this info to select these specific particles for analysis'''

def find_picked_particles(manual_csv_filename, t1_sort, imstack_max, bbox_pad=17):

    picked_particles = pd.read_csv(manual_csv_filename) 
    picked_particles.rename(columns={'Slice': 'frame'}, inplace=True)

    '''I want to loop through the the x,y,frame values of the picked_particles df and select the corresponding 
    'particle' values in the t1_sort df to get a list of particle ids. '''

    nrows = imstack_max.shape[2]
    ncols = imstack_max.shape[1]
    vw = bbox_pad
    vwl = vw - 1
    vwr = vw + 1
    picked_particles_list =[]


    for i in range(len(picked_particles)):
        picked_frame = int(picked_particles.frame[i])
        x = int(picked_particles.X[i])
        y = int(picked_particles.Y[i])
        cluster_id = t1_sort.loc[(t1_sort['frame'] > picked_frame-1) & (t1_sort['frame'] < picked_frame+1)\
                                & (t1_sort['x'] > x-bbox_pad) & (t1_sort['x'] < x+bbox_pad)\
                                & (t1_sort['y'] > y-bbox_pad) & (t1_sort['y'] < y+bbox_pad)]
        if len(cluster_id) >0:
            picked_particles_list.append(cluster_id['particle'].values[0])

    print(len(picked_particles_list))
    
    return picked_particles_list

'''Loops through every particle in the picked particle list and gets the sum bg value
(all of the pixels in the bounding box that will measure myosin intensity) in the ten frames before the myosin particle appears. 
Then it takes all of those frame+particle specific bg measurementsand determines the distributions of those measurements to 
then take about the 95% cofidence interval as the bg_threshold'''
def find_myo_bg_threshold(picked_particles_list,
                          t1_sort,
                          imstack_max,
                          imstack_sum,
                          bbox_pad = 17,
                          show_results=True):
    all_particle_bg = []
    
    for cid in range(len(picked_particles_list)):
        cluster_id = picked_particles_list[cid] 
        t1_sort_particle = t1_sort.loc[t1_sort['particle'] == cluster_id]
        t1_sort_particle = t1_sort_particle.reset_index()

        particle_bg = []
        for bg in np.arange(0,10):
            bg_frame = t1_sort_particle.frame[0] - bg
            ymin = int(t1_sort_particle.y[0]) - bbox_pad
            ymax = int(t1_sort_particle.y[0]) + bbox_pad
            xmin = int(t1_sort_particle.x[0]) - bbox_pad
            xmax = int(t1_sort_particle.x[0]) + bbox_pad
            sub_max = imstack_max[bg_frame, ymin:ymax, xmin:xmax]
            sub_sum = imstack_sum[bg_frame, ymin:ymax, xmin:xmax]
            particle_bg.append(np.sum(sub_sum))    

        all_particle_bg.extend(particle_bg)

    countsbg, binsbg = np.histogram(
        all_particle_bg, bins=150)    

    # This gets the center of the bin instead of the edges
    binsbg = binsbg[:-1] + np.diff(binsbg/2)

    # Make initial guesses as to fitting parameters
    hist_maxbg = np.argwhere(countsbg == np.max(countsbg))

    p0_bg = [np.max(countsbg), binsbg[hist_maxbg[0, 0]],
             np.std(all_particle_bg)]

    # Fit the curve
    try:
        paramsbg, paramsbg_covariance = optimize.curve_fit(
        gaussian_fit, binsbg, countsbg, p0_bg)
    except RuntimeError:
        paramsbg = p0_bg

    # Create a fit line using the parameters from your fit and the original bins
    bg_fitbg = gaussian_fit(binsbg, paramsbg[0], paramsbg[1], paramsbg[2])

    if show_results:
        bg_fit_fig, bg_fit_axes = plt.subplots(figsize=(6,4))
        bg_fit_axes.hist(all_particle_bg, bins=150, color='#270563')
        bg_fit_axes.plot(binsbg, bg_fitbg, '-k')
        
    myo_bg_threshold = paramsbg[1] + 2*paramsbg[2]
    
    return all_particle_bg, myo_bg_threshold



'''This loops back through the picked particles list and kicked out any bg frames that are too high and if it kicks out more than 4
of the ten bg frames for the particle, the particle is kicked out of the particle list. This is so we are truly measuring bg and not 
myosin in previous frames and also that we have enough bg frames to calculate the average'''
def clean_bg_frames(picked_particles_list, t1_sort, imstack_max, imstack_sum, myo_bg_threshold, bbox_pad = 17,show_results=True):
    clean_bg_picked_particles = []
    
    for cid in range(len(picked_particles_list)):
        cluster_id = picked_particles_list[cid] 
        t1_sort_particle = t1_sort.loc[t1_sort['particle'] == cluster_id]
        t1_sort_particle = t1_sort_particle.reset_index()

        particle_bg = []
        for bg in np.arange(0,10):
            bg_frame = t1_sort_particle.frame[0] - bg
            ymin = int(t1_sort_particle.y[0]) - bbox_pad
            ymax = int(t1_sort_particle.y[0]) + bbox_pad
            xmin = int(t1_sort_particle.x[0]) - bbox_pad
            xmax = int(t1_sort_particle.x[0]) + bbox_pad
            sub_max = imstack_max[bg_frame, ymin:ymax, xmin:xmax]
            sub_sum = imstack_sum[bg_frame, ymin:ymax, xmin:xmax]
            if np.sum(sub_sum) < myo_bg_threshold:
                particle_bg.append(np.sum(sub_sum))
        if len(particle_bg)>5:
            clean_bg_picked_particles.append(cluster_id)
    return clean_bg_picked_particles 

'''Creates function to process each individual particle that was identified by either the good_particle_list or picked_particle_list
In the function a 10-frame background is calculated, sum intensity is calculate, peaks are identified and then tracked with trackpy.
When the peaks are tracked, coordinates are saved at each step before filtering so that we can look back and verify that the filters are 
not throwing out real myosin peaks (qualitatively). The returned dataframe has all of the info from the selected particle that could 
be needed for data visualization and qualitative checks'''

def particle_process(czifilename, t1_sort, cluster_id, imstack_max, imstack_sum, myo_bg_threshold, bbox_pad = 17, xy_max = 90000, find_peak_dist = 1, find_peak_thresh = 500, peak_tp_search_range = 2, peak_tp_min_track_length = 3, peak_tp_max_gap = 2, micron_per_pixel=0.043, min_dist=0.1, max_dist=0.4, GFP_value = 2580, k = 5):
        
    t1_sort_particle = t1_sort.loc[t1_sort['particle'] == cluster_id]
    t1_sort_particle = t1_sort_particle.reset_index()
    
    particle_bg = []
    
    for bg in np.arange(0,10):
        bg_frame = t1_sort_particle.frame[0] - bg
        ymin = int(t1_sort_particle.y[0]) - bbox_pad
        ymax = int(t1_sort_particle.y[0]) + bbox_pad
        xmin = int(t1_sort_particle.x[0]) - bbox_pad
        xmax = int(t1_sort_particle.x[0]) + bbox_pad
        sub_max = imstack_max[bg_frame, ymin:ymax, xmin:xmax]
        sub_sum = imstack_sum[bg_frame, ymin:ymax, xmin:xmax]
        if np.sum(sub_sum) < myo_bg_threshold:
            particle_bg.append(np.sum(sub_sum))
    
    #if any value is bigger than bg threshold value then delete from the list before calculating mean (remove elements from the list)
    particle_bg_mean = np.mean(particle_bg)

    #need to make a mask of zeros that is the same shape as the im_stack
    im_mask = np.zeros_like(imstack_max)
    im_mask_blur = np.zeros_like(imstack_max)
    gaussian_kernel = cv2.getGaussianKernel(5, sigma=1) # 
    # Multiply the vector by it's transpose to make a matrix
    gaussian_kernel = gaussian_kernel * gaussian_kernel.T

    x = []
    y = []
    frame = []
    sum_intensity = []
    background = []
    bg_frame_count = []
    bg_frames = []
    good_border = []

    for p in range(len(t1_sort_particle)):
        particle_frame = int(t1_sort_particle.frame[p])
        ymin = int(t1_sort_particle.y[p]) - bbox_pad
        ymax = int(t1_sort_particle.y[p]) + bbox_pad + 1
        xmin = int(t1_sort_particle.x[p]) - bbox_pad
        xmax = int(t1_sort_particle.x[p]) + bbox_pad + 1
        sub_max = imstack_max[particle_frame, ymin:ymax, xmin:xmax]
        sub_sum = imstack_sum[particle_frame, ymin:ymax, xmin:xmax]
        sum_intensity.append(np.sum(sub_sum))
        background.append(particle_bg_mean)
        bg_frame_count.append(len(particle_bg))

        if (np.sum(sub_sum[0, :]) + np.sum(sub_sum[-1, :]) + np.sum(sub_sum[:, 0]) + np.sum(sub_sum[:, -1])) < xy_max:
            good_border.append(1)
        else:
            good_border.append(0)
        sub_peak_coords = peak_local_max(sub_max, min_distance=find_peak_dist, threshold_abs=find_peak_thresh, num_peaks=8)
        #peak coords is a list of xy coords as tuples. first slice is for list item, second is for x or y coord
        whole_peak_coords =  sub_peak_coords.copy()
        for peak in np.arange(0, len(sub_peak_coords)):
            xp = sub_peak_coords[peak][1]
            if xp > bbox_pad:
                whole_peak_coords[peak][1] = t1_sort_particle.x[p] + (xp - bbox_pad)
            elif xp < bbox_pad:
                whole_peak_coords[peak][1] = t1_sort_particle.x[p] - (bbox_pad - xp)
            else:
                whole_peak_coords[peak][1] = t1_sort_particle.x[p]

            yp = sub_peak_coords[peak][0]
            if yp > bbox_pad:
                whole_peak_coords[peak][0] = t1_sort_particle.y[p] + (yp - bbox_pad)
            elif yp < bbox_pad:
                whole_peak_coords[peak][0] = t1_sort_particle.y[p] - (bbox_pad - yp)
            else:
                whole_peak_coords[peak][0] = t1_sort_particle.y[p]
            x.append(whole_peak_coords[peak][1])
            y.append(whole_peak_coords[peak][0])
            frame.append(particle_frame)

        im_mask[particle_frame,whole_peak_coords[:,0],whole_peak_coords[:,1]]=255
        im_mask_blur[particle_frame] = cv2.filter2D(im_mask[particle_frame], -1, gaussian_kernel)

    io.imsave(czifilename[:-4] + '_cluster' + str(cluster_id) + '_mask.tif', im_mask_blur.astype('uint8'))    
        
    peak_dict = {'x': x,
             'y': y,
             'frame': frame}
    peak_df = pd.DataFrame(peak_dict)
    peak_t = tp.link(peak_df, peak_tp_search_range, memory=peak_tp_max_gap)
    peak_t1 = tp.filter_stubs(peak_t, peak_tp_min_track_length)

        # assign neighbors
    prefilter_peak_coords = []
    stub_filter_peak_coords = []
    neighbors = []
    neighbor_coords = []
    for i in range(len(t1_sort_particle)):
        frm = t1_sort_particle.frame[i]
    #peak_t is the data frame of peaks after particle tracking but before 'stubs' or short tracks are filtered out
        prefilter_peaks_df = peak_t.loc[peak_t['frame'] == frm]
        prefilter_peaks_arr = np.array(prefilter_peaks_df[['x','y']])
        prefilter_peak_coords.append(prefilter_peaks_arr)
        stubfilter_peaks_df = peak_t1.loc[peak_t1['frame'] == frm]
        stubfilter_peaks_arr = np.array(stubfilter_peaks_df[['x','y']])
        stub_filter_peak_coords.append(stubfilter_peaks_arr)
        pairwise_distances = squareform(pdist(stubfilter_peaks_arr))
        pairwise_distances *= micron_per_pixel
        neighbor_condition = np.logical_and(min_dist < pairwise_distances, pairwise_distances < max_dist)
        neighbors.append(len(np.unique(np.where(neighbor_condition)[0])))
        good_dist = np.unique(np.where(neighbor_condition)[0])
        good_dist_coords = []
        for i in range(len(good_dist)):
            good_dist_coords.append(stubfilter_peaks_arr[good_dist[i]].tolist())
        neighbor_coords.append(np.array(good_dist_coords))

    t1_sort_particle['neighbors'] = neighbors
    t1_sort_particle['sum_intensity'] = sum_intensity
    t1_sort_particle['good_border'] = good_border
    t1_sort_particle['background'] = background
    t1_sort_particle['bg_frame_count'] = bg_frame_count
    t1_sort_particle['prefilter_peak_coords'] = prefilter_peak_coords
    t1_sort_particle['stub_filter_peak_coords'] = stub_filter_peak_coords  
    t1_sort_particle['bg_sub_intensity'] = t1_sort_particle.sum_intensity - t1_sort_particle.background
    t1_sort_particle['GFPs'] = t1_sort_particle.bg_sub_intensity/GFP_value
    t1_sort_particle['monomers'] = t1_sort_particle.GFPs/2
    neighbors_smooth = medfilt(neighbors, kernel_size=k)
    t1_sort_particle['neighbors_smooth'] = neighbors_smooth
    t1_sort_particle['neighbor_coords'] = neighbor_coords

    return t1_sort_particle, particle_bg 


'''this conditional function check is any item in a list meets a condition with a specified value.
In this case asking if any item is equal to that desired value (can change to <, >, !=)'''
def check(list1, val):
    return(any(x == val for x in list1))


'''Partition_process function checks for a partitioning frame (3 neighbors) and gets the frame, monomer count, and the monomer count of the previous and following frame.
You can change the frame that is picked out by adding a different number to the index value that you are pulling'''

def partition_process(t1_sort_particle):
    #if any value in list of neighbors_smooth is equal to 3. Can change in check function.
    if check(t1_sort_particle['neighbors_smooth'], 3):
        smooth_partition_index = t1_sort_particle.index[t1_sort_particle['neighbors_smooth'] == 3].tolist()
        smooth_partition_index = smooth_partition_index[0]
        find_partition_frame = t1_sort_particle[(smooth_partition_index-3):(smooth_partition_index+3)]
        partition_index = find_partition_frame.index[find_partition_frame['neighbors']==3].tolist()
        partition_index = partition_index[0]
        partition_frame = find_partition_frame['frame'][partition_index]
        if any(x<3 for x in t1_sort_particle.neighbors_smooth[:partition_index]):
            partition_monomer_count = t1_sort_particle.iloc[partition_index]['monomers']
            #index-3 is 3 rows back, can change to previous frame
            pre_part_monomer_count = t1_sort_particle.iloc[partition_index-2]['monomers']
            post_part_monomer_count = t1_sort_particle.iloc[partition_index+2]['monomers']
        else:
            partition_frame = 0
            partition_monomer_count = 0
            pre_part_monomer_count = 0
            post_part_monomer_count = 0
    else:
        partition_frame = 0
        partition_monomer_count = 0
        pre_part_monomer_count = 0
        post_part_monomer_count = 0
    
    return (partition_frame, partition_monomer_count, pre_part_monomer_count, post_part_monomer_count)


'''Check_particle function allows you to look at the particle in a movie with the different neighbors that are picked out and filtered at different steps.
Use this to verify that the neighbors that are selected, and used to define the frame of partitioning, are being faithfully identified.'''


def check_particle(t1_sort_particle, imstack_max, qbk_cmap,bbox_pad=17, window_pad = 50):

    t1_sort_particle_clean_neighbors = t1_sort_particle[t1_sort_particle['neighbors']>1]
    t1_sort_particle_clean_neighbors = t1_sort_particle_clean_neighbors.reset_index(drop=True)
    smooth_partition_index = t1_sort_particle_clean_neighbors.index[t1_sort_particle_clean_neighbors['neighbors_smooth'] == 3].tolist()
    smooth_partition_index = smooth_partition_index[0]
    find_partition_frame = t1_sort_particle_clean_neighbors[(smooth_partition_index-3):(smooth_partition_index+3)]
    partition_index = find_partition_frame.index[find_partition_frame['neighbors']==3].tolist()
    partition_index = partition_index[0]
    partition_frame = find_partition_frame['frame'][partition_index]
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(len(t1_sort_particle_clean_neighbors)):
        frm = t1_sort_particle_clean_neighbors['frame'].iloc[i]
        if frm == partition_frame:
            for f in np.arange(0,6):
                plt.imshow(imstack_max[frm], cmap = qbk_cmap, vmin=50, vmax = np.max(imstack_max)*.4)
                plt.plot(t1_sort_particle_clean_neighbors.prefilter_peak_coords[i][:, 0], t1_sort_particle_clean_neighbors.prefilter_peak_coords[i][:, 1], '.', color='xkcd:raspberry')
                plt.plot(t1_sort_particle_clean_neighbors.stub_filter_peak_coords[i][:, 0], t1_sort_particle_clean_neighbors.stub_filter_peak_coords[i][:, 1], '+', color='xkcd:tangerine')
                plt.plot(t1_sort_particle_clean_neighbors.neighbor_coords[i][:, 0], t1_sort_particle_clean_neighbors.neighbor_coords[i][:, 1], 'x:', color='xkcd:lemon yellow')
                #Get the current reference
                ax = plt.gca()
                # Create a Rectangle patch
                rect = patches.Rectangle((t1_sort_particle_clean_neighbors.x[i]-bbox_pad,t1_sort_particle_clean_neighbors.y[i]-bbox_pad),(2*bbox_pad+1),(2*bbox_pad+1),linewidth=3,edgecolor='xkcd:royal purple',facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)
                camera.snap()
        else:
            plt.axes(xlim=(t1_sort_particle_clean_neighbors['x'].iloc[0]-window_pad, t1_sort_particle_clean_neighbors['x'].iloc[-1]+window_pad), ylim=(t1_sort_particle_clean_neighbors['y'].iloc[0]-2*window_pad, t1_sort_particle_clean_neighbors['y'].iloc[-1]+2*window_pad))
            plt.imshow(imstack_max[frm], cmap = qbk_cmap, vmin=50, vmax = np.max(imstack_max)*.4)
            plt.plot(t1_sort_particle_clean_neighbors.prefilter_peak_coords[i][:, 0], t1_sort_particle_clean_neighbors.prefilter_peak_coords[i][:, 1], '.', color='xkcd:raspberry')
            plt.plot(t1_sort_particle_clean_neighbors.stub_filter_peak_coords[i][:, 0], t1_sort_particle_clean_neighbors.stub_filter_peak_coords[i][:, 1], '+', color='xkcd:tangerine')
            plt.plot(t1_sort_particle_clean_neighbors.neighbor_coords[i][:, 0], t1_sort_particle_clean_neighbors.neighbor_coords[i][:, 1], 'x:', color='xkcd:lemon yellow')
            camera.snap()
    return camera


def save_tracked_filament_data(all_particle_df, czifilename):
    all_particle_df.to_hdf(czifilename[:-4] + '_track_data.hd5', key='Tracked_Filaments', mode='w')
                           
    return


def save_partition_data(good_partition_df, czifilename):
    good_partition_df.to_hdf(czifilename[:-4] + '_partition_data.hd5', key='Tracked_Filaments', mode='w')
                           
    return


def save_tracked_filament_excel(all_particle_df, czifilename):
    # writing to Excel 
    datatoexcel = pd.ExcelWriter(czifilename[:-4] + '_track_data.xlsx') 
    # write DataFrame to excel 
    all_particle_df.to_excel(datatoexcel) 
    # save the excel 
    datatoexcel.save() 
    
    return


def plot_tracked_filament_data(all_particle_df, czifilename):
    '''Plot monomers over time for each particle that is processed in the movie.'''
    #plot clean_all_particle_df to look at all particles
    clean_all_particle_df = all_particle_df[all_particle_df['good_border']>0]
    clean_all_particle_df = clean_all_particle_df[clean_all_particle_df['neighbors_smooth']>1]

    #plot subset if the data is too dense to get a sense of things
    subset = clean_all_particle_df[clean_all_particle_df['particle']<4000]

    sns.set(font_scale=2)
    sns.set_style("white")
    fig, axes = plt.subplots(figsize=(8,6))
    axes = sns.scatterplot(data=clean_all_particle_df, x="frame", y="monomers", hue = "particle", style = "neighbors_smooth", palette="YlGnBu")
    axes.set(xlabel='Frame', ylabel='Monomers')
    axes.get_legend().remove()
    plt.tight_layout(w_pad=0)
    plt.axhline(y=30, color='k', linestyle='dashdot')

    fig.savefig(czifilename[:-4]+'_picked_particles_over_time.eps', dpi=150)
    fig.savefig(czifilename[:-4]+'_picked_particles_over_time.png', dpi=150)
    
    return


def plot_partition_data(good_partition_df, czifilename):
    '''Plot monomers at pre-partition, partition, and post-partition frames from the good_partition_df that is a result of the partition_process function'''
    
    good_partition_df.rename(columns={'pre_partitioning_count': 'Pre-Partition', 'partitioning_count': 'Partition', 'post_partitioning_count':'Post-Partition'}, inplace=True)

    fig, axes = plt.subplots(figsize=(8,6))
    axes = sns.barplot(data=good_partition_df[["Pre-Partition", "Partition", "Post-Partition"]],palette = 'YlGnBu')
    axes = sns.swarmplot(data=good_partition_df[["Pre-Partition", "Partition", "Post-Partition"]],palette = 'cmr.swamp')
    # plt.xticks(rotation=30, ha='right')
    plt.tight_layout(w_pad=0)

    fig.savefig(czifilename[:-4] + '_picked_particle_count_bar.png', dpi=150)
    fig.savefig(czifilename[:-4] + '_picked_particle_count_bar.eps', dpi=150)
    
    return


def save_all_tracked_particle_data(all_tracked_particle_data, experiment_date):
    all_tracked_particle_data.to_hdf(experiment_date+'_all_tracked_data.hd5', key='Filaments', mode='w')
                           
    return


def save_all_partition_data(all_partition_data, experiment_date):
    all_partition_data.to_hdf(experiment_date+'_all_partition_data.hd5', key='Filaments', mode='w')
                           
    return


def plot_all_partition_data(all_partition_data, experiment_date):
    '''Plot monomers at pre-partition, partition, and post-partition frames from the good_partition_df that is a result of the partition_process function'''
    # all_partition_data_crop = all_partition_data[all_partition_data['partitioning_count']<600]
    # all_partition_data_crop.rename(columns={'pre_partitioning_count': 'Pre-Partition', 'partitioning_count': 'Partition', 'post_partitioning_count':'Post-Partition'}, inplace=True)
    all_partition_data.rename(columns={'pre_partitioning_count': 'Pre-Partition', 'partitioning_count': 'Partition', 'post_partitioning_count':'Post-Partition'}, inplace=True)
    fig, axes = plt.subplots(figsize=(8,6))
    sns.set_style("white")
    axes = sns.barplot(data=all_partition_data[["Pre-Partition", "Partition", "Post-Partition"]],palette = 'YlGnBu')
    axes = sns.swarmplot(data=all_partition_data[["Pre-Partition", "Partition", "Post-Partition"]],palette = 'cmr.swamp')
    # plt.xticks(rotation=30, ha='right')
    plt.tight_layout(w_pad=0)

    fig.savefig(experiment_date + '_picked_particle_count_bar.png', dpi=150)
    fig.savefig(experiment_date + '_picked_particle_count_bar.eps', dpi=150)
    
    return


def plot_all_tracked_filament_data(all_tracked_particle_data, experiment_date,qbk_cmap):
    '''Plot monomers over time for each particle that is processed in the movie.'''
    #plot clean_all_particle_df to look at all particles
    clean_all_particle_df = all_tracked_particle_data[all_tracked_particle_data['good_border']>0]
    clean_all_particle_df = clean_all_particle_df[clean_all_particle_df['neighbors_smooth']>1]

    #plot subset if the data is too dense to get a sense of things
    subset = clean_all_particle_df[clean_all_particle_df['particle']<4000]

    sns.set(font_scale=1.5)
    sns.set_style("white")
    fig, axes = plt.subplots(figsize=(10,10))
    axes = sns.scatterplot(data=clean_all_particle_df, x="frame", y="monomers", hue = "particle", style = "neighbors_smooth", palette=qbk_cmap)
    axes.set(xlabel='Frame', ylabel='Monomers')
    axes.get_legend().remove()
    plt.tight_layout(w_pad=0)
    plt.axhline(y=30, color='k', linestyle='dashdot')

    fig.savefig(experiment_date + '_picked_particles_over_time.eps', dpi=150)
    fig.savefig(experiment_date + '_picked_particles_over_time.png', dpi=150)
    
    return


def plot_all_experiments(all_experiment_data):
    all_experiment_data = all_experiment_data.astype({"n_maxima":'category'})
#     sns.set(rc={'figure.figsize':(10,7)})
    sns.set_context("talk", font_scale=1)
    plt.figure(figsize=(10,7))
    ax = sns.swarmplot(x=all_experiment_data['n_maxima'], y=all_experiment_data['monomers'], hue=all_experiment_data['date'], dodge=True, palette = "cmr.swamp", edgecolor="gray", alpha = 0.5, size = 5)
    ax = sns.boxplot(x=all_experiment_data['n_maxima'], y=all_experiment_data['monomers'], hue=all_experiment_data['date'], dodge=True,palette = "YlGnBu", showfliers = False)
    ax.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
    plt.tight_layout()
    ax.set(xlabel='Number of Maxima', ylabel='Monomers')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    # save the figure
    fig = ax.get_figure()
    fig.savefig('all_experiment_data.png', dpi=150)
    fig.savefig('all_experiment_data.eps', dpi=150)
    
    return

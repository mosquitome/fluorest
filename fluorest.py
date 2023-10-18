'''
#---------------------------------------------------------------------------------
#                                   F L U O R E S T
#---------------------------------------------------------------------------------

#                   (FLUOrescent RESponse Timecourse analyser)
# automated analysis of fluorescent responses in live cell microscopy timecourses.

# Author        :   David A. Ellis <https://github.com/mosquitome/>
# Organisation  :   University College London

# Requirements  :   Python 3.10
#                   czifile 2019.7.2
#                   pandas 2.1.1
#                   numpy 1.26.0
#                   cv2 4.8.1
#                   matplotlib 3.8.0

# Notes         :   Currently only compatible with CZI files (including multifiles).
                    Tested on Windows (Mac to come) with a few example files only.
                    "ZEISS" and "Carl Zeiss" are registered trademarks of Carl 
                    Zeiss AG.

#---------------------------------------------------------------------------------
'''

import czifile as czi
import pandas as pd
import os
import pathlib as pth
import numpy as np
import cv2
import matplotlib.pyplot as mplp
import matplotlib.gridspec as mplg
import matplotlib.colors as mplc

def nWells(img):
    return img.shape[5] # <- 5th axis of array from czifile describes the number of wells

def nChannels(img):
    return img.shape[6] # <- 6th axis of array from czifile describes the number of channels

def nPixels(img):
    return img.shape[9], img.shape[10] # <- 9th and 10th axes of array from czifile contain the image for a given well/channel

def formatImage(img):
    x, y = nPixels(img)
    well = {}
    for i in range(nWells(img)):
        well[i] = {}
        for j in range(nChannels(img)):
            well[i][j] = img[0, 0, 0, 0, 0, i, j, 0, 0, 0:x, 0:y, 0]
    return well

def getUnique(pdunique):
    a = []
    for i in pdunique:
        [a.append(j.strip()) for j in i.split(',')]
    b = sorted(set(a))
    return b

def loadImages(directory=os.getcwd()):
    '''

    Parameters
    ----------
    directory : Path to directory containing CZI image files and a tab-delimited
                metadata file called metadata.txt.

    Returns
    -------
    dictionary containing a sub-dictionary for each well, each containing a sub
    -dictionary for each timepoint, each containing a 2-axis array (i.e. dataframe)
    for each channel.

    '''
    metadata = pd.read_table(pth.Path(directory,'metadata.txt'))
    img_files = [i for i in os.listdir(directory) if i.split('.')[-1]=='czi']
    data = {}
    for i in img_files:
        if i not in metadata['filename'].values:
            raise LookupError(i+' is not in metadata')
        img = czi.imread(pth.Path(directory,i))
        data[i] = formatImage(img)
    wells = getUnique(metadata['wells (ordered as they appear in the file)'].unique())
    timepoints = metadata['timepoint'].unique()
    data2 = {}
    for i in wells:
        data2[i] = {}
        for j in timepoints:
            temp = metadata.loc[(metadata['wells (ordered as they appear in the file)'].apply(lambda x: i in [y.strip() for y in x.split(',')])) & (metadata['timepoint']==j)]
            if len(temp['filename']) != 1:
                raise LookupError('problem in metadata with ' + temp['filename'].values[0] + ', well ' + i)
            img_file = temp['filename'].values[0]
            well_list = [k.strip() for k in temp['wells (ordered as they appear in the file)'].values[0].split(',')]
            idx = well_list.index(i)
            print('loadImages(): ',i, j, img_file)
            data2[i][j] = data[img_file][idx].copy()
    return data2

def downScale(img):
    info = np.iinfo(img.dtype) # <- Get the information of the incoming image type
    temp = img.astype(np.float64) / info.max # <- Scale the data between 0 and 1
    temp = 255 * temp # <- Scale by 255
    img2 = temp.astype(np.uint8)
    return img2

def registerImages(target, template, n_features=5000, p_matches=0.9, f_threshold=5, override=True):
    '''
   
    Parameters
    ----------
    target          : image to be aligned
    template        : image to align to.
    n_features      : maximum number of features to search for. The default is 5000.
    p_matches       : proportion of best matching features to use. The default is 0.9
    f_threshold     : fast threshold used by cv2 ORB detector. The default is 5.
    override        : if not enough matching features are found for homography calculation
                      (at least 4 are required), this option will output an unmodified image 
                      (i.e. the original unregistered version) if set to True. Useful for
                      downstream functions like getFluorescence().

    Returns
    -------
    homography          : homography matrix that can then be used to align images in another
                          channel.
    target_transformed  : target image transformed using the homography matrix.

    '''
    target8 = downScale(target) # <- convert to 8-bit as this is required by cv2
    template8 = downScale(template)
    height, width = target.shape
   
    orb_detector = cv2.ORB_create(nfeatures=n_features, fastThreshold=f_threshold) # <- Create ORB detector with n_features number of features
    kp1, d1 = orb_detector.detectAndCompute(target8, None) # <- Find keypoints and descriptors. The first arg is the image, second arg is the mask (which is not required)
    kp2, d2 = orb_detector.detectAndCompute(template8, None)
 
    if isinstance(d1, np.ndarray)==False or isinstance(d1, np.ndarray)==False:
        n_matches = 0
        if override==False:
            raise ValueError('Try adjusting thresholds')
        elif override==True:
            print('registerImages(): No features detected. Outputting unregistered image in this instance.')
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # <- Match features between the two images (Brute Force matcher with Hamming distance).
        matches = matcher.match(d1, d2) # <- Match the two sets of descriptors.
        #matches.sort(key = lambda x: x.distance) # <- Sort matches on the basis of their Hamming distance.
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:int(len(matches) * p_matches)] # <- Take the top p_matches proportion of matches forward.
        n_matches = len(matches)

    if n_matches < 4 and override==False:
        raise ValueError('Not enough matches found to compute homography.')
    elif n_matches < 4 and override==True:
        print('registerImages(): Not enough matches found to compute homography. Outputting unregistered image in this instance.')
        homography = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype='float64')
        target_transformed = target.copy()
    else:
        p1 = np.zeros((n_matches, 2)) # <- Define empty matrices of shape n_matches * 2
        p2 = np.zeros((n_matches, 2))
        for i in range(len(matches)):
            p1[i, :] = kp1[matches[i].queryIdx].pt # <- Fill with info on matching features
            p2[i, :] = kp2[matches[i].trainIdx].pt
        
        homography, _ = cv2.findHomography(p1, p2, cv2.RANSAC) # <- Find the homography matrix.
        target_transformed = cv2.warpPerspective(target, homography, (width, height)) # <- Use this matrix to transform the original target image WRT template image.
        
    return homography, target_transformed

def compareImages(img1, img2):
    '''
    Score how similar two images are.
    '''
    img1_ = img1.astype(np.float64) / np.iinfo(img1.dtype).max # <- Scale the data between 0 and 1
    img2_ = img2.astype(np.float64) / np.iinfo(img2.dtype).max
    img1_norm = img1_/np.sqrt(np.sum(img1_**2))
    img2_norm = img2_/np.sqrt(np.sum(img2_**2))
    return np.sum(img1_norm*img2_norm)

def improvedRegistration(target, template, n_features=5000, p_matches=0.9, f_threshold=5, override=True):
    '''
    Dynamically adjust fast threshold to provide improved image registration.
    
    Parameters
    ----------
    target          : image to be aligned
    template        : image to align to.
    n_features      : maximum number of features to search for. The default is 5000.
    p_matches       : proportion of best matching features to use. The default is 0.9
    f_threshold     : STARTING fast threshold used by cv2 ORB detector. The default is 5.
    override        : if not enough matching features are found for homography calculation
                      (at least 4 are required), this option will output an unmodified image 
                      (i.e. the original unregistered version) if set to True. Useful for
                      downstream functions like getFluorescence().

    Returns
    -------
    flag                : flags whether there was an issue with registration or not.
                          Useful for downstream functions like getFluorescence().
    homography          : homography matrix that can then be used to align images in another
                          channel.
    target_transformed  : target image transformed using the homography matrix.
    
    '''
    homography, target_transformed = registerImages(target, template, f_threshold=f_threshold, n_features=n_features, p_matches=p_matches)
    score = compareImages(target, target_transformed)
    delta = 0
    while delta >= 0:
        f_threshold -= 1
        if f_threshold<0:
            print('improvedRegistration(): unable to improve registration.')
            homography, target_transformed = registerImages(target, template, f_threshold=0, n_features=n_features, p_matches=p_matches)
            flag = 'FLAG: image registration failure.'
            return flag, homography, target_transformed
        _, target_transformed2 = registerImages(target, template, f_threshold=f_threshold, n_features=n_features, p_matches=p_matches)
        delta = compareImages(target, target_transformed2) - score
        score = compareImages(target, target_transformed2)
        if delta >= 0:
            print('improvedRegistration(): ', f_threshold)
            homography, target_transformed = registerImages(target, template, f_threshold=f_threshold, n_features=n_features, p_matches=p_matches)
    flag = np.nan
    return np.nan, homography, target_transformed

def findCells(img, k_factor=500, min_area=500, max_area=10000):
    '''

    Parameters
    ----------
    img         : image (as array).
    k_factor    : factor to imfer kernel size relative to image size. The default is 500.
    min_area    : minimum area of a cell. The default is 500.
    max_area    : maximum area of a cell. The default is 10000.

    Returns
    -------
    cnts_   : countours of cells.
    img_    : thresholded, manipulated version of image used to find cells
    img2    : image with cell outlines

    '''
    k = (np.ceil(np.array(img.shape) / k_factor) // 2 * 2 + 1).astype(int) # kernel
    img_ = cv2.GaussianBlur(img, k, 0)
    img_ = cv2.threshold(img_, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1].astype(np.uint8)
    cnts, hierarchy = cv2.findContours(255-img_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [cnts[i] for i in range(len(cnts)) if hierarchy[0][i][2] == -1] # count contours with no child contours only

    cnts_ = []
    img2 = cv2.cvtColor(downScale(img).copy(), cv2.COLOR_GRAY2RGB)
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            cnts_.append(c)
    cv2.drawContours(img2, cnts_, -1, (255, 0, 0), 3)
    
    return cnts_, img_, img2

def showImage(IMAGE):
    cv2.namedWindow('w', cv2.WINDOW_NORMAL)    
    cv2.imshow('w', IMAGE)
    cv2.resizeWindow('w', 800, 800)
    cv2.waitKey(0)

def createSummary(well, img0, img1, img2, img3, img4, img5, img6, img7, directory=os.getcwd()):
    '''

    Parameters
    ----------
    well        : Current well as string.
    img0        : pre (timepoint), channel 0
    img1        : post (timepoint), channel 0, unregistered
    img2        : post (timepoint), channel 0, registered
    img3        : pre (timepoint), channel 1
    img4        : post (timepoint), channel 1, unregistered
    img5        : post (timepoint), channel 1, registered
    img6        : thresholded version of img0 used for identifying cells.
    img7        : cell outlines
    directory   : Current directory. The default is os.getcwd().

    Returns
    -------
    None. Creates a subdirectory (/tmp) into which summaries for each well are created.

    '''
    if not os.path.exists(pth.Path(directory,'tmp')):
        os.mkdir(pth.Path(directory,'tmp'))
    cmap = mplc.ListedColormap(['black','white'], name='cmap', N=None)
    titles = {0: 'pre, chan 0', \
              1: 'post, chan 0, unreg', \
              2: 'post, chan 0, reg', \
              3: 'pre, chan 1', \
              4: 'post, chan 1, unreg', \
              5: 'post, chan 1, reg', \
              6: 'thresh', \
              7: 'cell outlines'}
    fig = mplp.figure(figsize=(8,8))
    gs = mplg.GridSpec(3, 3, wspace=0)
    ax = {}
    for idx, i in enumerate([img0, img1, img2, img3, img4, img5, img6, img7]):
        ax[idx] = fig.add_subplot(gs[idx])
        ax[idx].set_title(titles[idx])
        if idx==6:
            ax[idx].imshow(i, cmap=cmap)
        else:
            ax[idx].imshow(i)
        ax[idx].tick_params(axis='both', labelbottom=False, labelleft=False)
        #mplp.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    mplp.savefig('tmp/' + well + '.svg', dpi=300, format='svg')


def getFluorescence(data, time_reference, time_response, channel_template=0, channel_response=1, k_factor=500, min_area=500, max_area=10000, output_summaries=False):
    '''

    Parameters
    ----------
    data                : dictionary returned by loadImages() containing image data.
    time_reference      : timepoint to use as T0 (e.g. pre-drug; also used as the template for image registration)
    time_response       : timepoint to use as T1 (e.g. post-drug)
    channel_template    : channel to use for registration of each image. The default is 0.
    channel_response    : channel used to measure fluorescence response. The default is 1.
    k_factor:           : findCells()... The default is 500.
    min_area:           : findCells()... The default is 500.
    max_area:           : findCells()... The default is 10000.
    output_summaries    : whether to output visual summaries to /tmp for each well. The default is False.

    Returns
    -------
    data2   : dictionary...

    '''
    wells = sorted(data.keys())
    columns=['well', 'timepoint', 'cell', 'channel', 'fluorescence_min', 'fluorescence_max', 'fluorescence_mean', 'fluorescence_std', 'fluorescence_median', 'fluorescence_sum', 'error_flags']
    fluo = pd.DataFrame(columns=columns)
    for i in wells:
        print('\ngetFluorescence():',str(i)+'...')
        img = {time_reference:{}, time_response:{}} # <- dictionary of reference (timepoint) images and registered response (timepoint) images for the current well.
        img[time_reference][channel_template] = data[i][time_reference][channel_template].copy()
        img[time_reference][channel_response] = data[i][time_reference][channel_response].copy()
        flag, homography, img[time_response][channel_template] = improvedRegistration(data[i][time_response][channel_template], data[i][time_reference][channel_template]) # <- perform registration on template timepoint/channel for current well to obtain homography matrix
        img[time_response][channel_response] = cv2.warpPerspective(data[i][time_response][channel_response], homography, data[i][time_response][channel_response].shape) # <- Use this matrix to transform the original target image WRT template image.
        cells, img_thresh, img_cells = findCells(img[time_reference][channel_template], k_factor=k_factor, min_area=min_area, max_area=max_area)
        for jdx, j in enumerate(cells):
            #mask = np.zeros_like(downScale(template))
            mask = np.full_like(img[time_reference][channel_template].astype(np.double), np.nan)
            cv2.drawContours(mask, [j], -1, (1), thickness=cv2.FILLED) # <- make a mask at the current cell where all pixels within the cell have a value of 1, and all outside have a value of nan
            for k in [time_reference, time_response]:
                for l in [channel_template, channel_response]:
                    cell = img[k][l] * mask
                    cell = cell.flatten()[~np.isnan(cell.flatten())] # <- array containing all values for pixels within the cell and nowhere else
                    temp = pd.DataFrame({'well': [i], \
                                         'timepoint': [k], \
                                         'cell': [jdx], \
                                         'channel': [l], \
                                         'fluorescence_min': [min(cell)], \
                                         'fluorescence_max': [max(cell)], \
                                         'fluorescence_mean': [np.mean(cell)], \
                                         'fluorescence_std': [np.std(cell)], \
                                         'fluorescence_median': [np.median(cell)], \
                                         'fluorescence_sum': [np.sum(cell)], \
                                         'error_flags': [flag]} \
                                       )
                    fluo = pd.concat([fluo, temp]).reset_index(drop=True)
        if output_summaries==True:
            createSummary(i, \
                          img[time_reference][channel_template], \
                          data[i][time_response][channel_template], \
                          img[time_response][channel_template], \
                          img[time_reference][channel_response], \
                          data[i][time_response][channel_response], \
                          img[time_response][channel_response], \
                          img_thresh, \
                          img_cells \
                          )
    return fluo

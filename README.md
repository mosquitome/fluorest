# fluorest
Python module for automated analysis of fluorescent responses in live cell microscopy timecourses.

### install
Note: This has not been checked! There may be issues with installing OpenCV.

Use [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-python) to create an environment (called "cells") with all required software.

Note: ensure that conda-forge has been [added as a channel](https://conda-forge.org/docs/user/introduction.html).
```
conda create -n cells python=3.10 czifile=2019.7.2 pandas=2.1.1 numpy=1.26.0 cv2=4.8.1 matplotlib=3.8.0
```
### example
We used the automated stage function on a confocal microscope to image a series of wells is quick succession. For sets of wells, images were taken before drug addition, then after. This produced two multi-image files for each set of wells: one "pre" and one "post" drug file. Note: a tab-delimited file (metadata.txt) detailing the image files and their contents is required in the same directory as the images.
```
data = loadImages() # load images from current working directory
example_img = data['A1']['pre'][0] # get a single image as an example (well: 'A1', timepoint: 'pre', channel:0)
showImage(example_img) # look at the example image
cell_contours, threshold_img, outline_img = findCells(example_img, min_area=500, max_area=10000) # find cells in the example image (adjust min/max depending on objective/zoom etc.)
fluo = getFluorescence(data, 'pre', 'post', output_summaries=True) # automatically analyse data (register images before/after e.g. drug addition; identify cells; extract fluorescence measures for each cell) producing a dataframe. 
```

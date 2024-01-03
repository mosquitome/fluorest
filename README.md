# fluorest
Python module for automated analysis of fluorescent responses in live cell microscopy timecourses.

### example
Note: a tab-delimited file (metadata.txt) detailing the image files and their contents is required in the same directory as the images.
```
data = loadImages() # load images from current working directory
example_img = data['A1']['pre'][0] # get a single image as an example (well: 'A1', timepoint: 'pre', channel:0)
showImage(example_img) # look at the example image
cell_contours, threshold_img, outline_img = findCells(example_img, min_area=500, max_area=10000) # find cells in the example image (adjust min/max depending on objective/zoom etc.)
fluo = getFluorescence(data, 'pre', 'post', output_summaries=True) # automatically analyse data (register images before/after e.g. drug addition; identify cells; extract fluorescence measures for each cell) producing a dataframe. 

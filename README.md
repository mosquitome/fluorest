# fluorest
Python module for automated analysis of fluorescent responses in live cell microscopy timecourses.

### install
Use [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-python) to create an environment (called "cells") with all required software (see note about OpenCV below):

_Note: Although this should work on Linux/mac, only Windows has been tested._

_Note: ensure that conda-forge has been [added as a channel](https://conda-forge.org/docs/user/introduction.html)._

_Note: OpenCV cannot be installed with conda, so first create a conda env (called "cells"), and then use pip to install OpenCV within this env. The pip version inside the env needs to be used, and the path to this may vary depending on your operating system. Here I used Windows, and the executable was withing the \Scripts directory of the env. On Linux, for example, it may be held with other binaries in a /bin directory._
```
conda create -p path\to\env\cells python=3.10 czifile=2019.7.2 pandas=2.1.1 numpy=1.26.0 matplotlib=3.8.0
conda activate cells
path\to\env\cells\Scripts\pip3.exe install opencv-python==4.8.1.78
```
### example
We used the automated stage function on a confocal microscope to image a series of wells is quick succession. For sets of wells, images were taken before drug addition, then after. This produced two multi-image files for each set of wells: one "pre" and one "post" drug file.

_Note: a tab-delimited file (metadata.txt) detailing the image files and their contents in required in the same directory as the images._
```
data = loadImages() # load images from current working directory
example_img = data['A1']['pre'][0] # get a single image as an example (well: 'A1', timepoint: 'pre', channel:0)
showImage(example_img) # look at the example image
cell_contours, threshold_img, outline_img = findCells(example_img, min_area=500, max_area=10000) # find cells in the example image (adjust min/max depending on objective/zoom etc.)
fluo = getFluorescence(data, 'pre', 'post', output_summaries=True) # automatically analyse data (register images before/after e.g. drug addition; identify cells; extract fluorescence measures for each cell) producing a dataframe. 
```

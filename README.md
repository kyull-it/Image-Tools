# Image_Toolkits
This repository contains codes for the common pre-processing steps of image.

***

### Resize
This codes is intended to compare image resizing and cropping speed and accuracy.

&emsp;- `resize_pil.py`<br>
&emsp;- `resize_vips.py`<br>
 - Libvips library is much faster than pillow.
 
&emsp;- `resize_cv2.py`
 - This code is specially resizing images over 100 pixels and under 1000 pixels.

***

### Image Display
This code has three methods of showing image on jupyter environment.

&emsp;- `display.ipynb`<br>

***

### Image Contour
This code is for finding contours of object in an image.

&emsp;- `contour.ipynb` is for a binary image.<br>
&emsp;- `morphology_detection.ipynb` is finding texts from an image. :point_right: [Document (Korean)](https://docs.google.com/document/d/17_5oZ-b1TVHLbEnNMz3IE_Vy0IOdGf4vuIC6rGxDHWw/edit?usp=sharing)<br>
&emsp;- `gold-skeleton` convert an RGB image to gold skeleton (edge) image. <br>

***

### Image Color Extraction
This code is for extracting color representations of an image.

&emsp;- `color_gradients.ipynb` 


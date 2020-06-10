# Projects

## Importances

This tool was created to quickly visualize the feature importances of pixels in inputted SEM images. The model applies a pre-fitted u-net neural network to an input image and analyzes the effect of different pixel inputs on the outputted binary classification mask different methods like DeepLIFT, integrated gradients, etc.

Users can click on either:
- Click on a pixel to see how the output class of that pixel is affected by the input pixels
- Perform a whole-image feature attribution analysis

Users can observe:
- Positive attributions
- Negative attributions
- All attributions
- Absolute attributions

The tool also has sliders to control floor and ceiling values for coloring the feature attributions. To fully use the tool, user must upload both input image and corresponding ground truth mask.

Requirements:

PyQt, Keras, Tensorflow, OpenCV2, scikit-learn, Pillow and DeepExplain (https://github.com/marcoancona/DeepExplain)

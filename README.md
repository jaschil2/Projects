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

## GSA-Image

This tool was developed as part of the Gr-ResQ node at the University of Illinois at Urbana-Champaign (Project can also be found at https://github.com/nanoMFG/GSA-Image). It is used for post-processing graphene SEM images and extracting relevant information, such as graphene coverage or graphene alignment. The tool is also hosted on nanoHUB.org at https://nanohub.org/tools/gsaimage.

Example Segmentation using K-Means:

<img width="1248" alt="Screen Shot 2021-01-04 at 5 59 06 PM" src="https://user-images.githubusercontent.com/12614221/103588201-ab4e9380-4eb6-11eb-8d11-931493ad94c3.png">

## U-Net

This project contains my work developing a U-Net neural network for classifying SEM images of graphene (Project can also be found at https://github.com/nanoMFG/unet-sem). The model is part of Gr-ResQ with the objective to conduct image segmentation with no human involvement. The goal is to be able to quantify graphene quality automatically when users upload their SEM images into the Gr-ResQ database. Using the inputted synthesis procedures along with the quality metrics, one will be able to optimize the graphene synthesis process.

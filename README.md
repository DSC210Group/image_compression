# image_compression

In this project, we initially explore the linear algebraic methods using Singular Value Decomposition (SVD), Principle Components Analysis (PCA), Walsh–Hadamard Transformation (WHT) and Discrete Cosine Transformation (DCT) to extract important components of images and then we look at a AI-based state of art image compression called MLIC++.

We perform experiments on Kodak image dataset, composing of 24 pieces of lossless, true color (24 bits per pixel, aka "full color") images (You can find the image dataset here: https://r0k.us/graphics/kodak/).

The structure of this repository:
- pca: fold contains the code to perform PCA
- result_comparison: fold contains the code to evaluate different methods by computing metrics
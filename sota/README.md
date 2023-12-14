# DSC210SOTA
1.DSC210sota.ipynb is the final file for the sota solution.  
2.results.csv saves Bpp loss, PSNR and MS-SSIM of 24 pictures.  
3.I commented out the GPU-related parts in the original code from the paper, to ensure that we could run the code successfully on-devide.  
When you check the assignment of DSC 210, you should open the DSC210sota.ipynb and you could see three compressed images as examples. In this GitHub repository, the 'archive' folder contains the original input images required for this assignment. When you want to run the code, you need to download all the files from this GitHub repository and sequentially execute the code in the Jupyter Notebook (DSC210sota.ipynb). Doing so will create a folder named output_images in your directory, where you will find all the images after compression. And you could find the results.csv in the GitHub repository.  

## Environment
CompressAI 1.2.0b3  
Pytorch 2.0.1

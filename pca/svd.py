from PIL import Image
import numpy as np
import os
from dataset import ImageDataset
from sklearn.decomposition import PCA
from numpy.linalg import svd

def svd_compress(dt,idx):
    #get image
    img=dt[idx]

    #extract rgb channels
    datar=img[:,:,0]
    datag=img[:,:,1]
    datab=img[:,:,2]
    
    #apply svd on rgb channel seperately
    def compress_channel(channel, k):
        U, s, Vt = svd(channel, full_matrices=False)
        compressed = np.dot(U[:, :k], np.dot(np.diag(s[:k]), Vt[:k, :]))
        return compressed
    
    # Compress each channel
    k = 30  # Number of singular values to keep
    datar=compress_channel(datar, k)
    datag=compress_channel(datag, k)
    datab=compress_channel(datab, k)
    
    #combine rgb channels
    compressed_image = np.stack((datar, datag, datab), axis=2)
    compressed_image = np.clip(compressed_image, 0, 255).astype('uint8')
    
    #save image
    pil_image = Image.fromarray(compressed_image, 'RGB')
    pil_image.save(os.path.join('./svd',dt.get_filename(idx)))
    

if __name__== "__main__":
    dt=ImageDataset()
    for i in range(len(dt)):
        svd_compress(dt,i)
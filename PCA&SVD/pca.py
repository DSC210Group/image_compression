from PIL import Image
import numpy as np
import os
from dataset import ImageDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def img_compress(dt):
    """
        input:img dataset, numpy array
        output:img dataset, numpy array
    """
    
    #flatten the dataset
    data=np.array([i.flatten() for i in dt])
    
    #standardize data
    scaler = StandardScaler()
    datas = scaler.fit_transform(data)
    
    
    #apply pca on rgb channel seperately
    n_components=20
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(datas)
    data_pca=pca.inverse_transform(transformed_data)
    
    
    #resize image
    data_reverse=scaler.inverse_transform(data_pca)
    
    
    #save image
    for idx,img in enumerate(data_reverse):
        ori_img=Image.fromarray(img.reshape((512, 768, 3)).astype(np.uint8))
        save_path=os.path.join('./pca_whole',dt.get_filename(idx))
        ori_img.save(save_path)
    
    
#implement pca on single image
if __name__== "__main__":
    dt=ImageDataset()
    img_compress(dt)
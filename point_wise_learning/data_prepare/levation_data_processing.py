from PIL import Image
import numpy as np

def read_elevation(path):
    im = Image.open(r'C:\Users\96349\Documents\Downscale_data\elevation\exportImage.tif')
    pix = np.array(im.getdata()).reshape(im.size[0], im.size[1])
    return pix

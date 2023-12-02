import glob
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage import zoom

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","limegreen","red","yellow"])

def plot_gt(img, gt, pred, name):
    img = Image.open(img).convert('RGB')
    gt = Image.open(gt).convert('L')
    pred = Image.open(pred).convert('L')        
    
    gt = np.array(gt) 
    gt_shape = gt.shape   
    
    # img = img.resize(gt_shape[::-1])    
    img = np.array(img)
    
    pred = pred.resize(gt_shape[::-1])
    pred = np.array(pred) 
    pred = np.where(pred > 0, 255, 0)
    
    # print(pred.shape, gt.shape)
        
    green = np.where(pred == gt, pred, 0)    
    yellow = pred - gt
    yellow = np.where(yellow < 0, 0, yellow)
    
    red = gt - pred
    red = np.where(red < 0, 0, red)    
    
    green = np.stack((green, )*3, axis=-1)
    red = np.stack((red, )*3, axis=-1)
    yellow = np.stack((yellow, )*3, axis=-1)
    
    # print(green.shape, red.shape, yellow.shape)
    
    img = np.where(green == 255, [50, 205, 50], img)
    img = np.where(red == 255, [255, 0, 0], img)
    img = np.where(yellow == 255, [255,255,0], img)
    
    img = img.astype(np.uint8)
    img = Image.fromarray(img.astype(np.uint8)).resize((400, 400))
            
    # d = ImageDraw.Draw(img)
    # d.rectangle((0, img.height, img.width+20, img.height+20), fill='#000')
    
    img.save(name)
        
    # print(img.shape)
    # print(green.shape)
    
    # plt.figure('gt')
    # plt.imshow(gt)
    
    # plt.figure('Pred')
    # plt.imshow(pred)
    
    # # plt.figure('img')
    # # plt.imshow(img)
    
    # plt.figure('Intersection')
    # plt.imshow(green)
    
    # plt.figure('Yellow')
    # plt.imshow(yellow)
    
    # # # plt.figure('Red')
    # # # plt.imshow(red)
        
    # plt.figure('final')
    # # plt.imshow(img)
    # plt.axis('off')
    # plt.imshow(img)     
    
    # plt.show()

if __name__ == '__main__':

    images = glob.glob('./Unet-CLearning/data/Foot Ulcer Segmentation Challenge/validation/images/*')
    gt = glob.glob('./Unet-CLearning/data/Foot Ulcer Segmentation Challenge/validation/labels/*')
    data = glob.glob('./predictions/masks/*')
    
    for i in range(len(images)):
        name = data[i].split('//')[0].split('.png')[0] + '_final.png'
        print(name)
        plot_gt(images[i], gt[i], data[i], name)

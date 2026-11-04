import os
from ultralytics import YOLO


import numpy as np
import cv2
from tqdm import tqdm


def raw_target_detection():

    DATA_ROOT = './data/DressCode'
    
    predict_model = YOLO('./best.pt')

    for kind in ['dresses', 'lower_body', 'upper_body']:
        
        files = os.listdir(os.path.join(DATA_ROOT, kind, 'images'))
        outpath = os.path.join(DATA_ROOT, kind, 'cloth-mask')
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        files = [file  for file in files if file.split('.')[0].split('_')[1] == '1']
        # files = ['024763_1.jpg']
        for i,img_file in enumerate(tqdm(files)):
            image_path = os.path.join(DATA_ROOT, kind, 'images', img_file)
            raw_img = cv2.imread(image_path)
            
            if np.mean(raw_img[0:10,0:10,:]) > 253:
                mask = ((raw_img < 253) * 255).astype(np.uint8)
                mask = cv2.resize(mask, (768, 1024), interpolation=cv2.INTER_CUBIC)
                mask = cv2.GaussianBlur(mask, ksize=(5,5), sigmaX=0)
                mask = ((mask > 128) * 255).astype(np.uint8)
                mask = cv2.GaussianBlur(mask, ksize=(9,9), sigmaX=0)
                mask = ((mask > 128) * 255).astype(np.uint8)
                mask = mask[:,:,0]
                cont,_ = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            else:
                cont = []
                results = predict_model.predict(raw_img)
                try:
                    for res in results:
                        for xy in res.masks.xy:
                            p = xy.astype(np.int32)
                            p = np.expand_dims(p,1)
                            cont.append(p)
                except AttributeError:
                    print(image_path)
                    print(results)
                    exit()
                
            
            
            cont = max(cont, key=cv2.contourArea)
            
            new_mask = np.zeros_like(raw_img)
            new_mask = cv2.drawContours(image=new_mask, contours=[cont], contourIdx=-1, color=(255,255,255), thickness=cv2.FILLED)
            new_mask = cv2.cvtColor(new_mask, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(outpath, img_file), img=new_mask)
        
if __name__ == '__main__':
    raw_target_detection()

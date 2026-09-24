from ultralytics import YOLO

import os

 
os.environ['ULTRALYTICS_OFFLINE'] = '1'
if __name__ == '__main__':

    model = YOLO(r"yolov8n.pt")


    results = model.train(
    data=r"dataset\data.yaml",  
    epochs=100,       
    imgsz=640,        
    batch=2,         
    device=0,
    workers=2, 

    ##resume=True,

    save_period=1,   
        plots=True 
    
                    )
from ultralytics import YOLO

def train_model():

    model = YOLO('yolov8n.pt')
    
    results = model.train(data='data.yaml', epochs=200, imgsz=640)
    
    print(results)  
    
if __name__ == '__main__':
    train_model()

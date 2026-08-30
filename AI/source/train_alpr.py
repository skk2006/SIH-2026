from ultralytics import YOLO
import os

def main():
    print("Initializing YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt")
    
    # Path to the dataset configuration
    data_yaml = os.path.abspath("dataset_lp/data.yaml")
    
    if not os.path.exists(data_yaml):
        print(f"Error: Could not find dataset config at {data_yaml}")
        return
        
    print(f"Starting training on {data_yaml} with GPU...")
    # Train the model. 
    # Using 10 epochs so it completes in a reasonable time on the RTX 3050 
    # while still providing a massive accuracy boost over the Haar cascade.
    model.train(
        data=data_yaml,
        epochs=10,
        imgsz=640,
        device="cuda",
        batch=16,
        workers=0,
        name="license_plate_detector"
    )
    
    print("Training complete! The best model weights are saved in runs/detect/license_plate_detector/weights/best.pt")

if __name__ == "__main__":
    main()

import cv2
import sys
import os

from pipeline import DeepMultimodalThreatPipeline

def test_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"Loaded {len(frames)} frames from {video_path}")
    
    pipeline = DeepMultimodalThreatPipeline(fight_threshold=0.0)
    
    fps = 30
    stride = 4
    window_size = pipeline.num_frames * stride
    
    for i in range(0, min(len(frames), window_size * 2), window_size):
        clip_frames = frames[i:i+window_size:stride][:pipeline.num_frames]
        clip = pipeline.preprocess_clip(clip_frames).to(pipeline.device)
        if pipeline.device == "cuda":
            clip = clip.half()
            
        import torch
        with torch.no_grad():
            preds = pipeline.model(clip)
            print(f"Raw logits: {preds}")
            probs = torch.softmax(preds, dim=1)[0]
            print(f"Clip {i} to {i+window_size}: Class 0: {probs[0].item():.4f}, Class 1: {probs[1].item():.4f}")

if __name__ == "__main__":
    test_video("c:/Users/Kanishk/Pictures/SIH2K26/project/AI/source/uploads/loitering_analyze_Screen Recording 2026-08-28 234523.mp4")

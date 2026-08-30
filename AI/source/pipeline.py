import cv2
import torch
import numpy as np
import os
from pytorchvideo.models.hub import x3d_s
import torch.nn as nn
import torchvision.transforms as T


class DeepMultimodalThreatPipeline:
    def __init__(self, fight_threshold=0.60, model_path="fight_detector_x3d_s_zipped.pt"):
        self.fight_threshold = fight_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load X3D-S model
        self.model = x3d_s(pretrained=False)
        # Adjust head for 2 classes
        self.model.blocks[5].proj = nn.Linear(2048, 2)
        
        # Load weights
        base_path = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_path, model_path)
        
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            # Remove "model." prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k.replace("model.", "")] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict)
            print(f"Loaded fight detector weights from {weights_path}")
        else:
            print(f"Warning: Fight detector weights not found at {weights_path}")
            
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
        if self.device == "cuda":
            self.model = self.model.half()
            
        self.frame_buffer = []
        self.num_frames = 13  # x3d_s default uses 13 frames
        
        # Define the transform
        self.transform = T.Compose([
            T.Lambda(lambda x: x / 255.0),
            T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            T.Resize((160, 160)),
        ])

    def preprocess_clip(self, frames):
        # Convert list of numpy frames (H, W, C) to (C, T, H, W) tensor
        frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        clip = np.stack(frames) # (T, H, W, C)
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float() # (C, T, H, W)
        
        # Subsample to exactly 13 frames
        step = max(1, clip.shape[1] // self.num_frames)
        indices = torch.linspace(0, clip.shape[1] - 1, self.num_frames).long()
        clip = torch.index_select(clip, 1, indices)
        
        # Apply spatial transforms per frame
        clip_t = []
        for i in range(clip.shape[1]):
            frame = clip[:, i, :, :]
            frame = self.transform(frame)
            clip_t.append(frame)
            
        clip = torch.stack(clip_t, dim=1) # (C, T, H, W)
        return clip.unsqueeze(0) # (1, C, T, H, W)

    def process_frame(self, frame):
        self.frame_buffer.append(frame.copy())
        threat_level = "NORMAL"
        conf = 0.0
        
        if len(self.frame_buffer) >= self.num_frames * 2: # Keep some history, process every 13 frames essentially
            clip = self.preprocess_clip(self.frame_buffer[-self.num_frames:])
            clip = clip.to(self.device)
            if self.device == "cuda":
                clip = clip.half()
                
            with torch.no_grad():
                preds = self.model(clip)
                probs = preds[0]
                fight_prob = probs[1].item()
                
            if fight_prob >= self.fight_threshold:
                threat_level = "PHYSICAL ALTERCATION"
                conf = fight_prob
            
            # Keep overlapping buffer for smooth detection
            self.frame_buffer = self.frame_buffer[-self.num_frames//2:]
            
        return frame, threat_level, conf

    def analyze_full_video_deep(self, all_frames, fps):
        threat_spans = []
        
        # We need a temporal stride. If we want 13 frames over ~1.7 seconds, stride = fps * 1.7 / 13 = ~4
        stride = 4
        window_size = self.num_frames * stride
        
        if len(all_frames) < window_size:
            return {"threat_spans": threat_spans}
            
        step = window_size
        for i in range(0, len(all_frames) - window_size, step):
            # Sample 13 frames uniformly from the window
            clip_frames = all_frames[i:i+window_size:stride][:self.num_frames]
            
            clip = self.preprocess_clip(clip_frames).to(self.device)
            if self.device == "cuda":
                clip = clip.half()
                
            with torch.no_grad():
                preds = self.model(clip)
                probs = preds[0]
                fight_prob = probs[1].item()
                
            if fight_prob >= self.fight_threshold:
                threat_spans.append({
                    "max_confidence": float(fight_prob),
                    "threat_level": "PHYSICAL ALTERCATION",
                    "start_frame": i,
                    "end_frame": i + window_size
                })
                
        return {"threat_spans": threat_spans}

import torch
from pytorchvideo.models.hub import x3d_s

state_dict = torch.load("c:/Users/Kanishk/Pictures/SIH2K26/project/AI/source/fight_detector_x3d_s_zipped.pt", map_location="cpu")

# Find the last layer's weight shape
for key, value in state_dict.items():
    if "blocks.5.proj.weight" in key:
        print(f"{key}: {value.shape}")
        
print("Number of classes from the last layer:")
last_layer = [v for k, v in state_dict.items() if 'weight' in k][-1]
print(last_layer.shape)


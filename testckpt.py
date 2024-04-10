import torch

# # Load the saved checkpoint
# checkpoint_path = '/home/shubhamp/Downloads/binarymodified_test_freezebackbone/checkpoints_binaryClf15k/CITY_768x768/best_deeplabv3plus_resnet101_cityscapes_os8.pth'
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Load on CPU

# # Inspect the keys of the loaded checkpoint
# print("Keys in the loaded checkpoint:")
# print(checkpoint.keys())

# # Inspect the 'model_state' dictionary to check for backbone keys
# model_state = checkpoint['model_state']
# print("Keys in the 'model_state' dictionary:")
# print(model_state.keys())

# # Load the saved checkpoint
# checkpoint_path1 = '/home/shubhamp/Downloads/Segmentation_models/DeepLabV3Plus_Emarg15k/checkpoints_dlv3+Seg(15k)/best_deeplabv3plus_resnet101_cityscapes_os8.pth'
# checkpoint = torch.load(checkpoint_path1, map_location=torch.device('cpu'))  # Load on CPU

# # Inspect the keys of the loaded checkpoint
# print("Keys in the loaded checkpoint:")
# print(checkpoint.keys())

# # Inspect the 'model_state' dictionary to check for backbone keys
# model_state = checkpoint['model_state']
# print("Keys in the 'model_state' dictionary:")
# print(model_state.keys())


import torch

# Load the first checkpoint
checkpoint_path1 = '/home/shubhamp/Downloads/binarymodified_test_freezebackbone/checkpoints_binaryClf15k/CITY_768x768/best_deeplabv3plus_resnet101_cityscapes_os8.pth'
checkpoint1 = torch.load(checkpoint_path1, map_location=torch.device('cpu'))

# Load the second checkpoint
checkpoint_path2 = '/home/shubhamp/Downloads/Segmentation_models/DeepLabV3Plus_Emarg15k/checkpoints_dlv3+Seg(15k)/best_deeplabv3plus_resnet101_cityscapes_os8.pth'
checkpoint2 = torch.load(checkpoint_path2, map_location=torch.device('cpu'))

# Get the backbone weights from both checkpoints
backbone_weights1 = {k: v.numpy() for k, v in checkpoint1['model_state'].items() if k.startswith('backbone.')}
backbone_weights2 = {k: v.numpy() for k, v in checkpoint2['model_state'].items() if k.startswith('backbone.')}

# Compare the keys of the backbone weights
if backbone_weights1.keys() == backbone_weights2.keys():
    # Compare the values of the backbone weights
    values_match = all((backbone_weights1[k] == backbone_weights2[k]).all() for k in backbone_weights1.keys())
    if values_match:
        print("Backbone weights in both checkpoints match.")
    else:
        print("Backbone weights in both checkpoints do not match.")
else:
    print("Keys of backbone weights in both checkpoints do not match.")




# AI_MI
Super resolution challenge for course AI for Medical Imaging

Extract data from this link into ```dataset``` directory.
(https://sandbox.zenodo.org/record/1104155#.YzGQFtJBxkg), including the test target resolution from canvas.
The resulting folder structure look like this:
```
dataset
|   README.md
|__ dataset_train_plus_test_source_resolution
    |__ test
    |   |__ source_resolution
    |   |       *.nii.gz
    |   |       ...
    |   |__ target_resolution
    |           *.nii.gz
    |           ...
    |__ train
    |   |__ source_resolution
    |   |       *.nii.gz
    |   |       ...
    |   |__ target_resolution
    |           *.nii.gz
    |           ...
```

Run ```main.py``` with argument ```--only_test=True``` to test on saved best model.

The discrete vae model is based on the implementation of: 
https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
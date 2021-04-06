# mvtecad_pytroch
Unofficial pytorch dataset class for MVTec Anomaly Detection Dataset

## Requirement
* torchvision

## Installation
```
$ python setup.py install
```

## Usage

1. download MVTecAD data from [the official site](https://www.mvtec.com/company/research/datasets/mvtec-ad).
2. unzip tar ball in `./data/`.
```
$ tar xJvf mvtec_anomaly_detection.tar.xz
$ rm mvtec_anomaly_detection.tar.xz
$ ls data/
bottle  cable  capsule  carpet  grid  hazelnut  leather  license.txt  metal_nut  pill  readme.txt  screw  tile  toothbrush  transistor  wood  zipper
```
3. run the following script to ensure installation complete collectly.
```python
from torchvision import transforms
from mvtecad_pytorch.dataset import MVTecADDataset


_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])
dataset = MVTecADDataset(root="./data", target="capsule", transforms=_transforms, mask_transforms=_transforms, train=False)
for i in range(len(dataset)):
    img, mask, label = dataset[i]
    print(img.shape, mask.shape, label)
```
4. use the dataset in your anomaly detection code!
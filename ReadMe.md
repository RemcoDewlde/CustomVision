[![HitCount](http://hits.dwyl.com/remcodewlde/CustomVision.svg)](http://hits.dwyl.com/remcodewlde/CustomVision)

# Custom Vision Demo

In this reposetory are two projects, one for the normal sized tensorflow protobuf file from azure custom vision, the other is the lite version of the same model.
 

## Prerequisites
* Python(3.5, 3,6, 3,7)
* Cuda toolkit 10.0
* CuDnn
* pip3


## Installation
You do not need to install Cuda and CuDnn but you wont have gpu hardware acceleration enabled. In laymen terms it will be slow. Do install the pip packages.

* **Cuda:** I won't go into the cuda install because it's terrifying, but here is the download [link](https://developer.nvidia.com/cuda-toolkit-archive). Make sure you are downloading 10.0!
* **CuDnn** [Download this zip](https://developer.nvidia.com/cudnn) and follow the readme instructions in this zip  
* Install all pip packages ```pip install -R requirements.txt```
 
 
## Usage
In both projects there is a `main.py` and a `optimize.py`. The main file is the project not optimized for cpu and gpu usage.
The optimeze file is the project better optimzed than the main file (there are probably more optimisations possible) it does however run a lot faster even on just the CPU (approx: ~5 times better in my case).
The following commands will work in the customVision project and in the lite version.

To use the main file:
```bash
    $ python main.py XAXwmMu8otM
    # the argument parsed is the youtube video id of the youtube video you want to detect objects in
```

To use the optimeze file:
```bash
    $ python optimize.py XAXwmMu8otM
    # the argument parsed is the youtube video id of the youtube video you want to detect objects in
```

 

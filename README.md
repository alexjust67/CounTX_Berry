## CounTX_Berry: Specialized CounTX fork for berry and cluster guided counting
[Alessandro Giustina](mailto:giustinalessandro@gmail.com), Fabio Poiesi

### Contents
* [Preparation](#preparation)
* [Pre-trained Weights](#pre-trained-weights)
* [Acknowledgements](#acknowledgements)

### Preparation
#### Set Up Anaconda Environment:

The following commands will create a suitable Anaconda environment for running the CounTX training and inference procedures.

```
conda create --name countx-environ python=3.7
conda activate countx-environ
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
pip install scipy
pip install imgaug
git clone git@github.com:niki-amini-naieni/CounTX.git
cd CounTX/open_clip
pip install .
```
* If torchvision==0.11.0+cu111 torchaudio==0.10.0 aren't found the defaults can be downloaded.
* This repository uses [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+. This fix can be implemented by replacing the file timm/models/layers/helpers.py in the timm codebase with the file [helpers.py](https://github.com/niki-amini-naieni/CounTX/blob/main/helpers.py) provided in this repository.

### Pre-Trained Weights
The model weights used in the paper can be downloaded from [Google Drive link (1.3 GB)](https://drive.google.com/file/d/1Vg5Mavkeg4Def8En3NhceiXa-p2Vb9MG/view?usp=sharing). To reproduce the results in the paper, run the program after activating the Anaconda environment set up in [Preparation](#preparation). Make sure that the model file name refers to the model that you downloaded.

The checkpoint directory must be changed in the evaluator.py file.
### Framework usage
The framework is divided in modules:

* The main module from which everything is controlled is evaluator.py 
* the main.py file contains the main function that is called in evaluator.py in which the modules are layed out:
  * density_map_creator is the model itself which needs as inputs the model, the query, the image, the kernel size and stride and outputs the density map.
  * clustercount is the module responsible for the image postprocessing and for the counting, it takes as input the density map, the treshold and the original image and it outputs the cluster map and the number of clusters.
  * the showimagefun is the visualization function and contains all the code that is needed to display the results and to use the genralized output part.

### Acknowledgements

The CounTX_Berry repository is based on the [CounTX repository](https://github.com/niki-amini-naieni/CounTX) and uses code from the [OpenCLIP repository](https://github.com/mlfoundations/open_clip). If you have any questions about our code implementation, please contact us at [agiustina@fbk.eu](mailto:agiustina@fbk.eu).      

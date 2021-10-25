# MS-STAN
# A Multi-Stage Spatio-Temporal Adaptive Network for Video Super-Resolution
 
The official pytorch source code (partially cleaned) for the [A Multi-Stage Spatio-Temporal Adaptive Network for Video Super-Resolution]. It is currently under review.*

## Architecture
![MS_STAN](https://user-images.githubusercontent.com/36630680/123551817-ea92d880-d7a5-11eb-83f8-55532ca28b98.jpg)

## Environment

- python == 3.7
- pytorch == 1.4.0

## Datasets

### Vimeo-90K
We use Vimeo-90K dataset for training. Vimeo90K dataset can be downloaded from [here](http://toflow.csail.mit.edu/).

## Examples to run the Codes

The basic usage of the codes for testing MS-STAN model on Vid4 dataset is as follows:

- **For testing**:

	```python test.py```


### **If you find our codes helpful, please kindly cite the following papers. Thanks!**

@inproceedings{MSTEnet-vcip,  
　　title={Video Super Resolution Using Temporal Encoding ConvLSTM and Multi-Stage Fusion},  
　　author={Zhang, Yuhang and Chen, Zhenzhong and Liu, Shan},  
　　booktitle={2020 IEEE International Conference on Visual Communications and Image Processing (VCIP)},  
　　pages={298--301},  
　　year={2020},  
　　organization={IEEE}  
}

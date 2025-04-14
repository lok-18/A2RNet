# A2RNet
[![AAAI](https://img.shields.io/badge/AAAI-2025-purple)](https://aaai.org/conference/aaai/aaai-25/)
[![LICENSE](https://img.shields.io/badge/License-MIT-green)](https://github.com/lok-18/IGNet/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-orange)](https://pytorch.org/)

### A2RNet: Adversarial Attack Resilient Network for Robust Infrared and Visible Image Fusion
in the 39th Annual AAAI Conference on Artificial Intelligence (**AAAI 2025**) 🔥🔥🔥   
by [Jiawei Li](https://scholar.google.com.hk/citations?user=xWy8RZEAAAAJ&hl=zh-CN), [Hongwei Yu](https://scholar.google.com.hk/citations?hl=zh-CN&user=cDidt64AAAAJ), [Jiansheng Chen](https://scholar.google.com.hk/citations?user=A1gA9XIAAAAJ&hl=zh-CN&oi=ao), [Xinlong Ding](https://scholar.google.com.hk/citations?user=JY9oXVIAAAAJ&hl=zh-CN&oi=ao), Jinlong Wang, [Jinyuan Liu](https://scholar.google.com.hk/citations?user=a1xipwYAAAAJ&hl=zh-CN&oi=ao), [Bochao Zou](https://scholar.google.com.hk/citations?user=Cb29A3cAAAAJ&hl=zh-CN&oi=ao) and [Huimin Ma](https://scholar.google.com.hk/citations?user=32hwVLEAAAAJ&hl=zh-CN&oi=ao) 
- [[*AAAI*]](https://ojs.aaai.org/index.php/AAAI/article/view/32504)
- [[*arXiv*]](https://arxiv.org/abs/2412.09954)
- [[*Supplementary*]](https://github.com/lok-18/A2RNet/blob/main/supp/Supplementary.pdf)

**Different adversarial operations in IVIF (Motivation):**
<div align=center>
<img src="https://github.com/lok-18/A2RNet/blob/main/fig/motivation.png" width="50%">
</div>  

**Framework of our proposed A2RNet:**
<div align=center>
<img src="https://github.com/lok-18/A2RNet/blob/main/fig/A2RNet.png" width="100%">
</div> 

### ‼️*Requirements* 
> - python 3.10  
> - torch 1.13.0
> - torchvision 0.14.0
> - opencv 4.9
> - numpy 1.26.4
> - pillow 10.3.0

### 📑*Dataset setting*
> We give several test image pairs as examples in [[*MFNet*]](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) and [[*M3FD*]](https://github.com/JinyuanLiu-CV/TarDAL) datasets, respectively.
> 
> Moreover, you can set your own test datasets of different modalities under ```./test_images/...```, like:   
> ```
> test_images
> ├── ir
> |   ├── 1.png
> |   ├── 2.png
> |   └── ...
> ├── pseudo_label
> |   ├── 1.png
> |   ├── 2.png
> |   └── ...
> ├── vis
> |   ├── 1.png
> |   ├── 2.png
> |   └── ...
> ```
> 
> Note that the detailed process of generating pseudo-labels is provided in the [[*Supplementary*]](https://github.com/lok-18/A2RNet/blob/main/supp/Supplementary.pdf). Alternatively, you may use the results from other SOTA methods as pseudo-labels and place them in the ```./test_images/pseudo_label/``` for supervision.
>
> The configuration of the training dataset is similar to the aforementioned format.

### 🖥️*Test*
> The pre-trained model ```model.pth``` has given in [[*Google Drive*]](https://drive.google.com/file/d/1X_e1T0dAq0pYQI_nmEUzZpSzWTJi4kgQ/view?usp=drive_link) and [[*Baidu Yun*]](https://pan.baidu.com/s/1hjSBWlhGy46M8oD6VG-2qw?pwd=AAAI).
> 
> Please put ```model.pth``` into ```./model/``` and run ```test_robust.py``` to get fused results. You can check them in:
> ```
> results
> ├── 1.png
> ├── 2.png
> └── ...

### ⌛*Train*
> You can also utilize your own data to train a new robust fusion model with:
> ```
> python train_robust.py
> ```

### 🌟*Experimental results*
> Under PGD attacks, we compared our proposed A2RNet with [[*TarDAL*]](https://github.com/JinyuanLiu-CV/TarDAL), [[*SeAFusion*]](https://github.com/Linfeng-Tang/SeAFusion), [[*IGNet*]](https://github.com/lok-18/IGNet), [[*PAIF*]](https://github.com/LiuZhu-CV/PAIF), [[*CoCoNet*]](https://github.com/runjia0124/CoCoNet), [[*LRRNet*]](https://github.com/hli1221/imagefusion-LRRNet) and [[*EMMA*]](https://github.com/Zhaozixiang1228/MMIF-EMMA).
> 
> **Fusion results:**
> <div align=center>
> <img src="https://github.com/lok-18/A2RNet/blob/main/fig/Fusion.png" width="100%">
> </div>
>
> <br>After retaining the fusion results of all methods on [[*YOLOv5*]](https://github.com/ultralytics/yolov5) and [[*DeepLabV3+*]](https://github.com/VainF/DeepLabV3Plus-Pytorch), we compare the corresponding detection and segmentation results with A2RNet.</br>
> 
> **Detection & Segmentation results:**
> <div align=center>
> <img src="https://github.com/lok-18/A2RNet/blob/main/fig/Downstream.png" width="100%">
> </div>
> Please refer to the paper for more experimental results and details.
>
### 🗒️*Citation*
> ```
> @inproceedings{li2025a2rnet,
>    title={A$^2$RNet: Adversarial Attack Resilient Network for Robust Infrared and Visible Image Fusion},
>    author={Li, Jiawei and Yu, Hongwei and Chen, Jiansheng and Ding, Xinlong and Wang, Jinlong and Liu, Jinyuan and Zou, Bochao and Ma, Huimin},
>    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
>    volume={39},
>    number={5},
>    pages={4770--4778},
>    year={2025}
> }
> ```
>
### 🧩*Realted works*
> - Jiawei Li, Jiansheng Chen, Jinyuan Liu and Huimin Ma. ***Learning a Graph Neural Network with Cross Modality Interaction for Image Fusion***. Proceedings of the 31st ACM International Conference on Multimedia (**ACM MM**), 2023: 4471-4479. [[*Paper*]](https://dl.acm.org/doi/10.1145/3581783.3612135) [[*Code*]](https://github.com/lok-18/IGNet)
> - Jiawei Li, Jinyuan Liu, Shihua Zhou, Qiang Zhang and Nikola K. Kasabov. ***GeSeNet: A General Semantic-guided Network with Couple Mask Ensemble for Medical Image Fusion***. IEEE Transactions on Neural Networks and Learning Systems (**IEEE TNNLS**), 2024, 35(11): 16248-16261. [[*Paper*]](https://ieeexplore.ieee.org/document/10190200) [[*Code*]](https://github.com/lok-18/GeSeNet)
> - Jiawei Li, Jinyuan Liu, Shihua Zhou, Qiang Zhang and Nikola K. Kasabov. ***Learning a Coordinated Network for Detail-refinement Multi-exposure Image Fusion***. IEEE Transactions on Circuits and Systems for Video Technology (**IEEE TCSVT**), 2023, 33(2): 713-727. [[*Paper*]](https://ieeexplore.ieee.org/abstract/document/9869621)
> - Jia Lei, Jiawei Li, Jinyuan Liu, Bin Wang, Shihua Zhou, Qiang Zhang, Xiaopeng Wei and Nikola K. Kasabov. ***MLFuse: Multi-scenario Feature Joint Learning for Multi-Modality Image Fusion***. IEEE Transactions on Multimedia (**IEEE TMM**), 2024. [[*Paper*]](https://github.com/jialei-sc/MLFuse) [[*Code*]](https://github.com/jialei-sc/MLFuse)
>
### 🙇‍♂️*Acknowledgement*
>We would like to express our gratitude to [[ESSAformer]](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_ESSAformer_Efficient_Transformer_for_Hyperspectral_Image_Super-resolution_ICCV_2023_paper.html) for inspiring our work! Please refer to their excellent work for more details.
> 
### 📬*Contact*
> If you have any questions, please create an issue or email to me ([Jiawei Li](mailto:ljw19970218@163.com)).

# M³-SLR
Official implement code of [M³-SLR: Self-Supervised Pretraining with MaxFlow MaskFeat for Improved Multi-View Sign Language Representation.](https://github.com/fossbk/M3-SLR/tree/main)
## Introduction
<img src="images/Pipeline.png">

SLR faces significant hurdles, notably with Visually Indistinguishable Signs (VISigns) – signs that appear identical from a single viewpoint – and the practical challenge of deploying computationally expensive multi-view systems on single cameras. This project introduces M³-SLR, a comprehensive framework designed to tackle these issues. The core proposal involves three key innovations: (1) a novel optical flow-guided MaxFlow Cube Masking strategy for MaskFeat pretraining method to compel models to learn fine-grained sign dynamics crucial for distinguishing similar signs, (2) an effective multi-view architecture, UF3V with Dual Co-Attention, specifically designed to fuse information from multiple viewpoints and resolve ambiguities like VISigns, and (3) an efficient knowledge distillation process to transfer the enhanced capabilities of the multi-view system into a high-performance, practical single-view model capable of distinguishing VISigns despite single-view input. The ultimate goal of M³-SLR is to significantly improve robustness against ambiguous signs and deliver a state-of-the-art SLR model suitable for real-world, single-camera applications.

## Performance
### Multi-VSL200
 Model                         | Top-1 | Top-5 | Checkpoint | 
|------------------------------|-----------|-----------|------------|
| UniFormer w MaxFlow MaskFeat |   86.37   |   96.94   | [link](https://drive.google.com/drive/folders/1WIv8MRgLc3MurnT1ApSN9w7KtU8A7ow2?usp=drive_link) |
| UsimKD                       |   88.14   |   96.28   | [link](https://drive.google.com/drive/folders/1bL7kGwUJRzRsATXZSXBH6-3YnGuKQDaX?usp=drive_link) |
| UF3V                         |   92.11   |   97.90   | [link](https://drive.google.com/drive/folders/1FuoiVl-v1GVkpH2HGtqWhY6NteahDKKG?usp=drive_link) |

### MM-WLAuslan
Model                          | STU Top-1 | STU Top-5 | ITW Top-1 | ITW Top-5 | SYN Top-1 | SYN Top-5 | TED Top-1 | TED Top-5 | AVG. Top-1 | AVG. Top-5 | Checkpoint |
|------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|------------|------------|------------|
| UniFormer w MaxFlow MaskFeat |   84.78   |   97.29   |   52.51   |   74.21   |   64.07   |   86.32   |   84.18   |   96.67   |    71.39   |    87.87   | [link](https://drive.google.com/drive/folders/1kFlA7Uf6_D2fWSMb0LA0h8d_l9b0_TLB?usp=drive_link) |
| UsimKD                       |   85.93   |   97.90   |   59.10   |   80.87   |   67.33   |   88.77   |   85.33   |   97.23   |    74.42   |    91.19   | [link](https://drive.google.com/drive/folders/1pJ6hadjbVxV8-S6eD0_Q0btNwFsS2hJt?usp=drive_link) |
| UF3V                         |   91.49   |   98.98   |   71.13   |   88.90   |   75.35   |   91.99   |   90.14   |   98.46   |    82.03   |    94.58   | [link](https://drive.google.com/drive/folders/1KLaA2XchXrU_RDS4F92Tj2dXEoH5mKAS?usp=drive_link) |

## Data preparation
Please follow the instruction in [DATA.md](DATA.md)
## Usage
### Package
Packages could be installed by:
```
pip install -r requirements.txt
```
### Training
- UniFormer pretraining with MaxFlow MaskFeat on Multi-VSL200 and MM-WLAuslan:
```
python3 main.py --config configs/Uniformer/MaskUFOneView_MultiVSL200.yaml
python3 main.py --config configs/Uniformer/MaskUFOneView_MMAuslan.yaml
```
- UniFormer finetuning on Multi-VSL200 and MM-WLAuslan:
```
python3 main.py --config configs/Uniformer/UFOneView_MultiVSL200.yaml
python3 main.py --config configs/Uniformer/UFOneView_MMAuslan.yaml
```
- UF3V finetuning on Multi-VSL200 and MM-WLAuslan:
```
python3 main.py --config configs/Uniformer/UFThreeView_MultiVSL200.yaml
python3 main.py --config configs/Uniformer/UFThreeView_MMAuslan.yaml
```
- UsimKD finetuning on Multi-VSL200 and MM-WLAuslan:
```
python3 main.py --config configs/Uniformer/UsimKD_MultiVSL200.yaml
python3 main.py --config configs/Uniformer/UsimKD_MMAuslan.yaml
```
### Testing
- UniFormer testing on Multi-VSL200 and MM-WLAuslan:
```
python3 main.py --config configs/Uniformer/test_cfg/UFOneView_MultiVSL200.yaml
python3 main.py --config configs/Uniformer/test_cfg/UFOneView_MMAuslan.yaml
```
- UF3V testing on Multi-VSL200 and MM-WLAuslan:
```
python3 main.py --config configs/Uniformer/test_cfg/UFThreeView_MultiVSL200.yaml
python3 main.py --config configs/Uniformer/test_cfg/UFThreeView_MMAuslan.yaml
```
- UsimKD testing on Multi-VSL200 and MM-WLAuslan:
```
python3 main.py --config configs/Uniformer/test_cfg/UsimKD_MultiVSL200.yaml
python3 main.py --config configs/Uniformer/test_cfg/UsimKD_MMAuslan.yaml
```
## Acknowledgement
This project is built by iBME lab at School of Electrical & Electronic Engineering, Hanoi University of Science and Technology, Vietnam. It is funded by Hanoi University of Science and Technology under project number T2023-PC-028.

# M³-SLR
Official implement code of [M³-SLR: Self-Supervised Pretraining with MaxFlow MaskFeat for Improved Multi-view Sign Dynamics Representation.](https://github.com/fossbk/M3-SLR/main/README.md)
## Introduction
<img src="images/Pipeline.png">

SLR faces significant hurdles, notably with Visually Indistinguishable Signs (VISigns) – signs that appear identical from a single viewpoint – and the practical challenge of deploying computationally expensive multi-view systems on single cameras. This project introduces M³-SLR, a comprehensive framework designed to tackle these issues. The core proposal involves three key innovations: (1) a novel optical flow-guided MaxFlow Cube Masking strategy for MaskFeat pretraining method to compel models to learn fine-grained sign dynamics crucial for distinguishing similar signs, (2) an effective multi-view architecture, UF3V with Dual Co-Attention, specifically designed to fuse information from multiple viewpoints and resolve ambiguities like VISigns, and (3) an efficient knowledge distillation process to transfer the enhanced capabilities of the multi-view system into a high-performance, practical single-view model capable of distinguishing VISigns despite single-view input. The ultimate goal of M³-SLR is to significantly improve robustness against ambiguous signs and deliver a state-of-the-art SLR model suitable for real-world, single-camera applications.

## Performance
 Model                         |   Dataset    | Top-1 Acc | Top-5 Acc | Checkpoint | 
|------------------------------|--------------|-----------|-----------|------------|
| UniFormer w MaxFlow MaskFeat | Multi-VSL200 |   86.37   |   96.94   | [link](https://drive.google.com/drive/folders/12dScaCjePvTyxvlWElGVTYv12UFcHN9U?usp=drive_link) |
| UsimKD                       | Multi-VSL200 |   88.14   |   96.28   | [link](https://drive.google.com/drive/folders/12dScaCjePvTyxvlWElGVTYv12UFcHN9U?usp=drive_link) |
| UF3V                         | Multi-VSL200 |   92.11   |   97.90   | [link](https://drive.google.com/drive/folders/12dScaCjePvTyxvlWElGVTYv12UFcHN9U?usp=drive_link) |
| UniFormer w MaxFlow MaskFeat | MM-WLAuslan  |   71.39   |   87.87   | [link](https://drive.google.com/drive/folders/12dScaCjePvTyxvlWElGVTYv12UFcHN9U?usp=drive_link) |
| UsimKD                       | MM-WLAuslan  |   --.--   |   --.--   | [link](https://drive.google.com/drive/folders/12dScaCjePvTyxvlWElGVTYv12UFcHN9U?usp=drive_link) |
| UF3V                         | MM-WLAuslan  |   82.03   |   94.58   | [link](https://drive.google.com/drive/folders/12dScaCjePvTyxvlWElGVTYv12UFcHN9U?usp=drive_link) |

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

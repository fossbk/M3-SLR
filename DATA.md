# Dataset Preparation

To run the code in this repository, the dataset directory should be organized as follows:

```plaintext
data/
├── MultiVSL200_videos/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...             # Other video files
├── MMAuslan_videos/
└── label/
    ├── Multi-VSL200/
    │   ├── labelCenter/
    │   │   ├── train_labels.csv
    │   │   ├── val_labels.csv
    │   │   └── test_labels.csv
    │   ├── labelLeft/
    │   ├── labelRight/
    │   └── labelThreeView/
    └── MMAuslan/
        ├── labelCenterSTU/
        │   ├── train_labels.csv
        │   ├── val_labels.csv
        │   └── test_labels.csv
        ├── labelLeftSTU/
        ├── labelRightSTU/
        ├── labelThreeViewSTU/
        └── ...             # other MMAuslan label types/files
```
## Multi-VSL

The Multi-VSL dataset can be downloaded from [here](https://drive.google.com/drive/folders/1yUU1m2hy_CjaXDDoR_6i9Y3T1XL2pD4C).

The lookup table (CSV file) mapping glosses to label IDs and their meanings is available [here](data/MultiVSL200).

## MM-WLAuslan

The MM-WLAuslan dataset and labels are available [here](https://uq-cvlab.github.io/MM-WLAuslan-Dataset/docs/en/dataset-download).

For use with this repository, the labels must be in a CSV file.
*   For the **single-view model**, the file should have two columns: `video path` and `gloss label ID`.
*   For the **multi-view model**, it should have four columns: `frontal video path`, `left video path`, `right video path`, and `gloss label ID`.

MM-WLAuslan labels pre-formatted in this way are available [here](data/MMAuslan), or you can generate them using the script at `data/utils/labelMaker.py`.

# PE-YOLO Based on YOLO11

This repository contains the implementation and experiment results of PE-YOLO, an improved object detection model based on YOLO11. The project focuses on small-object detection in drone-view scenes, especially VisDrone and AITOD datasets.

## Project Overview

PE-YOLO is built on the Ultralytics YOLO11 framework and introduces lightweight feature enhancement modules for dense and small targets. The main changes are placed in the YOLO11 model configuration and network modules.

Main features:

- YOLO11-based detection framework
- PE-YOLO model configs for nano and large variants
- SADEConv feature extraction modules
- RDC_Head detection head
- VisDrone and AITOD experiment results
- Training, validation, testing, and visualization outputs

## Repository Structure

```text
.
|-- ultralytics/
|   |-- cfg/models/11/          # YOLO11 and PE-YOLO model configs
|   |-- cfg/datasets/           # Dataset YAML files
|   `-- nn/                     # Network modules and model definitions
|-- runs/detect/                # Training and evaluation results
|-- run_eval.py                 # Evaluation script
|-- run_vis.py                  # Visualization script
|-- yolo11-visualize_heatmap.py # Heatmap visualization script
|-- calc_map.py                 # mAP calculation helper
|-- AP_small.py                 # Small-object AP calculation helper
`-- environment.yml             # Conda environment file
```

## Key Model Files

```text
ultralytics/cfg/models/11/PE-YOLO(n).yaml
ultralytics/cfg/models/11/PE-YOLO(l).yaml
ultralytics/cfg/models/11/PE-YOLO.yaml
ultralytics/cfg/models/11/PE-YOLO_without_SADEConv.yaml
ultralytics/cfg/models/11/PE-YOLO-R3_Head.yaml
```

Related modules can be found under:

```text
ultralytics/nn/modules/
ultralytics/nn/tasks.py
```

## Environment

Create the environment from the provided file:

```bash
conda env create -f environment.yml
conda activate pe-yolo
```

Alternatively, install the project in editable mode:

```bash
pip install -e .
```

## Training

Example command for training PE-YOLO on VisDrone:

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/PE-YOLO(n).yaml \
  data=ultralytics/cfg/datasets/VisDrone.yaml \
  epochs=200 \
  imgsz=640 \
  batch=4 \
  device=0
```

## Validation

```bash
yolo detect val \
  model=runs/detect/PE-YOLO(n)-visdrone-200/weights/best.pt \
  data=ultralytics/cfg/datasets/VisDrone.yaml \
  imgsz=640 \
  device=0
```

## Experiment Results

The main PE-YOLO(n) VisDrone training result is stored in:

```text
runs/detect/PE-YOLO(n)-visdrone-200/
```

Final epoch metrics:

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 0.49514 | 0.38815 | 0.40124 | 0.23888 |

Important output files:

```text
runs/detect/PE-YOLO(n)-visdrone-200/results.csv
runs/detect/PE-YOLO(n)-visdrone-200/results.png
runs/detect/PE-YOLO(n)-visdrone-200/confusion_matrix.png
runs/detect/PE-YOLO(n)-visdrone-200/weights/best.pt
runs/detect/PE-YOLO(n)-visdrone-200/weights/last.pt
```

## Visualization

Run visualization scripts:

```bash
python run_vis.py
python yolo11-visualize_heatmap.py
```

## Notes

This repository includes source code, model configuration files, trained weights, and experiment outputs. Some result files are large because they contain complete training artifacts and prediction outputs.

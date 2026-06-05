# Differences in Detection: Explainability Where it Matters

### [CVPR 2026 Workshop: HOW - How Do Vision Models Work?](https://sites.google.com/view/how-cvpr-workshop)

Paper: [CvF Open Access Link](https://openaccess.thecvf.com/content/CVPR2026W/HOW/papers/Theodoridis_Differences_in_Detection_Explainability_Where_it_Matters_CVPRW_2026_paper.pdf)

<img src="./imgs/dvd_overview.svg">

### Install Dependencies

```python
# optional
conda create --name dnd python=3.12
conda activate dnd

# dependencies
pip install jupyter, numpy, pycocotools, tidecv, tqdm, pywaffle, scikit-learn
```

### How to use it (same as `demo.ipynb`):

> Note: The prediction files must be in the coco results format, as consumed by pycocotools or tide. For example, MMDetection and Detectron 2 can both save the results in this format.

Prepare annotations and select predictions:

```python
from differences_in_detection.utils import download_coco_annotations
download_coco_annotations()

annotation_file = "./annotations/instances_val2017.json"

# Example 1
x_label = 'Model'
m1_name = "ConvNeXt-V2"
m1_pred = "./predictions/mask_rcnn_convnext_v2_b/ms-coco_results.bbox.json"

m2_name = "ViTDet"
m2_pred = "./predictions/mask_rcnn_vitdet_b/ms-coco_results.bbox.json"
```

Compare Differences in Detection:

```python
from differences_in_detection import get_dnd_results, plot_differences, plot_dnd_details

dnd_results = get_dnd_results(ann_file=annotation_file, results_file_1=m1_pred, results_file_2=m2_pred, iou_type='bbox')
plot_differences(dnd_results, m1_name, m2_name)
plot_dnd_details(dnd_results, m1_name, m2_name)
```

Compare COCO API (mAP):
```python
from differences_in_detection import get_cocoapi_results, plot_cocoapi_results

m1_coco_results = get_cocoapi_results(ann_file=annotation_file, results_file=m1_pred, iou_type='bbox')
m2_coco_results = get_cocoapi_results(ann_file=annotation_file, results_file=m2_pred, iou_type='bbox')

plot_cocoapi_results(m1_coco_results, m2_coco_results, m1_name, m2_name, x_label=x_label)
```

Compare TIDE Error Analysis:
```python
from differences_in_detection import get_tide_results, plot_tide_results

iou = 0.5
m1_tide_results = get_tide_results(ann_file=annotation_file, results_file=m1_pred, iou_type='bbox', threshold=iou)
m2_tide_results = get_tide_results(ann_file=annotation_file, results_file=m2_pred, iou_type='bbox', threshold=iou)
plot_tide_results(m1_tide_results, m2_tide_results, m1_name, m2_name, title_iou=iou)
```

### Citation

```bibtex
@InProceedings{Theodoridis_2026_CVPR,
    author    = {Theodoridis, Johannes and Maucher, Johannes and Schilling, Andreas},
    title     = {Differences in Detection: Explainability Where it Matters},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {4188-4192}
}
```

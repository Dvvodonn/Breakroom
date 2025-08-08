_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
from mmdet.datasets import CocoDataset
# Provide COCO class names so init_detector can populate model.dataset_meta
# MMDet 3.x expects a dataset "type" when building the registry entry.
# Use lazy_init=True so it doesn't try to load annotations.
test_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(
            classes=CocoDataset.METAINFO['classes']
        ),
        lazy_init=True,
    )
)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs')  # packs meta like img_id, shapes, etc.
]

# (Optional) Keep val consistent too, avoids warnings if used later
test_dataloader['dataset']['pipeline'] = test_pipeline
val_dataloader = test_dataloader
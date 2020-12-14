_base_ = [
    '../../_base_/models/memory_r50-d8.py', '../../_base_/datasets/camvid_video.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
model = dict(
    decode_head=dict(num_classes=12), auxiliary_head=dict(num_classes=12)
)
test_cfg = dict(mode='slide', crop_size=(640, 640), stride=(512, 512))


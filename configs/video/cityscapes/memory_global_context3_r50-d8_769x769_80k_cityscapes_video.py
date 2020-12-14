_base_ = [
    '../../_base_/models/memory_global_context_r50-d8.py', '../../_base_/datasets/cityscapes_video_769x769.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(type='MemoryGlobalContextHead3', align_corners=True, sequence_num=2,
                     key_channels=512, value_channels=512),
    auxiliary_head=dict(align_corners=True))
test_cfg = dict(mode='slide', crop_size=(769, 769), stride=(513, 513))

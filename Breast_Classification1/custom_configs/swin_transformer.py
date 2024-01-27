_base_ = [
    '../mmclassification/configs/_base_/models/swin_transformer/large_384.py',
    # '../mmclassification/configs/_base_/datasets/imagenet_bs64_swin_384.py',
    '../mmclassification/configs/_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../mmclassification/configs/_base_/default_runtime.py'
]

# ---- Model configs ----
# Here we use init_cfg to load pre-trained model.
# In this way, only the weights of backbone will be loaded.
# And modify the num_classes to match our dataset.

model = dict(
    backbone=dict(
        init_cfg = dict(
            type='Pretrained',
            checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_large_patch4_window12_384_22kto1k-0a40944b.pth',
            prefix='backbone')
    ),
    head=dict(
        num_classes=2,
        topk = (1, )
    ))

# ---- Dataset configs ----
# We re-organized the dataset as ImageNet format.
dataset_type = 'CustomDataset'


data_preprocessor = dict(
    num_classes=2,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=384, backend='pillow',interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),

]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=384, backend='pillow', interpolation='bicubic'),
    # dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

    
    # Specify the training dataset type and path
train_dataloader=dict(
    batch_size=4,
    num_workers=1,
    dataset =dict(
        type=dataset_type,
        data_root='data/breast/patches_split_data',
        data_prefix='train',
        # classes='../data/cats_dogs_dataset/classes.txt'),
        pipeline=train_pipeline
        ),
    )
    # Specify the validation dataset type and path
val_dataloader = dict(
    batch_size=4,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root='data/breast/patches_split_data',
        data_prefix='test',
#         ann_file='test.txt',
#         classes='../data/cats_dogs_dataset/classes.txt'),
        pipeline=test_pipeline
        ),
    )
test_dataloader=dict(
    batch_size=4,
    num_workers=1,
    dataset = dict(
        type=dataset_type,
        data_root='data/breast/patch_split_data',
        data_prefix='test',
#         ann_file='test.txt',
#         classes='../data/cats_dogs_dataset/classes.txt'),
        pipeline=test_pipeline
        ),
    )

# val_dataloader = None
val_cfg = dict()
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval = 1,val_begin=2)
# Specify evaluation metric
evaluation =  [
  dict(type='Accuracy', topk=(1,)),
  dict(type='SingleLabelMetric', items=['f1-score']),
]
# evaluation = dict(type='f1_score')
# ---- Schedule configs ----
# Usually in fine-tuning, we need a smaller learning rate and less training epochs.
# Specify the learning rate
    
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.015, weight_decay=0.3),
    paramwise_cfg = dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
    clip_grad=dict(max_norm=5.0),
)
# Set the learning rate scheduler
lr_config = dict(policy='step', step=1, gamma=0.1)
runner = dict(type='EpochBasedRunner')

# ---- Runtime configs ----
# Output training log every 10 iterations.
log_config = dict(interval=10)

train_dataloader = train_dataloader
# test_dataloader = test_dataloader
# val_dataloader = test_dataloader

test_evaluator = evaluation
val_evaluator = evaluation

python3 train.py -a resnext101 -d medicalSet --epochs 100 --schedule 81 122 \
    --gamma 0.1 --train-batch 16 --lr 0.001


# python3 train.py -a resnet -d cifar10 --epochs 100 --schedule 81 122 \
#     --gamma 0.1 --train-batch 128
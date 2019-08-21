#https://github.com/pytorch/examples/tree/master/imagenet
#https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
#https://github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py
python -m torch.distributed.launch --nproc_per_node=2 normal.py --name=efficientnet0 --dataset=imagenet --gpus=0,1 --batch_size=64 

'''
@Description: Normal Module of Pytorch, In this module we use the gpu dataloader \
    to accelarate and improve the usage of cpu
    tensorboard usage: tensorboard --logdir=xx
@Author: xieydd
@Date: 2019-08-14 09:54:49
@LastEditTime: 2019-08-21 11:19:29
@LastEditors: Please set LastEditors
'''
import torch
from config import NormalConfig
import utils
import numpy as np
import torch.nn as nn
from cifar_dataloader import get_cifar_iter_dali
from imagenet_dataloader import get_imagenet_iter_dali
from tensorboardX import SummaryWriter
from models import EfficientNet
import PIL
from tqdm import tqdm
import math
import torch.nn.functional as F
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
from torch.autograd import Variable
import torch.distributed as dist


config = NormalConfig()
device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

if config.fp16_allreduce or config.distributed:
    try:
        from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

def main():
    start = time.time()
    logger.info("Logger is set - training start,  Let`s get ready to the rumble!!! ")
    global best_acc1
    early_stopping = utils.EarlyStopping(patience=config.patience, verbose=True)

    if config.multiprocessing_distributed:
        
        if config.dataset.lower() != "imagenet":
            raise NameError("Not Imagenet dataset, if you really need mutil node, change the code for own dataset") 
            os._exit(0)
        import horovod.torch as hvd
        # horovod set
        hvd.init()

        # set default gpu device id
        torch.cuda.set_device(hvd.local_rank())
        
        # set seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        torch.backends.cudnn.benchmark = True

        # If set > 0, will resume training from a given checkpoint.
        resume_from_epoch = 0
        for try_epoch in range(config.epochs, 0, -1):
            if os.path.exists(config.checkpoint_format.format(epoch=try_epoch)):
                resume_from_epoch = try_epoch
                break

        # Horovod: broadcast resume_from_epoch from rank 0 (which will have
        # checkpoints) to other ranks.
        resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                        name='resume_from_epoch').item()

        # Horovod: print logs on the first worker.
        verbose = 1 if hvd.rank() == 0 else 0

        # Horovod: write TensorBoard logs on first worker.
        hvd_writer = SummaryWriter(log_dir=os.path.join(config.path, "tb_hvd")) if hvd.rank() == 0 else None

        kwargs = {'num_workers': config.workers, 'pin_memory': True} if len(config.gpus)>0 else {}    

        train_dataset = \
            datasets.ImageFolder(config.data_path+config.dataset+"/train",
                                transform=transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                ]))
        
        # Horovod: use DistributedSampler to partition data among workers. Manually specify
        # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_loader, num_replicas=hvd.size(), rank=hvd.rank())
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.allreduce_batch_size,
            sampler=train_sampler, **kwargs)
        if 'efficientnet' in config.arch:
            image_size = EfficientNet.EfficientNet.get_image_size(config.arch)
            val_dataset = \
                datasets.ImageFolder(config.data_path+config.dataset+"/val",
                                    transform=transforms.Compose([
                                        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                    ]))
            print('Using image size', image_size) 
        else:
            val_dataset = \
                datasets.ImageFolder(config.data_path+config.dataset+"/val",
                                    transform=transforms.Compose([
                                        transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                    ])) 
        print('Using image size', 224)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size,
                                         sampler=val_sampler, **kwargs)
        model = EfficientNet.EfficientNet.from_name(config.arch)
        # model = EfficientNet.from_pretrained('efficientnet-b0') # online
        model = model.to(device)

        # Horovod: scale learning rate by the number of GPUs.
        # Gradient Accumulation: scale learning rate by batches_per_allreduce
        optimizer = torch.optim.SGD(model.parameters(),
                            lr=(config.lr *
                                config.batches_per_allreduce * hvd.size()),
                            momentum=config.momentum, weight_decay=config.weight_decay)
        
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if config.fp16_allreduce else hvd.Compression.none
        
        # Horovod: wrap optimizer with DistributedOptimizer.
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters(),
            compression=compression,
            backward_passes_per_step=config.batches_per_allreduce)
        
        # Restore from a previous checkpoint, if initial_epoch is specified.
        # Horovod: restore on the first worker which will broadcast weights to other workers.
        if resume_from_epoch > 0 and hvd.rank() == 0:
            filepath = config.checkpoint_format.format(epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        criterion = nn.CrossEntropyLoss().cuda(device)

        # optionally resume from a checkpoint
        if config.resume:
            if os.path.isfile(config.resume):
                print("=> loading checkpoint '{}'".format(config.resume))
                checkpoint = torch.load(config.resume)
                config.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if config.gpus is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(config.gpus)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(config.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(config.resume))

        if config.evaluate:
            validate_hvd(valid_loader, model, 0, hvd_writer, verbose, early_stopping, hvd, start)
        
        for epoch in range(config.start_epoch, config.epochs):
            train_sampler.set_epoch(epoch)

            # train for one epoch
            train_hvd(train_loader, model, optimizer, epoch, config, hvd_writer, verbose, hvd)

            # evaluate on validation set
            validate_hvd(valid_loader, model, epoch, hvd_writer, verbose,early_stopping, hvd, start)

        utils.time(time.time() - start)
        logger.info("Final best Prec@1 = {:.4%}".format(best_acc1))
            
            
    else: 
        # set default gpu device id
        if config.fp16_allreduce or config.distributed:
            torch.cuda.set_device(config.local_rank % len(config.gpus))
            torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
            config.world_size = torch.distributed.get_world_size()
        else:
            torch.cuda.set_device(config.gpus[0])
            config.world_size = 1
        config.total_batch = config.world_size * config.batch_size
        # set seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        torch.backends.cudnn.benchmark = True

        if config.static_loss_scale != 1.0:
            if not config.fp16_allreduce:
                print("Warning:  if --fp16_allreduce is not used, static_loss_scale will be ignored.")

        if config.dataset.lower() == "imagenet":
            train_loader = get_imagenet_iter_dali(
                type='train',
                # image_dir="/googol/atlas/public/cv/ILSVRC/Data/"
                # use soft link `mkdir ./data/imagenet && ln -s /googol/atlas/public/cv/ILSVRC/Data/CLS-LOC/* ./data/imagenet/` 
                image_dir=config.data_path+config.dataset.lower(),
                batch_size = config.batch_size,
                num_threads = config.workers,
                world_size= config.world_size,
                local_rank= config.local_rank,
                crop=224, device_id=config.local_rank, num_gpus=config.gpus,
                dali_cpu=config.dali_cpu
            )
            if 'efficientnet' in config.arch:
                image_size = EfficientNet.EfficientNet.get_image_size(config.arch)
                val_transforms = transforms.Compose([
                    transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ])
                print('Using image size', image_size)
                valid_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(config.data_path+config.dataset.lower()+"/val", val_transforms),
                    batch_size=config.batch_size, shuffle=False,
                    num_workers=config.workers, pin_memory=True)
            else:
                valid_loader = get_imagenet_iter_dali(
                    type='val',
                    image_dir=config.data_path+config.dataset.lower(),
                    batch_size = config.batch_size,
                    num_threads = config.workers,
                    world_size= config.world_size,
                    local_rank= config.local_rank,
                    crop=224, device_id=config.local_rank, num_gpus=config.gpus
                )
        elif config.dataset.lower() == "cifar10":
            if 'efficientnet' in config.arch:
                raise NameError("don`t use cifar10 train efficientnet")
                os._exit(0)
            train_loader = get_cifar_iter_dali(
                type='train',
                image_dir=config.data_path+config.dataset.lower(),
                batch_size = config.batch_size,
                num_threads = config.workers,
                world_size=config.world_size,
                local_rank= config.local_rank
            )
            valid_loader = get_cifar_iter_dali(
                type='val',
                image_dir=config.data_path+config.dataset.lower(),
                batch_size = config.batch_size,
                num_threads = config.workers,
                world_size=config.world_size,
                local_rank= config.local_rank
            )
        else:
            raise NameError("No Support dataset config")
        
        
        '''
        @description: we need define model here!
        '''  
        model = EfficientNet.EfficientNet.from_name(config.arch)
        # model = EfficientNet.from_pretrained('efficientnet-b0') # online
        model = model.to(device)

        if config.fp16_allreduce:
            model = network_to_half(model)
        if config.distributed:
            # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
            # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
            model = DDP(model, delay_allreduce=True)
        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda(config.gpus)
        optimizer = torch.optim.SGD(model.parameters(), config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
        if config.fp16_allreduce:
            optimizer = FP16_Optimizer(optimizer,
                                    static_loss_scale=config.static_loss_scale,
                                    dynamic_loss_scale=config.dynamic_loss_scale)

        # optionally resume from a checkpoint
        if config.resume:
            if os.path.isfile(config.resume):
                print("=> loading checkpoint '{}'".format(config.resume))
                checkpoint = torch.load(config.resume)
                config.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if config.gpus is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(config.gpus)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(config.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(config.resume))
        
        if config.evaluate:
            res = validate(valid_loader, model, 0, criterion, config, early_stopping,writer, start)
            with open('res.txt', 'w') as f:
                print(res, file=f)
            return
        
        for epoch in range(config.start_epoch, config.epochs):

            # train for one epoch
            train(train_loader, model, criterion,optimizer, epoch, config, writer)

            # evaluate on validation set
            best_acc1 = validate(valid_loader, model, epoch, criterion, config, early_stopping, writer, start)

            # remember best acc@1 and save checkpoint
            #is_best = acc1 > best_acc1
            #best_acc1 = max(acc1, best_acc1)
            #utils.save_checkpoint(model, config.path, is_best)

            train_loader.reset()
            if 'efficientnet' not in config.arch:
                valid_loader.reset()
        utils.time(time.time() - start)
        logger.info("Final best Prec@1 = {:.4%} use {} s".format(best_acc1, ))

def train(train_loader, model, criterion, optimizer, epoch, config, writer):
    utils.adjust_learning_rate(optimizer, epoch, config)
    batch_time = utils.AverageMeters('Time', ':6.3f')
    data_time = utils.AverageMeters('Data', ':6.3f')
    losses = utils.AverageMeters('Loss', ':.4e')
    top1 = utils.AverageMeters('Acc@1', ':6.2f')
    top5 = utils.AverageMeters('Acc@5', ':6.2f')
    if 'DALIClassificationIterator' in train_loader.__class__.__name__:
        # TODO: IF need * config.world_size
        progress = utils.ProgressMeter(math.ceil(train_loader._size / config.batch_size), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
        cur_step = epoch*math.ceil(train_loader._size/ config.batch_size)
    else:
        progress = utils.ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
                            
        cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', config.lr, cur_step)

    model.train()

    end = time.time()
    if 'DALIClassificationIterator' in train_loader.__class__.__name__:
        for i, data in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            images = Variable(data[0]['data'])
            target = Variable(data[0]['label'].squeeze().cuda().long())

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if config.distributed:
                reduced_loss = reduce_tensor(loss.data, world_size=config.world_size)
                acc1 = reduce_tensor(acc1,world_size=config.world_size)
                acc5 = reduce_tensor(acc5, world_size=config.world_size)
            else:
                reduced_loss = loss.data
            losses.update(to_python_float(reduced_loss), images.size(0))
            top1.update(to_python_float(acc1), images.size(0))
            top5.update(to_python_float(acc5), images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            if config.fp16_allreduce:
                optimizer.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.print_freq == 0:
                progress.print(i)
            writer.add_scalar('train/loss', loss.item(), cur_step)
            writer.add_scalar('train/top1', top1.avg, cur_step)
            writer.add_scalar('train/top5', top5.avg, cur_step)
    else:
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            images = images.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.print_freq == 0:
                progress.print(i)
        
            writer.add_scalar('train/loss', loss.item(), cur_step)
            writer.add_scalar('train/top1', top1.avg, cur_step)
            writer.add_scalar('train/top5', top5.avg, cur_step)


def train_hvd(train_loader, model, optimizer, epoch, config, writer, verbose, hvd):
    train_loss = utils.Metric('train_loss', hvd)
    train_top1 = utils.Metric('train_top1',hvd)
    train_top5 = utils.Metric('train_top1',hvd)

    model.train()

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            utils.adjust_learning_rate_hvd(epoch, batch_idx, config, train_loader, hvd.size(), optimizer)

            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), config.batch_size):
                data_batch = data[i:i + config.batch_size]
                target_batch = target[i:i + config.batch_size]
                output = model(data_batch)
                prec1, prec5 = utils.accuracy(output, target_batch, topk=(1, 5))
                train_top1.update(prec1)
                train_top5.update(prec5)
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / config.batch_size))
                loss.backward()
            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({'loss': train_loss.avg.item(),
                               'top1': 100. * train_top1.avg.item(),
                               'top5': 100. * train_top5.avg.item()})
            t.update(1)

        writer.add_scalar('train/loss', train_loss.avg, epoch)
        writer.add_scalar('train/top1', train_top1.avg, epoch)
        writer.add_scalar('train/top5', train_top5.avg, epoch)

def validate(val_loader, model, epoch, criterion, config, early_stopping,writer,start):
    batch_time = utils.AverageMeters('Time', ':6.3f')
    losses = utils.AverageMeters('Loss', ':.4e')
    top1 = utils.AverageMeters('Acc@1', ':6.2f')
    top5 = utils.AverageMeters('Acc@5', ':6.2f')
    if 'DALIClassificationIterator' in val_loader.__class__.__name__: 
        progress = utils.ProgressMeter(math.ceil(val_loader._size/config.batch_size), batch_time, losses, top1, top5,
                             prefix='Test: ')
    else:
        progress = utils.ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ') 
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        if 'DALIClassificationIterator' in val_loader.__class__.__name__:
            for i, data in enumerate(val_loader):
                images = Variable(data[0]['data'])
                target = Variable(data[0]['label'].squeeze().cuda().long().cuda(non_blocking=True))

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                if config.distributed:
                    reduced_loss = reduce_tensor(loss.data, world_size=config.world_size)
                    acc1 = reduce_tensor(acc1, world_size=config.world_size)
                    acc5 = reduce_tensor(acc5, world_size=config.world_size)
                else:
                    reduced_loss = loss.data
                losses.update(to_python_float(reduced_loss), images.size(0))
                top1.update(to_python_float(acc1), images.size(0))
                top5.update(to_python_float(acc5), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.print_freq == 0:
                    progress.print(i)
        else:
            for i, (images, target) in enumerate(val_loader):
                images = images.cuda(device,non_blocking=True)
                target = target.cuda(device,non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.print_freq == 0:
                    progress.print(i)
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        
        early_stopping(losses.avg, model,ckpt_dir=config.path)
        if early_stopping.early_stop:
            print("Early stopping")
            utils.time(time.time() - start)
            os._exit(0)
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/top1', top1.val, epoch)
        writer.add_scalar('val/top5', top5.val, epoch)
    return top1.avg

def validate_hvd(val_loader, model, epoch, writer, verbose, early_stopping,hvd, start):
    model.eval()
    val_loss = utils.Metric('val_loss', hvd)
    val_top1 = utils.Metric('val_top1',hvd)
    val_top5 = utils.Metric('val_top5',hvd)

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
                val_top1.update(prec1)
                val_top5.update(prec5) 
                t.set_postfix({'loss': val_loss.avg.item(),
                               'top1': 100. * val_top1.avg.item(),
                               'top5': 100. * val_top5.avg.item()})
                t.update(1)

        early_stopping(val_loss.avg.item(), model,ckpt_dir=config.path) 
        if early_stopping.early_stop:
            print("Early stopping")
            utils.time(time.time() - start)
            os._exit(0)
        writer.add_scalar('val/loss', val_loss.avg, epoch)
        writer.add_scalar('val/top1', val_top1.avg, epoch)
        writer.add_scalar('val/top5', val_top5.avg, epoch)
    return  val_top1.avg  

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

if __name__ == "__main__":
    main()

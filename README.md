# Toolbox ZJU成像工程实验室

## 主要内容
1. Toolbox功能介绍
2. 环境依赖
3. 模型种类
4. 使用方法

## Toolbox功能介绍
- 支持多种low-level任务和主流图像复原网络。如denoise/super resolution/deblur/derain等任务，mimo-unet/restormer等图像复原网络。可以根据自身需求添加任务或者网络结构
- 实验管理方便。每一个实验下，均保存本次实验的原始配置json文件、训练日志文件、tensorboard的event文件、以及验证阶段都有对应checkpoint、可视化图像存储

## 环境依赖
- Python
- Pytorch
- scikit-image
- opencv-python
- Tensorboard
- scipy
- einops
- pdb
- matplotlib

## 模型种类
目前可供训练的模型如下：
- UNet
- RRDBNet
- MIMO-UNet / MIMO-UNet+
- MPRNet
- NAFNet
- Restormer
- Stripformer
- Uformer
- VapSR

## 使用方法
### 训练方法
    # 在项目根目录下直接运行训练脚本
    $ cd toolbox
    $ python main_train_sample.py 
注: 以上命令仅支持单卡训练。若想要多卡训练，请参考[BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/TrainTest_CN.md)的做法
### 项目架构
以下是项目根目录下主要功能介绍，主要修改options内的配置文件即可。\
若要修改数据预处理、网络结构、loss函数等，参照下述说明即可。

    toolbox 
      |-- main_train_sample.py # 训练代码入口
      |-- data # 数据集定义及预处理逻辑
      |-- logs # tensorboard可视化文件存储
      |-- models # 网络结构定义及选择
      |-- options # 训练配置json文件
      |-- results # 存储各次实验，以实验task命名
      |-- trainsets/testsets # 数据集，也可用软链接
      |-- utils # 一些功能的类

### JSON文件主要参数解读
    "task"：实验名称，建议是网络结构名称+一些重要参数+日期/编号，如rrdb_batchsize64_20230507
    "models"：模型的优化方式，和模型结构区分，loss不一样，如plain只支持pixel loss
    "gpu_ids"：单卡/多卡训练中，所使用的gpu编号，如4卡服务器为0 1 2 3
    "n_channels"：数据集读入时的通道数，一般为3
    "path/root"：任务名称，如results/superresolution。例，用rrdb做超分辨，那么实验结果可以在results/superresolution/rrdb_batchsize64_20230507目录下找到 
    "datasets"
        "dataset_type"：数据集类型，可以自己定义paired数据或者not paired数据等，默认plain为成对数据集
        "dataroot_H"：数据集路径
        "H_size"：Ground Truth的patch size
        "dataloader_num_workers"：每个GPU上的线程数，一般不要太大，2-8之间为宜
        "dataloader_batch_size"：每个GPU上的batch_size
    "netG"
        "net_type": 网络种类，目前支持rrdb rrdbnet unet mimounet mimounetplus mprnet nafnet restormer stripformer uformer vapsr
        "in_nc"：输入通道数
        注：其余参数可根据具体的网络结构进行定义
    "train"
        "checkpoint_test": 每多少iteration验证一次
        "checkpoint_save": 每多少iteration存储一次checkpoint
        "checkpoint_print": 每多少iteration打印一次训练情况
        注：训练总的iteration数目，需要去train_main_sample.py内line 160手动修改！

### 一些推荐使用习惯
- 如果实验报错了，debug时，最好把实验名改个名称，尽量不要重名（除非从断点开始训练）
- 待补充
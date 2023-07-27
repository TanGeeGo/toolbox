# Toolbox for ImagingLab@ZJU 

## 主要内容
0. 更新内容（New Features/Updates）
1. Toolbox功能介绍 （Introduction）
2. 环境依赖（Package dependencies）
3. 模型种类（Model Categories）
4. 使用方法（How To Use）

🚩 **更新内容（New Features/Updates）**

- ✅ July 26, 2023. 增加多卡训练，加入mimounet，restormer等复原模型的官方部署配置

## Toolbox功能介绍（Introduction）
- 支持多种low-level任务和主流图像复原网络。如denoise/super resolution/deblur/derain等任务，mimo-unet/restormer等图像复原网络，可以根据自身需求添加任务或者网络结构
- 主流复原模型均配备了官方部署配置（option_official_implementation_xxx.json），无需重构代码即可快速实验
- 实验管理方便。每一个实验下，均保存本次实验的原始配置json文件、训练日志文件、tensorboard的event文件、以及验证阶段都有对应checkpoint、可视化图像存储

## 环境依赖（Package dependencies）
Toolbox是在PyTorch 2.0.1+cu118, Python3.9.6, CUDA12.2的虚拟环境中测试的，（PyTorch 1.13.1+cu118, Python3.9.6, CUDA12.2 的环境也可以使用，不过分布式训练的命令会有些差异），下载需要的包可以通过以下命令：

<!-- The project is built with PyTorch 2.0.1+cu118, Python3.9.6, CUDA12.2 (PyTorch 1.13.1+cu118, Python3.9.6, CUDA12.2 also valid, besides the slight differences in DDP). For package dependencies, you can install them by: -->
```bash
pip install -r requirements.txt
```

## 模型种类（Model Categories）
目前可供训练的模型如下：
- UNet
- RRDBNet
- MIMO-UNet / MIMO-UNet+ /MIMO-Unet-MFF
- MPRNet
- NAFNet
- Restormer
- Stripformer
- Uformer
- VapSR

## 使用方法（How To Use）
我们提供简单的demo来训练/测试/推理模型，以便快速启动。 这些demo/command无法涵盖所有情况，更多详细信息将在后续更新中提供。

*TODO*

### 数据准备
组织数据的方式可参照这篇论文[Painter](https://github.com/baaivision/Painter)，或者按照你自己喜欢的方式~

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
      |-- dataset # 数据集，也可用软链接
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
    ...

### 训练模型
    # 在项目根目录下直接运行训练脚本
    $ cd toolbox
    $ python main_train_sample.py --opt options/option_xxxxx.json --dist False (单卡训练)
    $ torchrun --nproc_per_node=${GPU_NUMs} main_train_sample.py --opt options/option_xxxxx.json --dist True (多卡训练，注意此时option文件中的gpu_ids必须为list，例如：[0, 1, 2, 3])

### 测试模型
    # 在项目根目录下直接运行训练脚本
    $ cd toolbox
    $ python main_test_sample.py --opt options/option_xxxxx.json --dist False (单卡测试)

#### 注意测试模型和训练模型使用的是同一个json文件哦~

### 一些推荐使用习惯
- 待补充

## 引用（Citations）
如果我们的工具帮助到了您，不妨给我们点个星并引用一下吧
下面是BibTex的形式，使用需要Latex的 `url` 包.

``` latex
@misc{toolbox@zjuimaging,
  author =       {Shiqi Chen and Zida Chen and Ziran Zhang and Wenguan Zhang and Peng Luo and Zhengyue Zhuge and Jinwen Zhou},
  title =        {toolbox@zjuimaging: Open Source Image Restoration Toolbox of ZJU Imaging Lab},
  howpublished = {\url{https://github.com/TanGeeGo/toolbox}},
  year =         {2023}
}
```

> Shiqi Chen, Zida Chen, Ziran Zhang, Wenguan Zhang, Peng Luo, Zhengyue Zhuge, Jinwen Zhou. toolbox@zjuimaging: Open Source Image Restoration Toolbox of ZJU Imaging Lab. <https://github.com/TanGeeGo/toolbox>, 2023.
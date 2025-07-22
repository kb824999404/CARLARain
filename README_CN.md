<div align="center">

# CARLARain
**A Rainy Autonomous Driving Simulator based on CARLA**

![](Docs/license.svg)

<div>
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">中文</a>
</div>

</div>

CARLARain是一个支持复杂光照环境雨景模拟的自动驾驶模拟器框架，基于[CARLA](https://github.com/carla-simulator/carla)实现了驾驶场景的 环境仿真、车辆仿真和行人仿真，并结合[HRIGNet](https://kb824999404.github.io/HRIG/)和[CRIGNet](https://doi.org/10.1007/978-981-97-5597-4_8)，引入了复杂光照环境下可控且逼真的雨景模拟。该框架可为自动驾驶视觉感知算法构建丰富的雨景仿真训练环境，涵盖多样的时间段和光照条件，满足自动驾驶场景下的语义分割、实例分割、深度估计和目标检测等多个任务的需求。


<div align="center">

![](Docs/CARLARain图.svg)


<table>
<tr>
<td style="border: none; padding: 5px;"><img src="Docs/CARLARain_Clean.gif" /></td>
<td style="border: none; padding: 5px;"><img src="Docs/CARLARain_Rainy.gif" /></td>
</tr>
</table>

</div>

## 目录结构

* `configs`：配置文件
* `CarRain`：CARLA客户端代码，用于获取CARLA模拟的背景RGB图、语义分割图、实例分割图、深度图和物体边界框
* `CRIGNet`：CRIGNet代码，用于生成低分辨率雨纹图像
* `RainControlNet`：RainControlNet代码，用于将低分辨率雨纹图像扩大为高分辨率雨纹图像
* `HRIGNet`：HRIGNet代码，根据背景RGB图像和雨纹图像生成雨景图像
* `data`：输出数据路径

## 使用方法

### 配置环境

* CARLA：[下载CARLA服务器](https://carla.readthedocs.io/en/latest/start_quickstart/#carla-installation)，新建`carla`conda环境并[配置CARLA client library](https://carla.readthedocs.io/en/latest/start_quickstart/#install-client-library)
* ControlNet：`cd RainControlNet && conda env create -f environment.yaml`
* HRIGNet：`cd HRIGNet && conda env create -f environment.yaml`

### 准备模型权重

* 所有模型权重文件可在此获得：[BaiduCloud](https://pan.baidu.com/s/1FXNk-y86rxXeUYwPoGWnpQ?pwd=i4zi ) (提取码：i4zi)
* CRIGNet模型权重：下载`CRIGNet/00000-crig_fastgan_g1w0.5_rainTrainL_256-FastGAN`并解压至`CRIGNet/log_raintrainl_256`
* RainControlNet模型权重：下载`RainControlNet/crig_hint256-128_out512_4xgrad_1e-4-2024-10-05-T07-21-55`并解压至`RainControlNet/logs`
* HRIGNet模型权重：下载`HRIGNet/2023-10-19T19-31-52_blender-gdm-rainlayer-hw512-f4-em3`和`2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3`并解压至`HRIGNet/logs`

### 准备yaml配置文件

* 参考`configs/seqTest.yaml`，具体内容解释见[Docs/config.md](Docs/config.md)

### 运行

#### 1. 获取背景图像和标签数据

* 运行CARLA server：
  * 在CARLA目录下执行`./CarlaUE4.sh -RenderOffScreen -quality-level=Epic`
  * 配置`CarRain/config.py`中的`serverIP`和`serverPort`

* 激活conda环境：`conda activate carla`
* 运行CARLA client，获取背景RGB图、语义分割图、实例分割图、深度图和物体边界框：`python CarRain/main.py -c configs/seqTest.yaml`
* 获取目标检测数据：`python CarRain/isToDetect.py -c configs/seqTest.yaml`
* 获取视频：`python CarRain/seqToVideo.py -c configs/seqTest.yaml`

#### 2. 获取雨纹图像

> 可选择用现有方法获取雨纹图像作为雨层遮罩，也可用此处提供的CRIGNet+RainControlNet生成雨纹图像

1. **现有方法：**

   * weather-particle-simulator：https://github.com/astra-vision/weather-particle-simulator
   * 下载[RainMask](https://pan.baidu.com/s/1FXNk-y86rxXeUYwPoGWnpQ?pwd=i4zi )，选择所需分辨率和强度的雨纹图像，将其置于`data/rain`中

2. **CRIGNet+RainControlNet：**

   * 用CRIGNet获取低分辨率雨纹图像：
     * `conda activate hrig`
     * `cd CRIGNet&&python gen_rain_mask.py -c ../configs/seqTest.yaml&&cd ..`

   * 用RainControlNet获取高分辨率雨纹图像：
     * `conda activate control`
     * `cd RainControlNet&&python upscale_rain_mask.py -c ../configs/seqTest.yaml&&cd ..`

#### 3. 获取雨景图像

* 用HRIGNet获取雨景图像：
  * `conda activate hrig`
  * `cd HRIGNet&&python predict_car.py -c ../configs/seqTownsCombineTest.yaml`

## 运行Demo网页

* `conda activate hrig`
* `pip install flask`
* `cd Website`
* `flask run`
* 访问`http://127.0.0.1:5088`

## CARLARain数据集

基于CARLARain，我们构建了一个自动驾驶雨景图像数据集，利用了CARLA提供的8个不同的内置场景，并将时间分别设定为白天、傍晚和夜晚三个时间段，以模拟不同光照环境下的驾驶场景。在车辆和行人仿真方面，在每个场景中随机放置了100个车辆，500个行人。其中，渲染图像的分辨率皆为2048×1024。
该数据集包括8个不同场景，每个场景包括3个时间段，每个时间段包括1000帧的样本，每个样本包括自动驾驶清晰场景RGB图像、语义分割图像、实例分割图像、深度图像、雨纹图像、雨景RGB图像和物体边界框数据。根据场景将该数据集划分为训练集和测试集，训练集包括7个场景，测试集包括1个场景。

数据集可在[CARLARain-Dataset](https://pan.baidu.com/s/1FXNk-y86rxXeUYwPoGWnpQ?pwd=i4zi)获得(提取码：i4zi)

<table>
<tr>
<th>数据集类型 </th>
<th>场景 </th>
<th>时间段 </th>
<th>帧数 </th>
<th>样本数 </th>
<th>图像类型 </th>
</tr>
<tr>
<td>训练集</td>
<td>7</td>
<td>3</td>
<td>1000</td>
<td>21000</td>
<td rowspan=2>场景RGB图像、语义分割图像、实例分割图像、深度图像、雨纹图像、 雨景RGB图像、物体边界框</td>
</tr>
<tr>
<td>测试集</td>
<td>1</td>
<td>3</td>
<td>1000</td>
<td>3000</td>
</tr>
</table>

<table>
<tr>
<th>背景 </th>
<th>雨景 </th>
<th>深度估计 </th>
<th>语义分割 </th>
<th>实例分割 </th>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown01ClearSunset_002423..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown01ClearSunset_002423.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown01ClearSunset_002423.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown01ClearSunset_002423.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown01ClearSunset_002423.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown01Clear_000044..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown01Clear_000044.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown01Clear_000044.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown01Clear_000044.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown01Clear_000044.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown02ClearNight_007652..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown02ClearNight_007652.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown02ClearNight_007652.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown02ClearNight_007652.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown02ClearNight_007652.png" /></td>
</tr>

<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown02Clear_001262..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown02Clear_001262.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown02Clear_001262.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown02Clear_001262.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown02Clear_001262.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown03ClearNight_009422..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown03ClearNight_009422.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown03ClearNight_009422.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown03ClearNight_009422.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown03ClearNight_009422.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown03ClearSunset_009200..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown03ClearSunset_009200.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown03ClearSunset_009200.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown03ClearSunset_009200.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown03ClearSunset_009200.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown03Clear_001814..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown03Clear_001814.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown03Clear_001814.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown03Clear_001814.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown03Clear_001814.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown04ClearNight_064471..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown04ClearNight_064471.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown04ClearNight_064471.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown04ClearNight_064471.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown04ClearNight_064471.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown04ClearSunset_010612..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown04ClearSunset_010612.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown04ClearSunset_010612.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown04ClearSunset_010612.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown04ClearSunset_010612.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown04Clear_002690..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown04Clear_002690.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown04Clear_002690.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown04Clear_002690.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown04Clear_002690.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown05ClearNight_013922..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown05ClearNight_013922.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown05ClearNight_013922.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown05ClearNight_013922.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown05ClearNight_013922.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown05ClearSunset_012791..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown05ClearSunset_012791.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown05ClearSunset_012791.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown05ClearSunset_012791.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown05ClearSunset_012791.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown05Clear_003271..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown05Clear_003271.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown05Clear_003271.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown05Clear_003271.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown05Clear_003271.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown06ClearNight_000267..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown06ClearNight_000267.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown06ClearNight_000267.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown06ClearNight_000267.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown06ClearNight_000267.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown06ClearSunset_001191..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown06ClearSunset_001191.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown06ClearSunset_001191.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown06ClearSunset_001191.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown06ClearSunset_001191.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown06Clear_002307..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown06Clear_002307.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown06Clear_002307.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown06Clear_002307.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown06Clear_002307.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown07ClearSunset_004693..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown07ClearSunset_004693.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown07ClearSunset_004693.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown07ClearSunset_004693.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown07ClearSunset_004693.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown07Clear_004556..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown07Clear_004556.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown07Clear_004556.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown07Clear_004556.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown07Clear_004556.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown10ClearNight_065810..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown10ClearNight_065810.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown10ClearNight_065810.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown10ClearNight_065810.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown10ClearNight_065810.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown10ClearSunset_006789..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown10ClearSunset_006789.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown10ClearSunset_006789.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown10ClearSunset_006789.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown10ClearSunset_006789.png" /></td>
</tr>
<tr>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/background/seqTown10Clear_005656..jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/rainy/seqTown10Clear_005656.jpg" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/depth/seqTown10Clear_005656.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/semantic_segmentation/seqTown10Clear_005656.png" /></td>
<td style="padding: 0;width=20%;"><img src="Docs/CARLARainDataset/instance_segmentation/seqTown10Clear_005656.png" /></td>
</tr>

</table>


## 许可证

CARLARain 代码在 MIT 许可证下分发。

## 参考

* CARLA：https://github.com/carla-simulator/carla
* HRIGNet：https://kb824999404.github.io/HRIG/
* CRIGNet：https://doi.org/10.1007/978-981-97-5597-4_8
* ControlNet：https://github.com/lllyasviel/ControlNet
* Rain Rendering：https://github.com/cv-rits/rain-rendering/

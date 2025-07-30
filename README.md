<div align="center">

# CARLARain
**An Extreme Rainy Street Scene Simulator based on CARLA**

![](Docs/license.svg)

<div>
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">中文</a>
</div>

</div>

This is the repositories of the CARLARain in our paper "Learning from Rendering: Realistic and Controllable Extreme Rainy Image Synthesis for Autonomous Driving Simulation".

CARLARain is an extreme rainy street scene simulator, which integrats our proposed [Learning-from-Rendering](https://kb824999404.github.io/HRIG/) rainy image synthesizer with the [CARLA](https://github.com/carla-simulator/carla) driving simulator and the [CRIGNet](https://doi.org/10.1007/978-981-97-5597-4_8). CARLARain can obtain paired rainy-clean images and labels under complex illumination conditions. CARLARain configures RGB camera, depth camera, semantic segmentation camera, instance segmentation camera, and collision detection sensor based on CARLA in the street scene simulation module. During simulation, CARLARain can obtain data in the street scene environment for each frame, including RGB images, semantic segmentation images, instance segmentation images, depth images, and object bounding boxes. 


<div align="center">

![](Docs/CARLARain图EN.svg)


<table>
<tr>
<td style="border: none; padding: 5px;"><img src="Docs/CARLARain_Clean.gif" /></td>
<td style="border: none; padding: 5px;"><img src="Docs/CARLARain_Rainy.gif" /></td>
</tr>
</table>

</div>

## File Structure

* `configs`：Configuration files
* `CarRain`：CARLA client code, used to obtain the background RGB images, semantic segmentation maps, instance segmentation maps, depth maps, and object bounding boxes of CARLA simulations
* `CRIGNet`：CRIGNet code, used to generate low-resolution rain streak images
* `RainControlNet`：RainControlNet code, used to expand low-resolution rain streak images into high-resolution rain streak images
* `HRIGNet`：HRIGNet code, used to generate rainy scene images based on background RGB images and rain streak images
* `data`：Output data path


## How to run

### Environment Setup

* CARLA: [Download the CARLA server](https://carla.readthedocs.io/en/latest/start_quickstart/#carla-installation), create a new `carla` conda environment and [configure the CARLA client library](https://carla.readthedocs.io/en/latest/start_quickstart/#install-client-library).
* ControlNet: `cd RainControlNet && conda env create -f environment.yaml`
* HRIGNet: `cd HRIGNet && conda env create -f environment.yaml`


### Prepare Model Weights

* All model weight files can be obtained here: [BaiduCloud](https://pan.baidu.com/s/1FXNk-y86rxXeUYwPoGWnpQ?pwd=i4zi ) (Extraction code: i4zi)
* CRIGNet model weights: Download `CRIGNet/00000-crig_fastgan_g1w0.5_rainTrainL_256-FastGAN` and extract it to `CRIGNet/log_raintrainl_256`
* RainControlNet model weights: Download `RainControlNet/crig_hint256-128_out512_4xgrad_1e-4-2024-10-05-T07-21-55` and extract it to `RainControlNet/logs`
* HRIGNet model weights: Download `HRIGNet/2023-10-19T19-31-52_blender-gdm-rainlayer-hw512-f4-em3` and `2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3` and extract them to `HRIGNet/logs`

### Prepare YAML Configuration Files

* Refer to `configs/seqTest.yaml`, and for detailed explanations of the content, see [Docs/config.md](./Docs/config.md#EN)

### Run

#### 1. Obtain Background Images and Label Data

* Run the CARLA server:
  * In the CARLA directory, execute `./CarlaUE4.sh -RenderOffScreen -quality-level=Epic`
  * Configure `serverIP` and `serverPort` in `CarRain/config.py`

* Activate the conda environment: `conda activate carla`
* Run the CARLA client to obtain background RGB images, semantic segmentation maps, instance segmentation maps, depth maps, and object bounding boxes: `python CarRain/main.py -c configs/seqTest.yaml`
* Obtain object detection data: `python CarRain/isToDetect.py -c configs/seqTest.yaml`
* Obtain videos: `python CarRain/seqToVideo.py -c configs/seqTest.yaml`

#### 2. Obtain Rain Streak Images

> You can choose to obtain rain streak images as rain layer masks using existing methods, or generate rain streak images using CRIGNet + RainControlNet provided here.

1. **Existing methods**:
   * weather-particle-simulator：https://github.com/astra-vision/weather-particle-simulator
   * Download [RainMask](https://pan.baidu.com/s/1FXNk-y86rxXeUYwPoGWnpQ?pwd=i4zi ), select the rain streak images with the desired resolution and intensity, and place them in`data/rain`


2. **CRIGNet + RainControlNet**:
   * Use CRIGNet to obtain low-resolution rain streak images:
     * `conda activate hrig`
     * `cd CRIGNet&&python gen_rain_mask.py -c../configs/seqTest.yaml&&cd..`

   * Use RainControlNet to obtain high-resolution rain streak images:
     * `conda activate control`
     * `cd RainControlNet&&python upscale_rain_mask.py -c../configs/seqTest.yaml&&cd..`

#### 3. Obtain Rainy Images

* Use HRIGNet to obtain rainy images:
  * `conda activate hrig`
  * `cd HRIGNet&&python predict_car.py -c../configs/seqTownsCombineTest.yaml`


## Run Demo Website

* `conda activate hrig`
* `pip install flask`
* `cd Website`
* `python app.py`
* Visit `http://127.0.0.1:5088`

## Learning-from-Rendering Rainy Image Synthesizer

To incorporate both controllability and realism into rainy image synthesis, we propose a Learning-from-Rendering rainy image synthesizer, which combines the benefits of realism and controllability. 

In the rendering stage, we propose a 3D rainy scene rendering pipeline to render realistic high-resolution paired rainy-clean images. In the learning stage, we train a **H**igh-resolution **R**ainy **I**mage **G**eneration Network (HRIGNet) to controllably generate extreme rainy images conditioned on clean images. HRIGNet is used for rainy image generation in the CARLARain, which allows CARLARain to produce paired extreme rainy-clean images and label data under complex illumination conditions.

* Get the codes for the rendering stage and the learning stage: [Github](https://github.com/kb824999404/HRIG)

![](Docs/Learning-from-Rendering.svg)

## High-resolution Rainy Image Dataset

In the rendering stage, we create a High-resolution Rainy Image (HRI) dataset in the rendering stage of the proposed rainy image synthesizer. The HRI dataset comprises a total of 3,200 image pairs. Each image pair comprises a clean background image, a depth image, a rain layer mask image, and a rainy image. It contains three scenes: lane, citystreet and japanesestreet, with image resolutions of 2048 $\times$ 1024. We split the HRI dataset into train set and test set according to camera viewpoints.

* Get the HRI dataset and the Blender scene files: [Hugging Face](https://huggingface.co/datasets/Ian824/High-Resolution-Rainy-Image), [Google Drive](https://drive.google.com/drive/folders/1MSS-iNaLxI05K_10pHMWYibrDJtMJngP?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/14G4fE8_7lswvod6OtIbOew?pwd=v9b2)(Extraction Code: v9b2)

* Some rainy images of the HRI dataset:
    * **Lane Scene:**
      <div class="hri-images-container">
        <img src="Docs/HRIDataset/lane/front_100mm_frame_1000.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/lane/front_10mm_frame_0900.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/lane/front_25mm_frame_0200.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/lane/front_50mm_frame_0800.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/lane/low_100mm_frame_0240.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/lane/low_10mm_frame_0900.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/lane/low_25mm_frame_0800.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/lane/low_50mm_frame_1000.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/lane/mid_100mm_frame_1000.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/lane/mid_10mm_frame_0900.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/lane/mid_25mm_frame_0680.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/lane/mid_50mm_frame_0850.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/lane/side_100mm_frame_1000.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/lane/side_10mm_frame_0900.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/lane/side_25mm_frame_0680.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/lane/side_50mm_frame_0850.jpg" style="width: 24%;margin-bottom: 5px;"/>
      </div>
  * **Citystreet Scene:**
      <div class="hri-images-container">
        <img src="Docs/HRIDataset/citystreet/back_10mm_frame_0030.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/citystreet/back_10mm_frame_0110.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/citystreet/far_10mm_frame_0030.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/citystreet/far_10mm_frame_0110.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/citystreet/front_10mm_frame_0010.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/citystreet/front_10mm_frame_0150.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/citystreet/sideinner_50mm_frame_0010.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/citystreet/sideinner_50mm_frame_0250.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/citystreet/sideleft_10mm_frame_0010.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/citystreet/sideleft_10mm_frame_0150.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/citystreet/sideright_10mm_frame_0010.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/citystreet/sideright_10mm_frame_0150.jpg" style="width: 24%;margin-bottom: 5px;"/>
      </div>
  * **Japanesestreet Scene:**
      <div class="hri-images-container">
        <img src="Docs/HRIDataset/japanesestreet/camera10_10mm_frame_0030.jpg" style="width: 24%;margin-bottom: 5px;"/>        
        <img src="Docs/HRIDataset/japanesestreet/camera10_10mm_frame_0090.jpg" style="width: 24%;margin-bottom: 5px;"/>        
        <img src="Docs/HRIDataset/japanesestreet/camera1_10mm_frame_0070.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera1_10mm_frame_0130.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera2_10mm_frame_0010.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera2_10mm_frame_0110.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera3_10mm_frame_0030.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera3_10mm_frame_0140.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera4_10mm_frame_0010.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera4_10mm_frame_0250.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera5_10mm_frame_0070.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera5_10mm_frame_0110.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera6_10mm_frame_0050.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera6_10mm_frame_0120.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera7_10mm_frame_0010.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera7_10mm_frame_0090.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera8_10mm_frame_0010.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera8_10mm_frame_0090.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera9_10mm_frame_0010.jpg" style="width: 24%;margin-bottom: 5px;"/>
        <img src="Docs/HRIDataset/japanesestreet/camera9_10mm_frame_0130.jpg" style="width: 24%;margin-bottom: 5px;"/>
      </div>

## ExtremeRain Dataset

Based on CARLARain, we construct an extreme rainy street scene image dataset, ExtremeRain. This dataset contains 8 different street scenes and 3 illumination conditions: daytime, sunset, night. The rainy scenes feature a rain intensity ranging from 5 mm/h - 100 mm/h, covering extreme rainfalls under complex illumination conditions. The dataset contains comprehensive label information to meet the requirements of multi-task visual perception models, including semantic segmentation, instance segmentation, depth estimation, and object detection. We split the dataset into train set and test set according to different scenes.

* Get the ExtremeRain dataset: [Baidu Cloud](https://pan.baidu.com/s/1FXNk-y86rxXeUYwPoGWnpQ?pwd=i4zi) (Extraction code: i4zi)

<table>
<tr>
<th>Dataset Type </th>
<th>Scene </th>
<th>Time </th>
<th>Frame </th>
<th>Sample Count </th>
<th>Image Type </th>
</tr>
<tr>
<td>Trainset</td>
<td>7</td>
<td>3</td>
<td>1000</td>
<td>21000</td>
<td rowspan=2> Scene RGB image, semantic segmentation image, instance segmentation image, depth image, rain streak image, rainy RGB image, object bounding box</td>
</tr>
<tr>
<td>Testset</td>
<td>1</td>
<td>3</td>
<td>1000</td>
<td>3000</td>
</tr>
</table>

<table>
<tr>
<th>Background </th>
<th>Rainy </th>
<th>Depth </th>
<th>Semantic Segmentation </th>
<th>Instance Segmentation </th>
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

## Experiment - Realism And Controllability Of Rainy Image Generation

* **Compare with baseline:** To evaluate the performance of HRIGNet in high-resolution rainy image generation, we compare it with several baseline image generative models: LDM, DiT and CycleGAN. The figure below illustrates a comparison of rainy image generation results of these methods.

<table>
<tr>
<td style="padding: 0;width=30%;"><img src="Docs/HRIG_Baseline/Background_draw2.jpg" /></td>
<td style="padding: 0;width=30%;"><img src="Docs/HRIG_Baseline/LDM_draw2.jpg" /></td>
<td style="padding: 0;width=30%;"><img src="Docs/HRIG_Baseline/DiT_draw2.jpg" /></td>
</tr>
<tr>
<td style="padding: 0;width=30%;text-align:center;">Background</td>
<td style="padding: 0;width=30%;text-align:center;">LDM</td>
<td style="padding: 0;width=30%;text-align:center;">DiT</td>
</tr>
<tr>
<td style="padding: 0;width=30%;"><img src="Docs/HRIG_Baseline/Rainy_draw2.jpg" /></td>
<td style="padding: 0;width=30%;"><img src="Docs/HRIG_Baseline/CycleGAN_draw2.jpg" /></td>
<td style="padding: 0;width=30%;"><img src="Docs/HRIG_Baseline/HRIG_draw2.jpg" /></td>
</tr>
<tr>
<td style="padding: 0;width=30%;text-align:center;">Ground truth</td>
<td style="padding: 0;width=30%;text-align:center;">CycleGAN</td>
<td style="padding: 0;width=30%;text-align:center;">HRIGNet(ours)</td>
</tr>
</table>

* **Controlibility:** As shown in the figure below, some rainy images from ExtremeRain are presented. It is possible to control different background scenes, achieve variations in illumination such as daytime, sunset, and night, and control attributes like rain intensity and direction. The controllability of multiple attributes ensures the diversity of the dataset.

![](Docs/RainControlibility.svg)

## Experiment - Semantic Segmentation In Extreme Rainfall

* To improve the accuracy of semantic segmentation models in extreme rainy scenes, we conduct augmented training with the ExtremeRain dataset and evaluate several SOTA semantic segmentation models on real datasets. We collect real rainy scene images with different illumination conditions from the Internet, and use them as the test set.

![](Docs/augmented_semantic_test_on_real.svg)

## License

The CARLARain code is distributed under the MIT License.

## Reference

* CARLA：https://github.com/carla-simulator/carla
* HRIGNet：https://kb824999404.github.io/HRIG/
* CRIGNet：https://doi.org/10.1007/978-981-97-5597-4_8
* ControlNet：https://github.com/lllyasviel/ControlNet
* Rain Rendering：https://github.com/cv-rits/rain-rendering/

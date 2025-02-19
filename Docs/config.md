# yaml config
<div>
  <a href="#CN">中文</a> |
  <a href="#EN">English</a>
</div>

<span id="CN"></span>

## 中文
```yaml
scene:  # CARLA场景配置
  name: seqTest   # 序列名
  resolution:     # 图像分辨率
    - 2048
    - 1024
  fps: 10         # CARLA模拟fps
  dataRoot: CARLARain/data   # 输出数据路径
  cameras:                   # 配置的传感器
    - rgb
    - depth
    - is
    - ss
  imgTypes:                  # 传感器对应保存的图像类型
    - jpg
    - png
    - png
    - png
  main_vehicle: vehicle.tesla.model3     # 相机附着车辆的蓝图
  vehicleCount: 100                      # 场景中车辆数量
  walkerCount: 500                       # 场景行人数量
  percentWalkerCrossing: 0.3             # 行人横穿马路的概率
  cameraPos:                             # 相机相对于附着车辆的位置
    - [ 0.8,0 ]
    - [ 0,0 ]
    - [ 1,0.15 ]
  useKeyboard: true                       # 是否监听键盘输入
  frameCount: 1000                        # 保存数据的总帧数
  map: Town01                             # CARLA使用的地图
  environment:                            # CARLA环境配置
    sun: day
    weather: rain
    # altitude: 45
    # azimuth: 180
    # clouds: 100.0
    # rain: 50.0
    # puddles: 50.0
    # wind: 50.0
    # fog: 5.0
    # fogdist: 10.0
    # fogfalloff: 1.0
    # wetness: 50.0
    # scatteringintensity: 5.0
    # rayleighscatteringscale: 1.0
    # miescatteringscale: 1.0
    # dust_storm: 10.0
    # cars: [ Fog ]
    # lights:
    # lightgroup:

crig: # CRIGNet配置
  netED: CARLARain/CRIGNet/log_raintrainl_256/00000-crig_fastgan_g1w0.5_rainTrainL_256-FastGAN/model/ED_state_500.pt  # 网络权重路径
  patchSize: 256    # 生成图像大小
  nc: 3             
  nz: 128           # 隐变量大小
  nef: 32
  nlabel: 2         # 标签变量大小
  backbone: FastGAN   # 使用的主干网络：[Resnet,Transformer,FastGAN]
  w_dim: 128
  use_mapping: false
  gpu: 4
  # 变量采样模式
  # mode: RANDOM_Z_FIXED_LABEL    # 固定标签变量，随机采样隐变量
  mode: RANDOM_LABEL_FIXED_Z      # 固定隐变量，随机采样标签变量
  # mode: RANDOM_Z_AND_LABEL      # 随机采样标签变量和隐变量
  label: [ 1,1 ]                  # RANDOM_Z_FIXED_LABEL模式下用的标签变量
  seed: 123
  sampleNum: 1000                 # 生成图像数量
  
controlnet: # RainControlNet配置
  model_config: models/cldm_v15.yaml    # 模型配置路径
  source_path: rain_crig_256            # 输入图像在序列数据路径中的文件夹
  resume: CARLARain/RainControlNet/logs/crig_hint256-128_out512_4xgrad_1e-4-2024-10-05-T07-21-55/version_0/epoch=417-step=19999.ckpt            # 模型权重路径
  batch_size: 2
  resolution: 512                       # 输出图像分辨率
  cfg: 7
  steps: 20                             # Diffusion采样步数
  gpu: 4                                # 使用的GPU编号

hrig:   # HRIGNet配置
  resume:  /CARLARain/HRIGNet/logs/2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3    # 模型权重目录
  ckpt: epoch=000098                        # 模型权重文件名
  steps: 20                                 # Diffusion采样步数
  overlap: 64                               # 采样窗口重叠部分大小
  gpu: 4                                    # 使用的GPU编号
  use_lighten: true                         # 是否使用lighten后处理
  data:                                     # DataLoader配置
    target: main.DataModuleFromConfig
    params:
      batch_size: 2
      num_workers: 0
      validation:
        target: hrig_blender.data.rainReal.RainCARLA
        params:
          data_root: seqTest
          background_path: background       # 输入的背景图像路径
          rain_path: rain                   # 输入的RainMask路径
          size: 512
          readImg: true

```

<span id="EN"></span>

## English
```yaml
scene:  # the config of CARLA scene
  name: seqTest   # sequence name
  resolution:     # the output image resolution
    - 2048
    - 1024
  fps: 10         # the fps of CARLA
  dataRoot: CARLARain/data   # the output data root
  cameras:                   # the type of camera of output datas
    - rgb
    - depth
    - is
    - ss
  imgTypes:                  # the saved image type of the cameras
    - jpg
    - png
    - png
    - png
  main_vehicle: vehicle.tesla.model3     # the blueprint of the main vehicle
  vehicleCount: 100                      # the count of vehicle in the scene
  walkerCount: 500                       # the count of walker in the scene
  percentWalkerCrossing: 0.3             # the percent of walker crossing the road
  cameraPos:                             # the relative position of the camera to the main vehicle
    - [ 0.8,0 ]
    - [ 0,0 ]
    - [ 1,0.15 ]
  useKeyboard: true                       # if listen to the keyboard input
  frameCount: 1000                        # the frame count of data to saved
  map: Town01                             # the map used in CARLA
  environment:                            # the environment config of CARLA
    sun: day
    weather: rain
    # altitude: 45
    # azimuth: 180
    # clouds: 100.0
    # rain: 50.0
    # puddles: 50.0
    # wind: 50.0
    # fog: 5.0
    # fogdist: 10.0
    # fogfalloff: 1.0
    # wetness: 50.0
    # scatteringintensity: 5.0
    # rayleighscatteringscale: 1.0
    # miescatteringscale: 1.0
    # dust_storm: 10.0
    # cars: [ Fog ]
    # lights:
    # lightgroup:

crig: # the configs of CRIGNet
  netED: CARLARain/CRIGNet/log_raintrainl_256/00000-crig_fastgan_g1w0.5_rainTrainL_256-FastGAN/model/ED_state_500.pt  # the path of the model weights
  patchSize: 256    # the size of generated images
  nc: 3             
  nz: 128           # the size of latent code
  nef: 32
  nlabel: 2         # the size of label
  backbone: FastGAN   # the backbone of the generator：[Resnet,Transformer,FastGAN]
  w_dim: 128
  use_mapping: false
  gpu: 4
  # the mode of data sampling
  # mode: RANDOM_Z_FIXED_LABEL    # fixed label，random sampling latnet code
  mode: RANDOM_LABEL_FIXED_Z      # fixed latnet code，random sampling label
  # mode: RANDOM_Z_AND_LABEL      # random sampling latnet code and label
  label: [ 1,1 ]                  # the label for the mode RANDOM_Z_FIXED_LABEL
  seed: 123
  sampleNum: 1000                 # the count of generated image
  
controlnet: # the config of RainControlNet
  model_config: models/cldm_v15.yaml    # the path of the model config
  source_path: rain_crig_256            # the directory of input images
  resume: CARLARain/RainControlNet/logs/crig_hint256-128_out512_4xgrad_1e-4-2024-10-05-T07-21-55/version_0/epoch=417-step=19999.ckpt            # the path of the model weights
  batch_size: 2
  resolution: 512                       # the size of generated images
  cfg: 7
  steps: 20                             # the sampling steps in diffusion model
  gpu: 4                                # the used gpu id

hrig:   # the config of HRIGNet
  resume:  /CARLARain/HRIGNet/logs/2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3    # the path of the model weights
  ckpt: epoch=000098                        # the name of the model weights
  steps: 20                                 # the sampling steps in diffusion model
  overlap: 64                               # the overlap size of the sampling window
  gpu: 4                                    # the used gpu id
  use_lighten: true                         # if use lighted post processing
  data:                                     # the config of the DataLoader
    target: main.DataModuleFromConfig
    params:
      batch_size: 2
      num_workers: 0
      validation:
        target: hrig_blender.data.rainReal.RainCARLA
        params:
          data_root: seqTest
          background_path: background       # the path of background images
          rain_path: rain                   # the path of input rain masks
          size: 512
          readImg: true

```
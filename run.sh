# Render Images with CARLA
python CarRain/main.py -c configs/seqTest.yaml
python CarRain/isToDetect.py -c configs/seqTest.yaml
python CarRain/seqToVideo.py -c configs/seqTest.yaml

# Generate Rain Masks
cd CRIGNet&&python gen_rain_mask.py -c ../configs/seqTest.yaml&&cd ..

# Upscale Rain Masks
cd RainControlNet&&python upscale_rain_mask.py -c ../configs/seqTest.yaml&&cd ..

# Generate Rainy Images
cd HRIGNet&&python predict_car.py -c ../configs/seqTest.yaml
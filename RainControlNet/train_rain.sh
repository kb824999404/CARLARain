python train_hrig.py --task controlnet_rain_4xgrad_1e-4 \
    --data_root /home/zhoukaibin/data7/dataset/BlenderRain --data_json trainset.json \
    --gpus "[4]" -bs 2 --max_steps 10000 -lr 1e-4 --ckpt_save_freq 1000 \
    --gradient_accumulate 4

python train_crig.py --task crig_hint64_out512_4xgrad_1e-4 \
    --data_root /home/zhoukaibin/data7/dataset/rain-mask-wind-single/rain_512/ --data_json dataset.json \
    --hint_size 64 \
    --gpus "[4]" -bs 2 --max_steps 10000 -lr 1e-4 --ckpt_save_freq 1000 \
    --gradient_accumulate 4

python train_crig.py --task crig_hint128_out512_4xgrad_1e-4 \
    --data_root /home/zhoukaibin/data7/dataset/rain-mask-wind-single/rain_512/ --data_json dataset.json \
    --hint_size 128 \
    --gpus "[4]" -bs 2 --max_steps 10000 -lr 1e-4 --ckpt_save_freq 1000 \
    --gradient_accumulate 4

python train_crig.py --task crig_hint128_withlabel_out512_4xgrad_1e-4 \
    --data_root /home/zhoukaibin/data7/dataset/rain-mask-wind-single/rain_512/ --data_json dataset.json \
    --hint_size 128 \
    --gpus "[4]" -bs 2 --max_steps 10000 -lr 1e-4 --ckpt_save_freq 1000 \
    --gradient_accumulate 4 \
    --model_config ./models/cldm_v15_rain.yaml
CUDA_VISIBLE_DEVICES=4 python eval_crig.py --task crig_hint256-128_out512_4xgrad \
    --data_root /home/zhoukaibin/data7/code/CARLRain/data/seqTest/rain_crig_128 \
    --resume logs/crig_hint256-128_out512_4xgrad_1e-4-2024-10-05-T07-21-55/version_0/epoch=417-step=19999.ckpt \
    -bs 2
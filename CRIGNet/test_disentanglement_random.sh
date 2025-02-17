# 512
# name=00001-crig_fastgan_g1w0.5_rainTrainL_512-FastGAN
# python disentanglement_random.py \
#     --netED ./log_raintrainl_512/${name}/model/ED_state_600.pt \
#     --backbone FastGAN \
#     --patchSize 512 \
#     --result_path ./result_disentanglement/${name}

# 128
name=00005-crig_control_dual_cgan_g1w0.2_rainTrainL_128-VRGNet
python disentanglement_random.py \
    --netED ./log_raintrainl_128/${name}/model/ED_state_200.pt \
    --patchSize 128 \
    --result_path ./result_disentanglement/${name} --gpu_id 4
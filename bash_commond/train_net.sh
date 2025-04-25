# lr=0.00005 ignore_19
CUDA_VISIBLE_DEVICES=4,5,6 python train_net.py --num-gpus 3 --dist-url   tcp://127.0.0.1:50160 --config-file 
configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_uda.yaml OUTPUT_DIR ./output/inst_urbansyn >> inst_urbansyn.txt

# 8. train from coco and warm up together， s2t no filter 3k pixel, add random place to paste   
# 这个实验最高结果在37.3， 由于学习率设置问题， 学习率一直在高水平，10-5 左右， 导致模型波动很大， 但是有机会冲到更高的值
CUDA_VISIBLE_DEVICES=1,2 python train_net.py --num-gpus 2 --dist-url   tcp://127.0.0.1:50160 --config-file configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_uda.yaml OUTPUT_DIR ./output/inst/inst_urbansyn >> ./output/inst/inst_urbansyn.txt


# train from coco and warm up together， s2t no filter 3k pixel, add random place to paste max_iter=30k lr=0.0001 power=0.9
CUDA_VISIBLE_DEVICES=1,2 python train_net.py --num-gpus 2 --dist-url   tcp://127.0.0.1:50160 --config-file configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_uda.yaml OUTPUT_DIR ./output/inst/inst_urbansyn_30klr0.0001 >> ./output/inst/inst_urbansyn_30klr0.0001.txt

# train from coco and warm up together， s2t no filter 3k pixel, add random place to paste max_iter=30k lr=0.0001 power=3 没有uda warmup
CUDA_VISIBLE_DEVICES=1,2 python train_net.py --num-gpus 2 --dist-url   tcp://127.0.0.1:50160 --config-file configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_uda.yaml OUTPUT_DIR ./output/inst/inst_urbansyn_20klr0.0001_power3 >> ./output/inst/inst_urbansyn_20klr0.0001_power3.txt

# train from coco and warm up together， s2t no filter 3k pixel, add random place to paste max_iter=30k lr=0.0001 power=3 warmup_iter=1000 uda_warmup=5000
# use all instace for s2t and t2s, total_loss=0.5 s+0.25s2t+0.25t2s
CUDA_VISIBLE_DEVICES=1,2,3 python train_net.py --num-gpus 3 --dist-url   tcp://127.0.0.1:50160 --config-file configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_uda.yaml OUTPUT_DIR ./output/inst/inst_urbansyn_30klr0.0001_power3_allmix_add_source_loss >> ./output/inst/inst_urbansyn_30klr0.0001_power3_allmix_add_source_loss.txt


# train from coco and warm up together， s2t no filter 3k pixel, add random place to paste max_iter=30k lr=0.0001 power=3 warmup_iter=1000 uda_warmup=5000
# use all instace for s2t and t2s, total_loss=0.5 s+0.25s2t+0.25t2s  fix ego car log bug, pseudo label>由0.9 改为 0.8
CUDA_VISIBLE_DEVICES=1,2,3 python train_net.py --num-gpus 3 --dist-url   tcp://127.0.0.1:50160 --config-file configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_uda.yaml OUTPUT_DIR ./output/inst/inst_urbansyn_30klr0.0001_power3_allmix_add_source_loss_1 >> ./output/inst/inst_urbansyn_30klr0.0001_power3_allmix_add_source_loss_1.txt


# 1. use all instace for s2t and t2s, total_loss=mini batch loss,  fix ego car log bug, pseudo label>由0.9 改为 0.8
# /home/yguo/Documents/other/UDA4Inst/debug_in_img_2024-05-27-18:07:45   伪标签多了一些错误  但是小目标更好了 mAP 到了36.3左右稳定。
CUDA_VISIBLE_DEVICES=6 python train_net.py --num-gpus 1 --dist-url   tcp://127.0.0.1:50160 --config-file configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_uda.yaml OUTPUT_DIR ./output/inst/inst_urbansyn_30klr0.0001_power3_allmix_minibatch_loss >> ./output/inst/inst_urbansyn_30klr0.0001_power3_allmix_minibatch_loss.txt

# 2. same as upper. but use total_loss=0.5 s+0.25s2t+0.25t2s , to compare, bs = 1， 为了和实验1 比较,  mAP 到了36.3左右稳定, 在0.0001 poly power 2 下， 加权loss 和mini batch loss 基本相同
# use all instace for s2t and t2s, total_loss=total_loss=0.5 s+0.25s2t+0.25t2s ,  fix ego car log bug, pseudo label>由0.9 改为 0.8
# 
CUDA_VISIBLE_DEVICES=7 python train_net.py --num-gpus 1 --dist-url   tcp://127.0.0.1:50160 --config-file configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_uda.yaml OUTPUT_DIR ./output/inst/inst_urbansyn_30klr0.0001_power3_allmix_not_minibatch_loss_bs1 >> ./output/inst/inst_urbansyn_30klr0.0001_power3_allmix_not_minibatch_loss_bs1.txt


# 3. use all instace for s2t and t2s, total_loss=mini batch loss,  fix ego car log bug, pseudo label>由0.9 改为 0.8， lr 由0.0001提高到0.001， poly warmup 3000iter
# 提高学习率，为了和1/ 8 比较， 主要修改了学习率设置。想让模型有机会冲到更高的点 不要太保守----结果完全飞了， 每1000轮测评都是0 ，warm up 无效。
CUDA_VISIBLE_DEVICES=6 python train_net.py --num-gpus 1 --dist-url   tcp://127.0.0.1:50160 --config-file configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_uda.yaml OUTPUT_DIR ./output/inst/inst_urbansyn_30klr0.001_power2_warmup3k_allmix_minibatch_loss >> ./output/inst/inst_urbansyn_30klr0.001_power2_warmup3k_allmix_minibatch_loss.txt

# 4. use all instace for s2t and t2s, total_loss=mini batch loss,  fix ego car log bug, pseudo label>由0.9 改为 0.8  lr 0.0001, no poly warm up, 試驗restart poly, 每2000輪一次，縂5k
CUDA_VISIBLE_DEVICES=6 python train_net.py --num-gpus 1 --dist-url   tcp://127.0.0.1:50160 --config-file configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_uda.yaml OUTPUT_DIR ./output/inst/inst_urbansyn_50klr0.0001_power2_allmix_minibatch_loss_restart_poly >> ./output/inst/inst_urbansyn_50klr0.0001_power2_allmix_minibatch_loss_restart_poly.txt

# 5. use all instace for s2t and t2s, total_loss=mini batch loss,  fix ego car log bug, pseudo label0.9 lr 0.0001, 900k iter, power0.9, to see if the changes can reach AP 37.3 again ?
CUDA_VISIBLE_DEVICES=4 python train_net.py --num-gpus 1 --dist-url   tcp://127.0.0.1:50160 --config-file configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_uda.yaml OUTPUT_DIR ./output/inst/inst_urbansyn_900klr0.0001_power0.9_allmix_minibatch_loss >> ./output/inst/inst_urbansyn_900klr0.0001_power0.9_allmix_minibatch_loss.txt

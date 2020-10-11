GPUS=10

# CBP 20
python tools/train_net.py --eval-only --num-gpus $GPUS --config-file configs/COCO-PanopticSegmentation/panoptic_rfpn_R_101_cbp20_3x-test.yaml MODEL.WEIGHTS outputs/RFPN_101_cbp20/model_0269999.pth SOLVER.IMS_PER_BATCH 1000

# CBP20 tau 0.9
python tools/panoptic_visualization.py --config-file configs/COCO-PanopticSegmentation/panoptic_rfpn_R_50_cbp20_1x-test.yaml --ckpt outputs/RFPN_cbp20/model_0089999.pth --output-dir visualizations/CBP20_tau_0.9


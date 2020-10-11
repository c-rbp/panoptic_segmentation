i=20
echo Running $i timesteps
python tools/panoptic_visualization.py --config-file configs/COCO-PanopticSegmentation/panoptic_rfpn_R_50_cbp20_1x-test.yaml --ckpt outputs/RFPN_cbp20/model_0089999.pth --output-dir visualizations_ts/CBP20_tau_0.9_ts$i MODEL.ROI_MASK_HEAD.TIMESTEPS $i


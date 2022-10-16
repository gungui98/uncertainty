git pull && python runner.py +exp=camus/train-camus-baseline   data.labels=[BG,LV,MYO] comet_tags=[CAMUS]  system.module.dropout=0.1 --max_epochs=1
python runner.py +exp=camus/test-camus-crisp data.labels=[BG,LV,MYO] data.test_da=True comet_tags=[CAMUS]  system.module_ckpt=\${model_path}/camus-LV-MYO-segmentation-\${seed}.ckpt

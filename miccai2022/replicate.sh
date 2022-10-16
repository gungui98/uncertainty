git pull && python runner.py +exp=camus/train-camus-baseline   data.labels=[BG,LV,MYO] comet_tags=[CAMUS]  system.module.dropout=0.1 --max_epochs=1
python runner.py +exp=camus/test-camus-crisp data.labels=[BG,LV,MYO] data.test_da=True comet_tags=[CAMUS]  system.module_ckpt=outputs/2022-10-15/16-08-16/crisp-miccai2022/f279aa34afff47cd96bcbef6fda24608/checkpoints/epoch\=262-step\=6048.ckpt

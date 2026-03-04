# FNC-UCH
False Negative Cancellation for Unsupervised Cross-modal Hashing with Prompt Tuning.

## Visualization
https://github.com/user-attachments/assets/6c84fe3c-eada-408c-b4d1-7c7830cc7cf5

## Environment
You can use the following command to deploy the environment：  
```
pip install -r requirements.txt
```

## Datasets
We release the three experimental datasets as follows:  
[Baidu Pan](https://pan.baidu.com/s/1GesyJwARpvoLo6dHCOScOw?pwd=zzgx)  
After downloading these three datasets, please modify the relative path in ```src/load_mat.py```.

## Usage
Our model can be trained and verified by the following command:
```
Python main_mirflickr25k.py
Python main_nuswide.py
Python main_mscoco.py
```
You can get outputs as follows:
```
epoch=24, Train time=169.86
Base hash representation Avg Loss: 5.551 | LR: 0.0001
Eval time=49.62
patience_counter=0
Base mAP: i2t=0.783   t2i=0.792   avg=0.787
Saving Base Max mAP : i2t=0.783   t2i=0.792   avg=0.787
epoch=25, Train time=169.38
Base hash representation Avg Loss: 5.559 | LR: 0.0001
Eval time=49.64
patience_counter=1
Base mAP: i2t=0.782   t2i=0.792   avg=0.787
epoch=26, Train time=171.86
Base hash representation Avg Loss: 5.585 | LR: 0.0001
Eval time=50.00
patience_counter=2
Base mAP: i2t=0.784   t2i=0.789   avg=0.787
epoch=27, Train time=171.83
Base hash representation Avg Loss: 5.602 | LR: 0.0001
Eval time=49.47
patience_counter=3
Base mAP: i2t=0.785   t2i=0.788   avg=0.786
epoch=28, Train time=172.11
Base hash representation Avg Loss: 5.618 | LR: 0.0001
Eval time=49.72
patience_counter=4
Base mAP: i2t=0.784   t2i=0.788   avg=0.786
epoch=29, Train time=168.69
Base hash representation Avg Loss: 5.628 | LR: 0.0001
Eval time=49.49
patience_counter=5
Base mAP: i2t=0.784   t2i=0.789   avg=0.787
epoch=30, Train time=167.17
Base hash representation Avg Loss: 5.615 | LR: 0.0001
Eval time=49.59
patience_counter=6
Base mAP: i2t=0.784   t2i=0.790   avg=0.787
epoch=31, Train time=170.39
Base hash representation Avg Loss: 5.654 | LR: 0.0001
Eval time=49.38
patience_counter=0
Base mAP: i2t=0.785   t2i=0.790   avg=0.787
Saving Base Max mAP : i2t=0.785   t2i=0.790   avg=0.787
epoch=32, Train time=170.56
Base hash representation Avg Loss: 5.654 | LR: 0.0001
Eval time=49.57
patience_counter=0
Base mAP: i2t=0.785   t2i=0.791   avg=0.788
Saving Base Max mAP : i2t=0.785   t2i=0.791   avg=0.788
```

# MYEXP

## 在本机CPU上跑测试数据集

### 数据准备

添加路径`./dataset/ETT-small/`，并拷贝`ETTh1.csv`至该路径下

**`ETTh1.csv`数据梳理**：

样本数量

- 行数: 17,420；列数: 8

列说明

- date: 时间戳，格式为 `YYYY-MM-DD HH:MM:SS`
- HUFL: 未知上限流量
- HULL: 未知下限流量
- MUFL: 中部未知上限流量
- MULL: 中部未知下限流量
- LUFL: 下部未知上限流量
- LULL: 下部未知下限流量
- OT: 外部温度

数据概览

| date                | HUFL  | HULL  | MUFL  | MULL  | LUFL  | LULL  | OT     |
| ------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ------ |
| 2016-07-01 00:00:00 | 5.827 | 2.009 | 1.599 | 0.462 | 4.203 | 1.340 | 30.531 |
| 2016-07-01 01:00:00 | 5.693 | 2.076 | 1.492 | 0.426 | 4.142 | 1.371 | 27.787 |
| 2016-07-01 02:00:00 | 5.157 | 1.741 | 1.279 | 0.355 | 3.777 | 1.218 | 27.787 |
| 2016-07-01 03:00:00 | 5.090 | 1.942 | 1.279 | 0.391 | 3.807 | 1.279 | 25.044 |
| 2016-07-01 04:00:00 | 5.358 | 1.942 | 1.492 | 0.462 | 3.868 | 1.279 | 21.948 |

### 修改命令行

`./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh`

```bash
model_name=TimeMixer

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=16

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
```

多行命令行写成一行，并且设置`--num_workers`为0

> run.py中默认设置`--num_workers`为10，开启多进程
> ```python
> parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
> ```
>
> 但在Windows中引由于使用的不是fork机制，代码容易产生错误，但在Linux上就不会发生
>
> ```python
> RuntimeError:
>         An attempt has been made to start a new process before the
>         current process has finished its bootstrapping phase.
> 
>         This probably means that you are not using fork to start your
>         child processes and you have forgotten to use the proper idiom
>         in the main module:
> 
>             if __name__ == '__main__':
>                 freeze_support()
>                 ...
> 
>         The "freeze_support()" line can be omitted if the program
>         is not going to be frozen to produce an executable
> ```
>
> 所以需要在启动实验时配置数据的加载为单进程，即`--num_workers 0`

由于是在本机的CPU上运行，需要添加`--use_gpu False`。最后将输入转化为一行的命令行

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96 --model TimeMixer --data ETTh1 --features M --seq_len 96 --label_len 0 --pred_len 96 --e_layers 2 --enc_in 7 --c_out 7 --des 'Exp' --itr 1 --d_model 16 --d_ff 32 --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 128 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 --num_workers 0 --use_gpu False
```

输出

```bash
Args in experiment:
Namespace(task_name='long_term_forecast', is_training=1, model_id='ETTh1_96_96', model='TimeMixer', data='ETTh1', root_path='./dataset/ETT-small/', data_path='ETTh1.csv', features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=0, pred_len=96, seasonal_patterns='Monthly', inverse=False, 
top_k=5, num_kernels=6, enc_in=7, dec_in=7, c_out=7, d_model=16, n_heads=4, e_layers=2, d_layers=1, d_ff=32, moving_avg=25, factor=1, distil=True, dropout=0.1, 
embed='timeF', activation='gelu', output_attention=False, channel_independence=1, decomp_method='moving_avg', use_norm=1, down_sampling_layers=3, down_sampling_window=2, down_sampling_method='avg', num_workers=0, itr=1, train_epochs=10, batch_size=128, patience=10, learning_rate=0.01, des="'Exp'", loss='MSE', lradj='TST', pct_start=0.2, use_amp=False, comment='none', use_gpu=False, gpu=0, use_multi_gpu=False, devices='0,1', p_hidden_dims=[128, 128], p_hidden_layers=2)        
Use CPU
>>>>>>>start training : long_term_forecast_ETTh1_96_96_none_TimeMixer_ETTh1_sl96_pl96_dm16_nh4_el2_dl1_df32_fc1_ebtimeF_dtTrue_'Exp'_0>>>>>>>>>>>>>>>>>>>>>>>>>>train 8449
val 2785
test 2785
Epoch: 1 cost time: 11.956937074661255
Epoch: 1, Steps: 66 | Train Loss: 0.4895856 Vali Loss: 0.7318788 Test Loss: 0.4053843
Validation loss decreased (inf --> 0.731879).  Saving model ...
Updating learning rate to 0.005257554516726605
Epoch: 2 cost time: 11.2597496509552
Epoch: 2, Steps: 66 | Train Loss: 0.3659438 Vali Loss: 0.6979342 Test Loss: 0.3866798
Validation loss decreased (0.731879 --> 0.697934).  Saving model ...
Updating learning rate to 0.009999911494779062
Epoch: 3 cost time: 11.18156099319458
Epoch: 3, Steps: 66 | Train Loss: 0.3519753 Vali Loss: 0.7207789 Test Loss: 0.3777077
EarlyStopping counter: 1 out of 10
Updating learning rate to 0.009607932724027022
Epoch: 4 cost time: 11.184751987457275
Epoch: 4, Steps: 66 | Train Loss: 0.3423146 Vali Loss: 0.7217298 Test Loss: 0.3773392
EarlyStopping counter: 2 out of 10
Updating learning rate to 0.008514441011874726
Epoch: 5 cost time: 11.611813306808472
Epoch: 5, Steps: 66 | Train Loss: 0.3347292 Vali Loss: 0.6961078 Test Loss: 0.3754612
Validation loss decreased (0.697934 --> 0.696108).  Saving model ...
Updating learning rate to 0.00688591055897031
Epoch: 6 cost time: 11.138419151306152
Epoch: 6, Steps: 66 | Train Loss: 0.3242840 Vali Loss: 0.7074998 Test Loss: 0.3752505
EarlyStopping counter: 1 out of 10
Updating learning rate to 0.004970270364103151
Epoch: 7 cost time: 11.133660554885864
Epoch: 7, Steps: 66 | Train Loss: 0.3151764 Vali Loss: 0.7150953 Test Loss: 0.3751171
EarlyStopping counter: 2 out of 10
Updating learning rate to 0.0030591592816201674
Epoch: 8 cost time: 11.032402515411377
Epoch: 8, Steps: 66 | Train Loss: 0.3069362 Vali Loss: 0.7209707 Test Loss: 0.3770323
EarlyStopping counter: 3 out of 10
Updating learning rate to 0.00144352664956429
Epoch: 9 cost time: 11.139173030853271
Epoch: 9, Steps: 66 | Train Loss: 0.2999568 Vali Loss: 0.7208083 Test Loss: 0.3793353
EarlyStopping counter: 4 out of 10
Updating learning rate to 0.0003693378904197436
Epoch: 10 cost time: 11.008979558944702
Epoch: 10, Steps: 66 | Train Loss: 0.2963341 Vali Loss: 0.7178647 Test Loss: 0.3791638
EarlyStopping counter: 5 out of 10
Updating learning rate to 1.2850522093768517e-07
>>>>>>>testing : long_term_forecast_ETTh1_96_96_none_TimeMixer_ETTh1_sl96_pl96_dm16_nh4_el2_dl1_df32_fc1_ebtimeF_dtTrue_'Exp'_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<test 2785
test shape: (21, 128, 96, 7) (21, 128, 96, 7)
test shape: (2688, 96, 7) (2688, 96, 7)
mse:0.375461220741272, mae:0.4003911018371582
rmse:0.612748920917511, mape:0.6779062747955322, mspe:48360.48828125
```

## 使用GPU跑测试数据集

**使用单张GPU运行模型**

由于是在Linux上跑数据，去掉`--num_workers 0`配置, 由于GPU的参数均为默认真值，故不需要设置

> run.py中GPU设置相关代码
> ```python
> # GPU
> parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
> parser.add_argument('--gpu', type=int, default=0, help='gpu')
> parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
> parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
> ```
>

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96 --model TimeMixer --data ETTh1 --features M --seq_len 96 --label_len 0 --pred_len 96 --e_layers 2 --enc_in 7 --c_out 7 --des 'Exp' --itr 1 --d_model 16 --d_ff 32 --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 128 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2
```

使用多张GPU运行模型

> 由于实验室的服务器有很多人在使用，使用多卡训练时容易内存不足，故不采取次策略
>
> ```bash
> RuntimeError: CUDA error: out of memory
> CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
> For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
> Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
> ```

## 基于ETTm1在本机上使用自定义数据集




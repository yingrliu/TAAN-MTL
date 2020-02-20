## data preparation
In `data.py`, there are some tools to transform TFRecord into 
HDF5. As our models are built on _Pytorch_, it is not beneficial
to use TF-Record. To run the script in `data.py`, please change
the `csv_path/train_path/valid_path` to the paths of the categories
csv file, the training data and the valid data. We don't use
test data in our experiment. The function `TFRecord2hdf5(mode='train' or 'valid')`
will save the hdf5 file in `./data`. Each hdf5 file has the following
domains:
- 'features' -- the input feature of data;
- 'labels' -- the ground-true label of data;
- .attrs['num_classes'] -- the number of classes in this task.

## Pytorch Data Loader
The Pytorch data loader is implemented in `torch_loader.py` as `MultiTask_Dataloader`. If you
change the saveto path in `data.py` in pre-processing, please also
change the `data_path` in `torch_loader.py`.

## Examples command lines to run the code
 _Note: all the models will have three layers._
- Single-Task:
```shell
python  Trainer.py --batch_size 256 --max_epoch 10 --hidden_feature 1024 --lr 0.0001 \
--model STL --saveto results/STL
```

- Hard-Sharing Model (sharing all the hidden layers):
```shell
python  Trainer.py --batch_size 256 --max_epoch 10 --hidden_feature 1024 --lr 0.0001 \
--model Hard3 --saveto results/Hard3
```

- Soft-Order:
```shell
python  Trainer.py --batch_size 256 --max_epoch 10 --hidden_feature 1024 --lr 0.0001 \
--model SoftOrder --saveto results/SoftOrder
```

- Cross-Stitch:
```shell
python  Trainer.py --batch_size 256 --max_epoch 10 --hidden_feature 1024 --lr 0.0001 \
--model CrossStitch --saveto results/CrossStitch
```

- MMoE:
```shell
python  Trainer.py --batch_size 256 --max_epoch 10 --hidden_feature 1024 --lr 0.0001 \
--model MMoE --saveto results/MMoE
```

- MRN:
```shell
python  Trainer.py --batch_size 256 --max_epoch 10 --hidden_feature 1024 --lr 0.0001 \
--model MRN --regularization_task True --regularization_feature False --regularization_input False\
--mrn_constant 1e-3 --saveto results/MRN
```

- DMTRL:
    - Tucker
    ```shell
    python  Trainer.py --batch_size 256 --max_epoch 10 --hidden_feature 1024 --lr 0.0001 \
    --model DMTRL --method Tucker --saveto results/DMTRL_Tucker
    ```
    - TT
    ```shell
    python  Trainer.py --batch_size 256 --max_epoch 10 --hidden_feature 1024 --lr 0.0001 \
    --model DMTRL --method TT --saveto results/DMTRL_TT
    ```
    - LAF
    ```shell
    python  Trainer.py --batch_size 256 --max_epoch 10 --hidden_feature 1024 --lr 0.0001 \
    --model DMTRL --method LAF --saveto results/DMTRL_LAF
    ```

- TAAN:
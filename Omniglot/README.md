## Data Loading
To load the data, download the Omniglot dataset [[link]](https://github.com/brendenlake/omniglot/tree/master/python) and
extract both the `images_background.zip` and `images_evaluation.zip` into the same folder. The dataset is
originally designed for one-shot learning, the train/valid splitting can not be used in our Multi-Task Learning
experiments. To generate the dataset splitting, we should first change the `dataTool.MAIN_DIR` to
the path of the dataset and run `dataTool.convert_paths()` to randomly split the images of 
each task (language) into train and test sets.

## Run
Change the `TAAN.MAIN_DIR` to the path of the dataset and run:
```shell
python Evaluation_TAAN.py --regularize cosine --gpu 0
```

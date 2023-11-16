# rna-active-learning

Capstone project for 02763 MSAS Capstone 1 at CMU

This project solves multiple rna related problem using xgboost with active learning.

## Environment

To set up the environment, you can run
```shell
conda env create -f environment.yml
```

To activate the conda environment, you can run
```shell
conda activate 02763
```

## Data

There are two types of data. One describes the class of each rna sequence. The other describes a function based on rna features.

### Class labels

To extract classes from rna sequences, you can run
```python
python data/classifier.py
```

### Regression labels

To generate a regression object from rna sequences, you can run
```python
python data/prediction.py
```

## Training

There are three models solving classification problem. The 

### Class prediction

To predict the class of the rna sequence, with the level (eg. 3) of class we want to predict, we can run 
```python
python models/classifier.py -i inputs/SILVA_138_3_8_sliding_0 -c 3
```

Note here active learning is not implemented due to computation complexity.

### Class pairing

To predict whether a pair of rna sequences comes from the same class (eg. level 3), you can run
```python
python models/pair.py -i inputs/SILVA_138_3_8_sliding_0 -c 3
```

### Function predicting

To predict the function, which is also formed as a classification problem by setting a threshold, we can run
```python
python models/prediction.py -i inputs/SILVA_138_3_8_sliding_0 -p 80
```

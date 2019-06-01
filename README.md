# Incorporating-Priors

This repository contains implementation of paper --
Incorporating Priors with Feature Attribution on Text Classification (ACL2019)

## How to train a model 
```
python train.py --num_classes=2 --output_dir='/tmp/attr/' --target_words_file='./data/wiki/identity.txt'
```
By default, the command trains the model with joint loss. (attribution + classification)

Check 
```
python train.py -h
```
for other settings.

## How to predict and generate global attributions (average of local attributions). 
```
python explain.py --model_dir=/tmp/attr --pred_output=/tmp/pred.txt
```
Check 
```
python explain.py -h
```
for other settings.

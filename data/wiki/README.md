# Resources

## Wikipedia Talk Labels: Toxicity
### Source
[Wikipedia Talk Labels: Toxicity](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973)
* toxicity_annotated_comments.tsv
* toxicity_annotations.tsv
* toxicity_worker_demographics.tsv
### Processed
The dataset is processed into 
* wiki_train.txt
* wiki_dev.txt
* wiki_test.txt

with Data_Preperation.ipynb
modified from
[unintended-ml-bias-analysis](https://github.com/conversationai/unintended-ml-bias-analysis/blob/master/unintended_ml_bias/Prep%20Wikipedia%20Data.ipynb)

## Identity Terms
### Source
[unintended-ml-bias-analysis](https://github.com/conversationai/unintended-ml-bias-analysis/tree/master/unintended_ml_bias/new_madlibber/input_data/English)
* words.csv
### Processed
The identity terms are extracted into
* identitiy.txt 

## Toxic Terms
We manually picked 50 swearwords from the vocabulary of the corpus as toxic terms.
### Processed
The handpicked swearwords
* toxic.txt

## Synthetic evaluation data
### Source
[unintended-ml-bias-analysis](https://github.com/conversationai/unintended-ml-bias-analysis/tree/master/unintended_ml_bias/new_madlibber/input_data/English)
The synthetic evaluation data is generated with the following files.
* words.csv
* intersectional_sentence_templates.csv
### Processed
Syntehtic test set
* synth_test.txt


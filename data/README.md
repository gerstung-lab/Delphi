# Data preparation

If you would like to train Delphi on your own data, you can use the following instructions for data preparation.

## Event records

First, you need to prepare the actual disease records.
An easy way of doing this is to prepare a Numpy array with the following columns of dtype `np.uint32`:
- `patient_id`
- `patient_age` (in days)
- `token_id`

`patient_id` should be a unique identifier for each patient. It is used exclusively for identifying which records belong to the same patient and it is not used in the training. The entries should be consecutive entries, meaning that first in the array come all the rows related to the first patient, then all the rows related to the second patient and so on.
`patient_age` is the age of the patient in days at the time of recording the disease.
`token_id` is some unique identifier for the data record, a `uint32` number, starting from `1` (`0` is reserved for adding `no event` and `padding` tokens later). In Delphi-2M, it could be a disease event, a lifestyle token or sex token. All the tokens are processed by the model in a unified way (however, there are some tokens that are not used for calculating the gradients - see below for more details).

Then split this numpy array into train/val in a desired ratio and save into `data/%your_name%` folder.

```python
data_val.astype(np.uint32).tofile('val.bin')
data_train.astype(np.uint32).tofile( 'train.bin')
```

## Label file

See `data/ukb_simulated_data/labels.csv` for the example.
The labels aren't required for training, but are later used in the downstream notebooks for `token_id` -> `event_name` mapping.
The labels file should be a CSV with only one column:
- `event_name`

**None!** The row N+1 of this file corresponds to the N-th token. E.g. if in the `train.bin` file you used `token_id` 42 for `Common cold`, then the `Common cold` should be the 43-th row in the `labels.csv` file. This happens because we insert `no event` and `padding` tokens in the beginning of the file.

Place this file into the `data/%your_name%` folder.


## Conversion of UK Biobank records to delphi format

We provide an example notebook to illustrate how to convert UK Biobank first occurances data into the format needed for Delphi:
- data/ukb_simulated_data/example_ukb_to_bin.ipynb

This involves using a mapping file between UK Biobank field IDs, the ICD10 code scheme and the delphi index file.
This process results in the splitting of the full dataset into training and validation .bin files.
It will also require a UK biobank basket download file which needs to be in ".tab" format.
See UK Biobank documentation for further details.

## Preparing the model

Either by defining in the `config/%my_config%.py` file or by passing as a command line argument.

The following parameters likely need to be adjusted:
- `vocab_size` - maximum `token_id` in your data + 1 (for padding)
- `ignore_tokens` - list of tokens that are not used for calculating the gradients (we exclude padding, sex and lifestyle tokens)
- `dataset` - name of the dataset, e.g. `ukb_real_data` or `ukb_simulated_data`

## Training

Now you are ready to train the model.

```bash
python train.py config/%my_config%.py --dataset %your_name% --out_dir=%your_model_name%
``` 

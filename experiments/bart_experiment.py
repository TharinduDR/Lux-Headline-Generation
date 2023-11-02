import os

import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from config.model_args import Seq2SeqArgs
from experiments.evaluation import bleu, ter

from seq2seq.seq2seq_model import Seq2SeqModel


model_name = "facebook/mbart-large-cc25"
model_type = "mbart"
model_representation = model_name.replace('/', '-')


SEED = 777

train = Dataset.to_pandas(load_dataset('cucolab/lux-headlines', split='train'))
test = Dataset.to_pandas(load_dataset('cucolab/lux-headlines', split='test'))

train["prefix"] = ""
test["prefix"] = ""

train = train.rename(columns={'content': 'input_text', 'headline': 'target_text'})
test = test.rename(columns={'content': 'input_text', 'headline': 'target_text'})

model_args = Seq2SeqArgs()
model_args.num_train_epochs = 10
model_args.no_save = False
model_args.fp16 = False
model_args.learning_rate = 1e-5
model_args.train_batch_size = 8
model_args.max_seq_length = 256
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.evaluate_during_training_steps = 10000
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False
model_args.use_multiprocessed_decoding = False
model_args.overwrite_output_dir = True
model_args.save_recent_only = True
model_args.logging_steps = 10000
model_args.manual_seed = SEED
model_args.early_stopping_patience = 25
model_args.save_steps = 10000

model_args.output_dir = os.path.join("outputs", model_representation)
model_args.best_model_dir = os.path.join("outputs", model_representation, "best_model")
model_args.cache_dir = os.path.join("cache_dir", model_representation)

model_args.wandb_project = "LUX Headline Generation"
model_args.wandb_kwargs = {"name": model_name}

model = Seq2SeqModel(
    encoder_decoder_type=model_type,
    encoder_decoder_name=model_name,
    args=model_args,
    use_cuda=torch.cuda.is_available()
)

train, eval = train_test_split(train, test_size=0.1, random_state=SEED)
model.train_model(train, eval_data=eval)

input_list = test['input_text'].tolist()
truth_list = test['target_text'].tolist()

model = Seq2SeqModel(
    encoder_decoder_type=model_type,
    encoder_decoder_name=model_args.best_model_dir,
    args=model_args,
    use_cuda=torch.cuda.is_available()
)

preds = model.predict(input_list)

del model

test["predictions"] = preds
test.to_csv(os.path.join("outputs", model_representation, "predictions.tsv"), sep='\t', encoding='utf-8', index=False)


print("Bleu Score ", bleu(truth_list, preds))
print("Ter Score ", ter(truth_list, preds))


import pandas as pd
from tokenizer import get_tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset

tokenizer = get_tokenizer() #CodonBERT's tokenizer
print(tokenizer.encode("[CLS] [SEP]").ids) #Prints [2, 2, 3, 3]

# Fast tokenizer from huggingface for faster more efficient tokenization
pre_tok_fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                       do_lower_case=False,
                                       clean_text=False,
                                       tokenize_chinese_chars=False,
                                       strip_accents=False,
                                       unk_token='[UNK]',
                                       sep_token='[SEP]',
                                       pad_token='[PAD]',
                                       cls_token='[CLS]',
                                       mask_token='[MASK]')

# use BERT's cls token as BOS token and sep token as EOS token
# in CodonBERT, the bos_token id is 2 and eos_token id is 3
encoder = BertGenerationEncoder.from_pretrained("../CodonBERT/codonbert_models/codonbert", bos_token_id=2, eos_token_id=3)
# add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
decoder = BertGenerationDecoder.from_pretrained("../CodonBERT/codonbert_models/codonbert", add_cross_attention=True, is_decoder=True, bos_token_id=2, eos_token_id=3)
bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)
bert2bert.config.decoder_start_token_id = pre_tok_fast.vocab['[CLS]']
bert2bert.config.pad_token_id = pre_tok_fast.vocab['[PAD]']

# training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch", #how often should the model be evaluated on the validation set
    learning_rate=2e-5, #learning rate of the optimizer
    per_device_train_batch_size=8, # number of samples to be processed at a time
    per_device_eval_batch_size=8, #number of samples to be validated at a time
    weight_decay=0.01, #L2 regularization
    save_total_limit=3, # number of checkpoint files saved during training
    num_train_epochs=3, # number of times the model will iterate over the entire training dataset
    predict_with_generate=True, # the model will generate sequences during evaluation rather than just returning logits or probabilities
)

# Create the trainer for model
data_collator = DataCollatorForSeq2Seq(pre_tok_fast, model=bert2bert)

# Prepare dataset for fine-tuning
# Load the data of low and high quality mRNAs
data = pd.read_csv('sample_dataset.csv')
# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(data)

# Preprocess data and format source and taregt sequences to be compatible with CodonBERT
def preprocess_function(examples):
    source = examples['Low_Expression_mRNA']
    target = examples['High_Expression_mRNA']
    spaced_source = ' '.join([source[i:i+3] for i in range(0, len(source), 3)])
    spaced_target = ' '.join([target[i:i+3] for i in range(0, len(target), 3)])
    model_inputs = {'input_ids': tokenizer.encode(spaced_source).ids}
    labels = tokenizer.encode(spaced_target)
    model_inputs['labels'] = labels.ids
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=False)

trainer = Seq2SeqTrainer(
    model=bert2bert,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=pre_tok_fast,
)

trainer.train() #Fine-tune the model
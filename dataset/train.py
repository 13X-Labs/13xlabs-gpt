from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("anonymous-german-nlp/german-gpt2")

train_path = 'small.train.jsonl'
test_path = 'small.test.jsonl'

def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)

    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator

train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)


model = AutoModelForCausalLM.from_pretrained("anonymous-german-nlp/german-gpt2")

training_args = TrainingArguments(
    output_dir="./gpt2-gerchef", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=3, # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    eval_steps = 400, # 400 Number of update steps between two evaluations.
    save_steps=400, # 400 after # steps model is saved
    warmup_steps=500,#  500number of warmup steps for learning rate scheduler
	# num_samples=1,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
	
    # prediction_loss_only=True,
)


trainer.train(

)

trainer.save_model(
	
)


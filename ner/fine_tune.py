import json
import os
import random
import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator


INPUT_FOLDER = "data"
INPUT_FILE_NAME = "20240922-222908.json"

os.environ["TOKENIZERS_PARALLELISM"] = "true"


if __name__ == "__main__":
    print("--Handle Dataset--")
    full_path = os.path.realpath(__file__)
    dir_name = os.path.dirname(full_path)
    data_path = dir_name + "/" + INPUT_FOLDER + "/" + INPUT_FILE_NAME

    print(f"Reading the {data_path} dataset.")
    with open(data_path, "r") as f:
        data = json.load(f)

    print('Dataset size:', len(data))

    random.shuffle(data)
    print('Dataset is shuffled...')

    train_dataset = data[:int(len(data)*0.9)]
    test_dataset = data[int(len(data)*0.9):]

    print('Dataset is splitted...')
    print('Training dataset size:', len(train_dataset))
    print('Testing dataset size:', len(test_dataset))

    print("--Handle Configurations--")
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    print(f"Using {device}...")
    model = GLiNER.from_pretrained("urchade/gliner_small")
    model.to(device)

    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

    num_steps = 500
    batch_size = 2
    data_size = len(train_dataset)
    num_batches = data_size // batch_size
    num_epochs = max(1, num_steps // num_batches)

    training_args = TrainingArguments(
        output_dir="models",
        learning_rate=5e-6,
        weight_decay=0.01,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear", #cosine
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2,
        num_train_epochs=num_epochs,
        evaluation_strategy="steps",
        save_steps = 100,
        save_total_limit=10,
        dataloader_num_workers = 0,
        use_cpu = False,
        report_to="none",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    print("--Handle Training--")
    trainer.train()



import wandb
from jcqa_dataset import get_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
import torch
from trl import SFTTrainer
import time
from utils import get_model_name


def fine_tuning(model_path, tokenizer, torch_dtype, model_save_dir):
    """ Fine-tuneを行う
        openorca-stx, JCommonsenseQAでのみ動作確認済み
    """
    # NOTE: you need to login to wandb before running this script
    wandb.init(project=f"llmsummer2023-{get_model_name(model_path)}")

    # bitsandbytes parameters
    # Activate 4-bit precision base model loading
    use_4bit = True
    # Compute dtype for 4-bit base models
    # bnb_4bit_compute_dtype = "float16"
    bnb_4bit_compute_dtype = "bfloat16"
    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"
    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = True
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    model_q = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=use_4bit,  # TODO
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    # TODO: ?
    model_q.config.use_cache = False  # for fine-tuning; disable this for inference
    model_q.config.pretraining_tp = 1
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py

    # LoRA parameters
    task_type = "CAUSAL_LM"
    target_modules = [
        "k_proj",
        "o_proj",
        "q_proj",
        "gate_proj",
        "v_proj",
        "down_proj",
        "up_proj",
    ]
    bias = "none"
    lora_r = 64  # LoRA rank dimension
    lora_alpha = 16  # Alpha parameter for LoRA scaling
    # lora_dropout = 0.1  # Dropout probability for LoRA layers
    lora_dropout = 0.05
    output_dir = model_save_dir  # model predictions and checkpoints will be stored
    num_train_epochs = 1
    bf16 = False
    fp16 = False
    if torch_dtype == torch.bfloat16:
        bf16 = True
    elif torch_dtype == torch.float16:
        fp16 = True
    per_device_train_batch_size = 4  # Batch size per GPU for training
    # per_device_eval_batch_size = 4  # Batch size per GPU for evaluation
    # per_device_train_batch_size = 8
    # per_device_eval_batch_size = 16
    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1
    # gradient_accumulation_steps = 2
    gradient_checkpointing = True  # Enable gradient checkpointing
    max_grad_norm = 0.3  # Maximum gradient normal (gradient clipping)
    learning_rate = 2e-5  # Initial learning rate (AdamW optimizer)
    # learning_rate = 2e-4
    # Weight decay to apply to all layers except bias/LayerNorm weights
    # weight_decay = 0.01
    weight_decay = 0.001
    # weight_decay = 0.0
    optim = "paged_adamw_32bit"  # Optimizer to use
    # optim = "adamw_bnb_8bit"  # TODO
    # optim = "adafactor"  # TODO
    # Learning rate schedule (constant a bit better than cosine)
    lr_scheduler_type = "constant"
    max_steps = -1  # Number of training steps (overrides num_train_epochs)
    warmup_ratio = 0.03  # Ratio of steps for a linear warmup (from 0 to learning rate)
    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True
    save_steps = 1000  # Save checkpoint every this updates steps
    logging_steps = 10  # Log every this updates steps

    # SFT parameters
    # Maximum sequence length to use
    # max_seq_length = 512
    max_seq_length = 256
    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias=bias,
        task_type=task_type,
        target_modules=target_modules,
        inference_mode=False,
    )

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb",
    )

    dataset = get_dataset(tokenizer)

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model_q,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],  # TODO
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    start_time = time.time()
    trainer.train()
    print("学習にかかった秒数", time.time() - start_time)

    # 学習済みモデルの保存
    trainer.save_model(model_save_dir / "checkpoint")

    model = AutoPeftModelForCausalLM.from_pretrained(  # load non-quantized QLoRA model
        model_save_dir / "checkpoint",
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model = model.merge_and_unload()
    model.save_pretrained(model_save_dir)  # TODO: safe_serialization=True?
    tokenizer.save_pretrained(model_save_dir)

    return model, tokenizer

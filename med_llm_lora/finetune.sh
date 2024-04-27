python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path './data/train.json' \
    --micro_batch_size 32 \
    --output_dir './med_llm_lora' 
    
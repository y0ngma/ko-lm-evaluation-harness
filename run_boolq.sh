# ./run_all.sh jyoung105/KoR-Orca-Platypus-13B-neft
export CUDA_VISIBLE_DEVICES=$2
export TOKENIZERS_PARALLELISM=false

RESULT_DIR='/data/yhjeong/home/output/eval/kobest_boolq'
TASKS='kobest_boolq'
MODEL=$1 #/home/beomi/coding-ssd2t/EasyLM/llama-2-ko-7b
MODEL_PATH=$(echo $MODEL | awk -F/ '{print $(NF-1) "/" $NF}')
echo $MODEL
echo $MODEL_PATH
# peft_model_id = "/data/yhjeong/home/samsum/2024-02-29/checkpoint-40"
# peft_model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16, offload_folder="lora_results/lora_7/temp")

echo "mkdir -p $RESULT_DIR/$MODEL_PATH/$CURRENT_TRAINED_TOKENS"
mkdir -p $RESULT_DIR/$MODEL_PATH/$CURRENT_TRAINED_TOKENS

SRC_DIR="/data/yhjeong/home/ko-lm-evaluation-harness"
# model에는 hf-causal-experimental 같은 문자열 또는 lm_eval.base.LM 객체(이때 model_args생략)
CUDA_VISIBLE_DEVICES=$2 python $SRC_DIR/main.py \
--model hf-causal-experimental \
--model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
--tasks $TASKS \
--num_fewshot 0 \
--no_cache \
--batch_size 8 \
--output_path $RESULT_DIR/$MODEL/0_shot.json

# CUDA_VISIBLE_DEVICES=$2 python $SRC_DIR/main.py \
# --model hf-causal-experimental \
# --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
# --tasks $TASKS \
# --num_fewshot 5 \
# --no_cache \
# --batch_size 4 \
# --output_path $RESULT_DIR/$MODEL/5_shot.json

# CUDA_VISIBLE_DEVICES=$2 python $SRC_DIR/main.py \
# --model hf-causal-experimental \
# --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
# --tasks $TASKS \
# --num_fewshot 10 \
# --no_cache \
# --batch_size 2 \
# --output_path $RESULT_DIR/$MODEL/10_shot.json

# CUDA_VISIBLE_DEVICES=$2 python $SRC_DIR/main.py \
# --model hf-causal-experimental \
# --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
# --tasks $TASKS \
# --num_fewshot 50 \
# --no_cache \
# --batch_size 1 \
# --output_path $RESULT_DIR/$MODEL/50_shot.json


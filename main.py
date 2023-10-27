from dotenv import load_dotenv
import json
import random
import time
from pathlib import Path

from auto_gptq import exllama_set_max_input_length
from langchain.utilities import GoogleSerperAPIWrapper
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from utils import get_model_name, flush, trunc_input, make_answer, make_prompt
from ft import fine_tuning

from datasets import disable_caching

disable_caching()
load_dotenv()

root = Path(__file__).parent
nas_root = root  # FIXME: your path to save models
model_path = "TheBloke/StableBeluga2-70B-GPTQ"
torch_dtype = torch.float16
# torch_dtype = torch.bfloat16

do_fine_tuning = False
# extra_disc = ""
extra_disc = "-newtest"

seed = 42

torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
random.seed(seed)

model_name = get_model_name(model_path)
# test_json = root / "data" / "test.json"
test_json = root / "data" / "new_test.json"
pred_json = root / "preds" / f"{model_name}{extra_disc}.json"
model_save_dir = nas_root / "models" / f"{model_name}{extra_disc}"
pred_json.parent.mkdir(exist_ok=True, parents=True)
model_save_dir.mkdir(exist_ok=True, parents=True)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False,  # TODO: True?
)
tokenizer.pad_token_id = tokenizer.eos_token_id

if not do_fine_tuning:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        # use_cache=False,
    )
    if "GPTQ" in model_name:
        model = exllama_set_max_input_length(model, 4096)
        # The temp_state buffer is too small in the exllama backend. Please call
        # the exllama_set_max_input_length function to increase the buffer size. Example:
        # from auto_gptq import exllama_set_max_input_length
        # model = exllama_set_max_input_length(model, 4096)
else:
    model, tokenizer = fine_tuning(model_path, tokenizer, torch_dtype, model_save_dir)

print(model.generation_config)

search = GoogleSerperAPIWrapper()

# 推論
start_time = time.time()
with open(test_json, "r") as f:
    test_data = json.load(f)

choices_ids = tokenizer.convert_tokens_to_ids(["1", "2", "3", "4", "5"])

for data in test_data:
    print("\n" + "*" * 30)
    print(f"question {data['id']}")

    rag_prompt = ""

    if data["task_type"] == "multiple_choice":
        # 各選択肢番号のトークンが次に来る確率を計算
        prompt = make_prompt(data, model_name)
        inputs = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model(inputs)
        logits = outputs.logits[0, -1, torch.tensor(choices_ids)]
        probs = torch.softmax(logits, dim=-1)
        print("probs:" + ", ".join(f"{p:.3f}" for p in probs.tolist()))

        # 確率が基準以下の場合はGoogle検索でRAGする
        thr_prob_abs = 0.7  # top1の確率がこの値以下ならRAG
        thr_prob_rel = 0.15  # top2の確率の差がこの値以下ならRAG
        num_retrieval = 2  # 使う検索結果の個数

        p_sorted, _ = torch.sort(probs, descending=True)
        do_retrieval = (p_sorted[0] < thr_prob_abs) or (
            p_sorted[0] - p_sorted[1] < thr_prob_rel
        )
        if do_retrieval:
            rag_prompt = "\n[参考: Google検索結果]:\n"
            for choice in data["choices"]:
                query = f"{data['text']} {choice['text']}"
                ret = search.run(query)
                ret_use = "...".join(ret.split("...")[:num_retrieval])
                rag_prompt += f"{choice['choice_id']}. 検索ワード: \"{query}\"\n"
                rag_prompt += f"{ret_use}\n"

        del inputs, outputs

    prompt = make_prompt(data, model_name, additional_prompt=rag_prompt)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(
        model.device
    )

    inputs, max_length = trunc_input(inputs, data["task_type"])

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    output = tokenizer.decode(
        output_ids.tolist()[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
    )

    # log
    trunced_prompt = tokenizer.decode(
        inputs.input_ids.tolist()[0], skip_special_tokens=True
    )
    print(f"{'-' * 10}\n{trunced_prompt}{'-' * 10}\n{output}\n{'-' * 10}")

    data["answer"] = make_answer(data["task_type"], output)

    # メモリ解放
    del inputs, output_ids
    flush()

print("推論にかかった秒数", time.time() - start_time)


# 推論結果の保存
# id, task_type, text, answerのkeyがあることを確認してください
# (正しく入っていない場合、スコア付けが行われません)
with open(pred_json, "w") as f:
    json.dump(test_data, f, indent=4)

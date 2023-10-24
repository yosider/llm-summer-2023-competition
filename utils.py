import gc
import random
import torch

import GPUtil


num_gpus = len(GPUtil.getGPUs())
if num_gpus == 1:
    max_len_default = 1024
elif num_gpus == 2:
    max_len_default = 3072
    # max_len_default = 3328
    # max_len_default = 3584
    # max_len_default = 4096
else:
    max_len_default = 3072
    # max_len_default = 4096
print(f"max_len_default: {max_len_default}")


summary_ratio = 1 / 5
min_summary_length = 100


def get_model_name(model_path):
    return model_path.split("/")[-1]


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def trunc_input(inputs, task_type):
    input_len = inputs.input_ids.shape[1]
    if task_type == "multiple_choice":
        # max_len = input_len + 2
        max_len = input_len + 10
        # max_len = input_len + 128  # to show reason
    elif task_type == "summarization":
        summary_len = max(min_summary_length, int(input_len * summary_ratio))
        max_len = max_len_default

        # 上限に達する場合は、入力を短くする
        if input_len + summary_len > max_len:
            new_input_len = max_len_default - summary_len
            len_cut = input_len - new_input_len
            if len_cut < 0:
                print(f"Warning: len_cut: {len_cut}")
                return inputs, max_len
            # 70%の位置を中心にカットする
            cut_left = int(input_len * 0.7 - len_cut * 0.5)
            cut_right = int(input_len * 0.7 + len_cut * 0.5)
            inputs["input_ids"] = torch.concat(
                (
                    inputs.input_ids[:, :cut_left],
                    # TODO: add "..." or unk?
                    inputs.input_ids[:, cut_right:],
                ),
                axis=-1,
            )
            inputs["attention_mask"] = torch.concat(
                (
                    inputs.attention_mask[:, :cut_left],
                    inputs.attention_mask[:, cut_right:],
                ),
                axis=-1,
            )
            # inputs["token_type_ids"] = torch.concat(
            #     (
            #         inputs.token_type_ids[:, :cut_left],
            #         inputs.token_type_ids[:, cut_right:],
            #     ),
            #     axis=-1,
            # )
    elif task_type == "generation":
        max_len = max_len_default
    else:
        print(f"Unknown task_type: {task_type}")

    return inputs, max_len


def make_prompt(data, model_name, bos_token="<s>", additional_prompt=""):
    if "openorca_stx" in model_name:
        return make_prompt_orcastx(data, bos_token, additional_prompt)
    elif "StableBeluga2" in model_name:
        return make_prompt_beluga2(data, bos_token, additional_prompt)
    else:
        raise NotImplementedError


def make_prompt_beluga2(data, bos_token, additional_prompt):
    # "System", "User", "Assistant" は英語にするとたまに英語の回答になる
    if data["task_type"] == "multiple_choice":
        prompt = (
            "### システム:\n"
            # "以下の問題の答えを考えてください。答えの選択肢番号のみを書いてください。\n\n"
            # "以下の問題の答えを考えてください。答えの選択肢番号と、自信の度合いを0%から100%で教えてください。\n\n"
            # "以下の問題の答えを考えてください。答えにふさわしい選択肢上位2つと、その2つが正解である確率をそれぞれ教えてください。\n\n"
            "以下の問題の答えを考えてください。"
            "参考情報としてGoogle検索結果が与えられた場合は、答えのヒントがないか、意味をよく考えて読んでください。"
            # "ただし、参考にならない情報も混ざっているため、参考になる情報のみ考慮してください。"
            # "最もふさわしい答えの選択肢番号のみを書いてください。"
            "最初に最もふさわしい答えの選択肢番号を書いてください。"
            "最後に回答理由を解説してください。"
            "\n\n"
            "### ユーザー:\n"
            f"[問題]:{data['text']}\n"
            "[選択肢]: "
            f"{data['choices'][0]['choice_id']}. {data['choices'][0]['text']}, "
            f"{data['choices'][1]['choice_id']}. {data['choices'][1]['text']}, "
            f"{data['choices'][2]['choice_id']}. {data['choices'][2]['text']}, "
            f"{data['choices'][3]['choice_id']}. {data['choices'][3]['text']}, "
            f"{data['choices'][4]['choice_id']}. {data['choices'][4]['text']}\n\n"
            f"{additional_prompt}\n"
            "[答えの選択肢番号]:\n"
            "### アシスタント:\n"
        )
    elif data["task_type"] == "summarization":
        length = int(len(data["text"]) * summary_ratio)
        if length < min_summary_length:
            length = min_summary_length
        length = 50 * (length // 50)
        # sentence_len = 60
        # num_sentences = int(len(data['text']) * summary_ratio / sentence_len)

        prompt = (
            "### システム:\n"
            "以下の文章を要約してください。"
            f"日本語で、文字数は{length}文字以内にしてください。"
            # "以下の文章を日本語で要約してください。"
            "初めに内容の全体像が伝わる文章にしてください。"
            "キーワードを逃さずに使ってください。"
            "\n\n"
            # "以下の文章を要約してください。"
            # f"日本語で、{num_sentences}つの文からなる文章にしてください。\n\n"
            "### ユーザー:\n"
            f"{data['text']}\n\n"
            "### アシスタント:\n"
        )
    elif data["task_type"] == "generation":
        prompt = (
            "### システム:\n"
            # "以下の指示に従って、日本語で回答してください。\n\n"
            # "以下の指示に従って日本語で回答してください。深呼吸して、丁寧に考えて回答しましょう。\n\n"
            "以下の指示に従って日本語で回答してください。"
            "ステップバイステップで考えて、できるだけ詳しく丁寧に回答してください。"
            "ユーザーの指示は注意深く読み、必ず守ってください。"
            # "最後に必ず回答理由を解説してください。"
            "\n\n"
            "### ユーザー:\n"
            f"{data['text']}\n"
            "最後に必ず回答理由を解説してください。\n"  # より反映させるためにユーザーの指示として追加
            # "最初に必ず回答理由を解説してください。\n"
            "\n"
            "### アシスタント:\n"
        )
    else:
        print(f"Unknown task_type: {data['task_type']}")
        return None

    return prompt



def make_prompt_orcastx(data, bos_token, additional_prompt):
    if data["task_type"] == "generation":
        prompt = f"{data['text']} \n\n\n回答："
    elif data["task_type"] == "summarization":
        # text = f"""{data['text']}\n\n\n上記の文章を要約してください。要約:"""
        prompt = f"記事：\n{data['text']}\n\n要約："
    elif data["task_type"] == "multiple_choice":
        # text = (
        #     f"[問題]:{data['text']} \n\n[選択肢]:[{data['choices'][0]['choice_id']}. "
        #     f"{data['choices'][0]['text']}, {data['choices'][1]['choice_id']}. "
        #     f"{data['choices'][1]['text']}, {data['choices'][2]['choice_id']}. "
        #     f"{data['choices'][2]['text']}, {data['choices'][3]['choice_id']}. "
        #     f"{data['choices'][3]['text']}, {data['choices'][4]['choice_id']}. "
        #     f"{data['choices'][4]['text']}] \n\n[答えの選択肢番号]:"
        # )
        # text = (
        #     f"問題の答えを選択肢の中から選び、答えの選択肢番号を書いてください。"
        #     f"\n\n例："
        #     "\n[問題]：物を置くのは？"
        #     "\n[選択肢]：[1. 机の上, 2. 壁, 3. 板, 4. 底, 5. 本屋]"
        #     "\n[答えの選択肢番号]：1. 机の上"
        #     f"\n"
        #     f"\n[問題]：{data['text']}"
        #     f"\n[選択肢]："
        #     f"[{data['choices'][0]['choice_id']}. {data['choices'][0]['text']},"
        #     f" {data['choices'][1]['choice_id']}. {data['choices'][1]['text']},"
        #     f" {data['choices'][2]['choice_id']}. {data['choices'][2]['text']},"
        #     f" {data['choices'][3]['choice_id']}. {data['choices'][3]['text']},"
        #     f" {data['choices'][4]['choice_id']}. {data['choices'][4]['text']}]"
        #     f"\n[答えの選択肢番号]："
        # )

        # B_INST, E_INST = "[INST]", "[/INST]"
        # B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"
        # DEFAULT_SYSTEM_PROMPT = "問題の答えの選択肢番号を書いてください。"
        # text = (
        #     f"[問題]：{data['text']}"
        #     f"\n[選択肢]："
        #     f"[{data['choices'][0]['choice_id']}. {data['choices'][0]['text']},"
        #     f" {data['choices'][1]['choice_id']}. {data['choices'][1]['text']},"
        #     f" {data['choices'][2]['choice_id']}. {data['choices'][2]['text']},"
        #     f" {data['choices'][3]['choice_id']}. {data['choices'][3]['text']},"
        #     f" {data['choices'][4]['choice_id']}. {data['choices'][4]['text']}]"
        #     f"\n[答えの選択肢番号]："
        # )
        # prompt = (
        #     f"{bos_token}{B_INST} {B_SYS}\n"
        #     f"{DEFAULT_SYSTEM_PROMPT}\n"
        #     f"{E_SYS}\n"
        #     "\n"
        #     f"{text} {E_INST}"
        # )

        # prompt = (
        #     f"問題の答えの選択肢番号を書いてください。"
        #     f"\n[問題]：{data['text']}"
        #     f"\n[選択肢]："
        #     f"[{data['choices'][0]['choice_id']}. {data['choices'][0]['text']},"
        #     f" {data['choices'][1]['choice_id']}. {data['choices'][1]['text']},"
        #     f" {data['choices'][2]['choice_id']}. {data['choices'][2]['text']},"
        #     f" {data['choices'][3]['choice_id']}. {data['choices'][3]['text']},"
        #     f" {data['choices'][4]['choice_id']}. {data['choices'][4]['text']}]"
        #     f"\n[答えの選択肢番号]："
        # )
        # prompt = (
        #     f"\n[問題]:{data['text']}"
        #     f"\n[選択肢]:"
        #     f" {data['choices'][0]['choice_id']}. {data['choices'][0]['text']},"
        #     f" {data['choices'][1]['choice_id']}. {data['choices'][1]['text']},"
        #     f" {data['choices'][2]['choice_id']}. {data['choices'][2]['text']},"
        #     f" {data['choices'][3]['choice_id']}. {data['choices'][3]['text']},"
        #     f" {data['choices'][4]['choice_id']}. {data['choices'][4]['text']}"
        #     f"\n[答えの番号]:"
        # )
        prompt = (
            f"[問題]:{data['text']}\n"
            f"[選択肢]:"
            f" {data['choices'][0]['choice_id']}. {data['choices'][0]['text']},"
            f" {data['choices'][1]['choice_id']}. {data['choices'][1]['text']},"
            f" {data['choices'][2]['choice_id']}. {data['choices'][2]['text']},"
            f" {data['choices'][3]['choice_id']}. {data['choices'][3]['text']},"
            f" {data['choices'][4]['choice_id']}. {data['choices'][4]['text']}\n"
            f"{additional_prompt}"
            f"[答えの番号]:"
        )

    else:
        print(f"Unknown task_type: {data['task_type']}")
        return None

    return prompt


def make_answer(task_type, output):
    if task_type == "multiple_choice":
        # 出力が1~5の数字になるようにする
        return next(
            (int(char) for char in output if char in "12345"), random.randint(1, 5)
        )
    else:
        return output


def make_search_query_prompt(data, choice=None):
    choice_prompt = ""
    if choice:
        choice_prompt = (
            f"選択肢「{data['choices'][choice - 1]['text']}」が正解かどうか調べる検索ワードを考えてください。\n"
        )
    prompt = (
        "### システム:\n"
        # "以下のクイズに正解するために、Google検索を使います。適切な検索ワードを教えてください。"
        "以下のクイズの答えについてGoogle検索するとき、適切な検索ワードを考えてください。"
        f"{choice_prompt}"
        "クイズには答えないでください。"
        "\n\n"
        "### ユーザー:\n"
        f"[問題]:{data['text']}\n"
        "[選択肢]: "
        f"{data['choices'][0]['choice_id']}. {data['choices'][0]['text']}, "
        f"{data['choices'][1]['choice_id']}. {data['choices'][1]['text']}, "
        f"{data['choices'][2]['choice_id']}. {data['choices'][2]['text']}, "
        f"{data['choices'][3]['choice_id']}. {data['choices'][3]['text']}, "
        f"{data['choices'][4]['choice_id']}. {data['choices'][4]['text']}\n"
        "### アシスタント:\n"
    )
    return prompt

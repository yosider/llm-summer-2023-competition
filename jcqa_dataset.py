from datasets import load_dataset


def get_dataset(tokenizer):
    """ Load JCommonsenseQA dataset for fine-tuning
    """
    dataset = load_dataset("shunk031/JGLUE", "JCommonsenseQA")
    text_dataset = dataset.map(
        lambda data: {"text": _make_prompt(data, tokenizer.bos_token)},
        remove_columns=[
            "q_id",
            "question",
            "choice0",
            "choice1",
            "choice2",
            "choice3",
            "choice4",
            "label",
        ],
        load_from_cache_file=False,
    )
    return text_dataset


def _make_prompt(data, bos_token="<s>"):
    # B_INST, E_INST = "[INST]", "[/INST]"
    # B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"
    # DEFAULT_SYSTEM_PROMPT = "問題の答えの選択肢番号を書いてください。"

    # text = (
    #     f"[問題]：{data['question']}"
    #     f"\n[選択肢]："
    #     f"[1. {data['choice0']},"
    #     f" 2. {data['choice1']},"
    #     f" 3. {data['choice2']},"
    #     f" 4. {data['choice3']},"
    #     f" 5. {data['choice4']}]"
    #     f"\n[答えの選択肢番号]："
    # )

    # prompt = (
    #     f"{bos_token}{B_INST} {B_SYS}\n"
    #     f"{DEFAULT_SYSTEM_PROMPT}\n"
    #     f"{E_SYS}\n"
    #     "\n"
    #     f"{text} {E_INST}{data['label']}"
    # )

    # prompt = (
    #     f"問題の答えの選択肢番号を書いてください。"
    #     f"\n[問題]：{data['question']}"
    #     f"\n[選択肢]："
    #     f"[1. {data['choice0']},"
    #     f" 2. {data['choice1']},"
    #     f" 3. {data['choice2']},"
    #     f" 4. {data['choice3']},"
    #     f" 5. {data['choice4']}]"
    #     f"\n[答えの選択肢番号]：{data['label']}"
    # )

    prompt = (
        f"[問題]:{data['question']}\n"
        f"[選択肢]:"
        f" 1. {data['choice0']},"
        f" 2. {data['choice1']},"
        f" 3. {data['choice2']},"
        f" 4. {data['choice3']},"
        f" 5. {data['choice4']}\n"
        f"[答えの番号]:{data['label'] + 1}"  # 選択肢の番号に合わせる
    )

    return prompt

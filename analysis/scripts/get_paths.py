"""Reads the W&B results and finds very low performing, and high performing models with similar number of parameters. Returns the path to those checkpoints on RC

We will use these to make predictions and study their outputs."""

import pandas as pd


def get_no_cross_attn_size(row, encdec):
    emb_layer = row["source_vocab_size"] * row["embedding_size"]
    pos_emb = row["max_target_length"] * row["embedding_size"]
#     qkv_size = row["embedding_size"] / row["source_attention_heads"]
#     self_attn = row["embedding_size"] * qkv_size * row["source_attention_heads"] * 3
    # qkv matrices + bias * 3 for q, k, v
    self_attn = ((row["embedding_size"] ** 2) + row["embedding_size"]) * 3
    # projection after attn_head concat
    self_attn += ((row["embedding_size"] ** 2) + row["embedding_size"])
    self_attn *= row[f"{encdec}_layers"]
    # FFN + bias
    ffn = (row["embedding_size"] * row["hidden_size"] * 2) + row["embedding_size"] + row["hidden_size"]
    ffn *= row[f"{encdec}_layers"]
    # layer norms
    lnorms = 4 * row["embedding_size"]
    lnorms *= row[f"{encdec}_layers"]
    out_layer = row["target_vocab_size"] * row["embedding_size"] + row["embedding_size"]
    return emb_layer + pos_emb + self_attn + ffn + lnorms + out_layer

def compute_num_params(row):
    if row["arch_name"] == "encoder-decoder":
        # enc-dec + cross attention in the dec
        cross = ((row["embedding_size"] ** 2) + row["embedding_size"]) * 3
        # projection after attn_head concat
        cross += ((row["embedding_size"] ** 2))
        cross *= row[f"decoder_layers"]
        return get_no_cross_attn_size(row, "encoder") + get_no_cross_attn_size(row, "decoder") + cross
    else:
        return get_no_cross_attn_size(row, "decoder")
    


def main(sweep_csv):
    df = pd.read_csv(sweep_csv)
    df["arch_name"] = "decoder-only"
    # Ignore single attention head case
    df = df.loc[df["source_attention_heads"] > 1]
    df["params"] = df.apply(compute_num_params, axis=1)
    df.loc[(df["params"] > (10**6) - 1) & (df["params"] < (10**7) + 1)]
    # "local_run_dir"
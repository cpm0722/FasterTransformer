import os, sys
import time, math
import configparser

import torch
import numpy as np
from transformers import T5Config, AutoTokenizer, T5ForConditionalGeneration

from utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder
from utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5


def load_op_t5_from_file(model_path, fp16=False):
    lib_path = "../../../build/lib/libth_t5.so"
    ckpt_config = configparser.ConfigParser()

    ckpt_config_path = os.path.join(model_path, 'config.ini')
    if os.path.isfile(ckpt_config_path):
        ckpt_config.read(ckpt_config_path)
    else:
        assert False, "[ERROR] This example only support loading model with FT format directly."

    weight_data_type = np.float32
    weight_data_type = {"fp16": np.float16, "fp32": np.float32}[ckpt_config.get("encoder", "weight_data_type")]
    relative_attention_max_distance = 128
    encoder_config = T5Config(vocab_size=ckpt_config.getint("encoder", "vocab_size"),
                              d_model=ckpt_config.getint("encoder", "d_model"),
                              d_kv=ckpt_config.getint("encoder", "d_kv"),
                              d_ff=ckpt_config.getint("encoder", "d_ff"),
                              num_layers=ckpt_config.getint("encoder", "num_layers"),
                              num_decoder_layers=ckpt_config.getint("encoder", "num_decoder_layers"),
                              num_heads=ckpt_config.getint("encoder", "num_heads"),
                              relative_attention_num_buckets=ckpt_config.getint(
                                  "encoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                              feed_forward_proj=ckpt_config.get("encoder", "feed_forward_proj"),
                              pad_token_id=ckpt_config.getint("encoder", "pad_token_id"),
                              eos_token_id=ckpt_config.getint("encoder", "eos_token_id"),
                              is_gated_act=ckpt_config.getboolean("encoder", "is_gated_act", fallback=0),
                              )
    decoder_config = T5Config(vocab_size=ckpt_config.getint("decoder", "vocab_size"),
                              d_model=ckpt_config.getint("decoder", "d_model"),
                              d_kv=ckpt_config.getint("decoder", "d_kv"),
                              d_ff=ckpt_config.getint("decoder", "d_ff"),
                              num_layers=ckpt_config.getint("decoder", "num_layers"),
                              num_decoder_layers=ckpt_config.getint("decoder", "num_decoder_layers"),
                              num_heads=ckpt_config.getint("decoder", "num_heads"),
                              relative_attention_num_buckets=ckpt_config.getint(
                                  "decoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                              feed_forward_proj=ckpt_config.get("decoder", "feed_forward_proj"),
                              pad_token_id=ckpt_config.getint("decoder", "pad_token_id"),
                              eos_token_id=ckpt_config.getint("decoder", "eos_token_id"),
                              decoder_start_token_id=ckpt_config.getint("decoder", "decoder_start_token_id"),
                              is_gated_act=ckpt_config.getboolean("decoder", "is_gated_act", fallback=0),
                              )
    assert decoder_config.feed_forward_proj == encoder_config.feed_forward_proj
    assert decoder_config.feed_forward_proj == encoder_config.feed_forward_proj

    t5_with_bias = ckpt_config.getboolean("structure", "t5_with_bias")
    use_gated_activation = encoder_config.is_gated_act
    position_embedding_type = 0 if ckpt_config.get('structure', 'position_embedding_type') == 'relative' else 1
    activation_type = encoder_config.feed_forward_proj

    tensor_para_size = 1
    pipeline_para_size = 1
    tie_word_embeddings = ckpt_config.getboolean("decoder", "tie_word_embeddings")
    ft_encoder_weight = FTT5EncoderWeight(
        encoder_config,
        tensor_para_size,
        pipeline_para_size,
        t5_with_bias=t5_with_bias,
        use_gated_activation=use_gated_activation,
        position_embedding_type=position_embedding_type,
        weight_data_type=weight_data_type
    )
    ft_decoding_weight = FTT5DecodingWeight(
        decoder_config,
        tensor_para_size,
        pipeline_para_size,
        t5_with_bias=t5_with_bias,
        use_gated_activation=use_gated_activation,
        position_embedding_type=position_embedding_type,
        weight_data_type=weight_data_type,
    )

    ft_encoder_weight.load_from_bin(model_path, "Megatron")
    ft_decoding_weight.load_from_bin(model_path, "Megatron")
    if fp16 is False:
        ft_encoder_weight.to_float()
        ft_decoding_weight.to_float()
    else:
        ft_encoder_weight.to_half()
        ft_decoding_weight.to_half()

    ft_encoder_weight.to_cuda()
    ft_decoding_weight.to_cuda()

    q_scaling = 1.0 / (math.sqrt(encoder_config.d_kv))
    remove_padding = True
    ft_encoder = FTT5Encoder(ft_encoder_weight.w, lib_path, encoder_config.num_heads,
                             encoder_config.d_kv, encoder_config.d_ff,
                             encoder_config.d_model, remove_padding, encoder_config.num_layers,
                             encoder_config.relative_attention_num_buckets,
                             0, # num_experts
                             [], # moe_layer_index
                             relative_attention_max_distance, False, q_scaling, tensor_para_size,
                             pipeline_para_size, t5_with_bias,
                             position_embedding_type, moe_k=0, activation_type=activation_type)

    ft_decoding = FTT5Decoding(ft_decoding_weight.w, lib_path,
                               decoder_config.num_heads, decoder_config.d_kv,
                               decoder_config.d_ff, encoder_config.d_model,
                               decoder_config.d_model, decoder_config.num_layers,
                               decoder_config.decoder_start_token_id, decoder_config.eos_token_id,
                               decoder_config.vocab_size, q_scaling,
                               decoder_config.relative_attention_num_buckets,
                               0, # num_experts
                               [], # moe_layer_index
                               max_distance=relative_attention_max_distance,
                               tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size,
                               t5_with_bias=t5_with_bias, position_embedding_type=position_embedding_type,
                               moe_k=0, activation_type=activation_type, tie_word_embeddings=tie_word_embeddings)

    ft_t5 = FTT5(ft_encoder, ft_decoding)

    return ft_t5


def main():
    torch.cuda.set_device(0)
    model = load_op_t5_from_file("t5-base-ft/1-gpu", fp16=True)
    tokenizer = AutoTokenizer.from_pretrained("./t5-base")

    input_text = ["hello, world!"] * 4
    input_token = tokenizer(input_text, return_tensors="pt", max_length=2048).to("cuda")

    output_ids, output_seq_len = model.forward(input_token=input_token,
                                               inputs_embeds=None,
                                               beam_size=3,
                                               max_seq_len=32,
                                               top_k=50,
                                               top_p=0.6,
                                               beam_search_diversity_rate=0.0,
                                               temperature=0.75,
                                               len_penalty=1.0,
                                               repetition_penalty=2.0,
                                               random_seed=0,
                                               is_return_output_log_probs=False,
                                               is_return_cum_log_probs=False,
                                               is_return_cross_attentions=False,
                                               )

    output_ids = output_ids.reshape(-1, output_ids.shape[-1])
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
    [print(text) for text in output_text]


if __name__ == "__main__":
    main()

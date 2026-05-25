#include "gguf.h"

namespace fastllm {
    std::vector <GGUFWeightReplaceRule> GetGGUFWeightReplaceRules(const std::string &arch) {
        static std::map <std::string, std::vector <GGUFWeightReplaceRule> > originalArchRulesDict = {
            {
                "default", 
                {
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k|v)\.(weight|bias))"),
                        "model.layers.$1.self_attn.$2_proj.$3"
                    ), // qkv
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k)_norm\.weight)"),
                        "model.layers.$1.self_attn.$2_norm.weight"
                    ), // qk norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_output\.(weight|bias))"),
                        "model.layers.$1.self_attn.o_proj.$2"
                    ), // o 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_(gate|up|down)\.(weight|bias))"),
                        "model.layers.$1.mlp.$2_proj.$3"
                    ), // mlp 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_norm\.weight)"),
                        "model.layers.$1.input_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_norm\.weight)"),
                        "model.layers.$1.post_attention_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(token_embd.weight)"),
                        "model.embed_tokens.weight", 
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(output.weight)"),
                        "lm_head.weight"
                    ), 
                    GGUFWeightReplaceRule (
                        std::regex(R"(output_norm.weight)"),
                        "model.norm.weight"
                    ),

                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_(gate|up|down)_exps.weight)"),
                        std::vector <std::string> ({"model.layers.$1.mlp.experts.", ".$2_proj.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ), // experts
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_(gate|up|down)_shexp.weight)"),
                        "model.layers.$1.mlp.shared_experts.$2_proj.weight"
                    ) // shared experts
                }
            },
            {
                "qwen3_moe", 
                {
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k|v)\.(weight|bias))"),
                        "model.layers.$1.self_attn.$2_proj.$3"
                    ), // qkv
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k)_norm\.weight)"),
                        "model.layers.$1.self_attn.$2_norm.weight"
                    ), // qk norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_output\.(weight|bias))"),
                        "model.layers.$1.self_attn.o_proj.$2"
                    ), // o 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_(gate|up|down)\.(weight|bias))"),
                        "model.layers.$1.mlp.$2_proj.$3"
                    ), // mlp 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_norm\.weight)"),
                        "model.layers.$1.input_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_norm\.weight)"),
                        "model.layers.$1.post_attention_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(token_embd.weight)"),
                        "model.embed_tokens.weight", 
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(output.weight)"),
                        "lm_head.weight"
                    ), 
                    GGUFWeightReplaceRule (
                        std::regex(R"(output_norm.weight)"),
                        "model.norm.weight"
                    ),

                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_gate_inp\.weight)"),
                        "model.layers.$1.mlp.gate.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_(gate|up|down)_exps.weight)"),
                        std::vector <std::string> ({"model.layers.$1.mlp.experts.", ".$2_proj.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ), // experts
                }
            },
            {
                "deepseek_v2", 
                {
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_q_(a|b)\.(weight|bias))"),
                        "model.layers.$1.self_attn.q_$2_proj.$3"
                    ), // q_a, q_b
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_kv_a_mqa\.(weight|bias))"),
                        "model.layers.$1.self_attn.kv_a_proj_with_mqa.$2"
                    ), // kv_a_mqa
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_kv_a_norm\.weight)"),
                        "model.layers.$1.self_attn.kv_a_layernorm.weight"
                    ), // kv_a_layernorm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_kv_b\.(weight|bias))"),
                        "model.layers.$1.self_attn.kv_b_proj.$2",
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP16
                    ), // kv_b
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_k_b\.(weight|bias))"),
                        "model.layers.$1.self_attn.kv_b_proj.$2__0",
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP16
                    ), // k_b, v_b，有时候这两个分开了
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_v_b\.(weight|bias))"),
                        "model.layers.$1.self_attn.kv_b_proj.$2__1",
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP16
                    ), // k_b, v_b，有时候这两个分开了
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_output\.(weight|bias))"),
                        "model.layers.$1.self_attn.o_proj.$2"
                    ), // o 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_norm\.weight)"),
                        "model.layers.$1.input_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_q_a_norm\.weight)"),
                        "model.layers.$1.self_attn.q_a_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_norm\.weight)"),
                        "model.layers.$1.post_attention_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(token_embd.weight)"),
                        "model.embed_tokens.weight", 
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(output.weight)"),
                        "lm_head.weight"
                    ), 
                    GGUFWeightReplaceRule (
                        std::regex(R"(output_norm.weight)"),
                        "model.norm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_(gate|up|down)\.(weight|bias))"),
                        "model.layers.$1.mlp.$2_proj.$3"
                    ), // mlp 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_gate_inp\.weight)"),
                        "model.layers.$1.mlp.gate.weight"
                    ), // gate weight
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.exp_probs_b\.bias)"),
                        "model.layers.$1.mlp.gate.e_score_correction_bias"
                    ), // gate bias
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_(gate|up|down)_exps.weight)"),
                        std::vector <std::string> ({"model.layers.$1.mlp.experts.", ".$2_proj.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ), // experts
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_(gate|up|down)_shexp.weight)"),
                        "model.layers.$1.mlp.shared_experts.$2_proj.weight"
                    ) // shared experts
                }
            },
            {
                "minimax_m2", 
                {
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k|v)\.(weight|bias))"),
                        "model.layers.$1.self_attn.$2_proj.$3"
                    ), // qkv
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k)_norm\.weight)"),
                        "model.layers.$1.self_attn.$2_norm.weight"
                    ), // qk norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_output\.(weight|bias))"),
                        "model.layers.$1.self_attn.o_proj.$2"
                    ), // o 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_norm\.weight)"),
                        "model.layers.$1.input_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_norm\.weight)"),
                        "model.layers.$1.post_attention_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(token_embd.weight)"),
                        "model.embed_tokens.weight", 
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(output.weight)"),
                        "lm_head.weight"
                    ), 
                    GGUFWeightReplaceRule (
                        std::regex(R"(output_norm.weight)"),
                        "model.norm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_gate_inp\.weight)"),
                        "model.layers.$1.block_sparse_moe.gate.weight"
                    ), // gate weight
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.exp_probs_b\.bias)"),
                        "model.layers.$1.block_sparse_moe.e_score_correction_bias"
                    ), // gate bias
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_gate_exps.weight)"),
                        std::vector <std::string> ({"model.layers.$1.block_sparse_moe.experts.", ".w1.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ), // experts gate -> w1
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_up_exps.weight)"),
                        std::vector <std::string> ({"model.layers.$1.block_sparse_moe.experts.", ".w3.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ), // experts up -> w3
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_down_exps.weight)"),
                        std::vector <std::string> ({"model.layers.$1.block_sparse_moe.experts.", ".w2.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ), // experts down -> w2
                }
            },
            {
                "glm4_moe", 
                {
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k|v)\.(weight|bias))"),
                        "model.layers.$1.self_attn.$2_proj.$3"
                    ), // qkv
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_output\.(weight|bias))"),
                        "model.layers.$1.self_attn.o_proj.$2"
                    ), // o 
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k)_norm\.weight)"),
                        "model.layers.$1.self_attn.$2_norm.weight"
                    ), // qk norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_(gate|up|down)\.(weight|bias))"),
                        "model.layers.$1.mlp.$2_proj.$3"
                    ), // mlp 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_gate_inp\.weight)"),
                        "model.layers.$1.mlp.gate.weight"
                    ), // gate weight
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.exp_probs_b\.bias)"),
                        "model.layers.$1.mlp.gate.e_score_correction_bias"
                    ), // gate bias
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_(gate|up|down)_exps.weight)"),
                        std::vector <std::string> ({"model.layers.$1.mlp.experts.", ".$2_proj.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ), // experts
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_(gate|up|down)_shexp.weight)"),
                        "model.layers.$1.mlp.shared_experts.$2_proj.weight"
                    ), // shared experts
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.post_attention_norm\.weight)"),
                        "model.layers.$1.post_attention_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_norm\.weight)"),
                        "model.layers.$1.input_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(token_embd.weight)"),
                        "model.embed_tokens.weight", 
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(output.weight)"),
                        "lm_head.weight"
                    ), 
                    GGUFWeightReplaceRule (
                        std::regex(R"(output_norm.weight)"),
                        "model.norm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(.*nextn.*)"),
                        "ignore"
                    ), // ignore
                }
            },
            {
                "gemma4",
                {
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_(q|k|v)\.(weight|bias))"),
                        "model.language_model.layers.$1.self_attn.$2_proj.$3"
                    ), // qkv
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_output\.(weight|bias))"),
                        "model.language_model.layers.$1.self_attn.o_proj.$2"
                    ), // o
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_(gate|up|down)\.(weight|bias))"),
                        "model.language_model.layers.$1.mlp.$2_proj.$3"
                    ), // mlp (dense / shared expert layers)
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_norm\.weight)"),
                        "model.language_model.layers.$1.input_layernorm.weight"
                    ), // attn_norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_norm\.weight)"),
                        "model.language_model.layers.$1.pre_feedforward_layernorm.weight"
                    ), // ffn_norm -> pre_feedforward
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.post_attention_norm\.weight)"),
                        "model.language_model.layers.$1.post_attention_layernorm.weight"
                    ), // post_attention_norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.post_ffw_norm\.weight)"),
                        "model.language_model.layers.$1.post_feedforward_layernorm.weight"
                    ), // post_ffw_norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_gate_inp\.weight)"),
                        "model.language_model.layers.$1.router.proj.weight"
                    ), // router
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_(gate|up)_exps\.weight)"),
                        "model.language_model.layers.$1.experts.gate_up_proj"
                    ), // fused gate_up experts
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_down_exps\.weight)"),
                        "model.language_model.layers.$1.experts.down_proj"
                    ), // fused down experts
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_(gate|up|down)_shexp\.weight)"),
                        "model.language_model.layers.$1.mlp.$2_proj.$3"
                    ), // shared expert (same name as dense MLP)
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.inp_gate\.weight)"),
                        "model.language_model.layers.$1.per_layer_input_gate.weight"
                    ), // per-layer input gate
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.proj\.weight)"),
                        "model.language_model.layers.$1.per_layer_projection.weight"
                    ), // per-layer projection
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.post_norm\.weight)"),
                        "model.language_model.layers.$1.post_per_layer_input_norm.weight"
                    ), // per-layer post norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.layer_output_scale\.weight)"),
                        "model.language_model.layers.$1.layer_scalar",
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ), // layer_scalar (force fp32)
                    GGUFWeightReplaceRule (
                        std::regex(R"(token_embd\.weight)"),
                        "model.language_model.embed_tokens.weight",
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ), // token embedding (force fp32)
                    GGUFWeightReplaceRule (
                        std::regex(R"(per_layer_token_embd\.weight)"),
                        "model.language_model.embed_tokens_per_layer.weight",
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ), // per-layer token embedding (force fp32)
                    GGUFWeightReplaceRule (
                        std::regex(R"(per_layer_model_proj\.weight)"),
                        "model.language_model.per_layer_model_projection.weight"
                    ), // per-layer model projection
                    GGUFWeightReplaceRule (
                        std::regex(R"(per_layer_proj_norm\.weight)"),
                        "model.language_model.per_layer_projection_norm.weight"
                    ), // per-layer projection norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(output_norm\.weight)"),
                        "model.language_model.norm.weight"
                    ), // output norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(rope_freqs\.weight)"),
                        "ignore"
                    ) // ignore rope_freqs (computed locally)
                }
            }
        };

        static std::map <std::string, std::vector <GGUFWeightReplaceRule> > archRulesDict = {
            {"qwen2", originalArchRulesDict["default"]},
            {"kimi_k2", originalArchRulesDict["deepseek_v2"]},
        };

        for (auto &it : originalArchRulesDict) {
            if (archRulesDict.find(it.first) == archRulesDict.end()) {
                archRulesDict[it.first] = it.second;
            }
        }

        if (archRulesDict.find(arch) != archRulesDict.end()) {
            return archRulesDict[arch];
        }

        printf("Warning: gguf arch %s not found, use default arch.\n", arch.c_str());
        return archRulesDict["default"];
    }
}

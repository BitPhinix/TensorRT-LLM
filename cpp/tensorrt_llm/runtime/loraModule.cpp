/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/runtime/loraModule.h"

namespace tensorrt_llm::runtime
{

std::vector<LoraModule> LoraModule::createLoraModules(std::vector<std::string> const& loraModuleNames,
    SizeType32 hiddenSize, SizeType32 mlpHiddenSize, SizeType32 numAttentionHeads, SizeType32 numKvAttentionHeads,
    SizeType32 attentionHeadSize, SizeType32 tpSize, SizeType32 numExperts,
    std::optional<SizeType32> kvALoraOutFeatures, std::optional<SizeType32> kvBLoraInFeatures,
    std::optional<SizeType32> kvBLoraOutFeatures, std::optional<SizeType32> wqBLoraInFeatures,
    std::optional<SizeType32> wqBLoraOutFeatures)
{
    auto const hidden = hiddenSize * tpSize;
    auto const mlpHidden = mlpHiddenSize * tpSize;
    auto const numHeads = numAttentionHeads * tpSize;
    auto const numKvHeads = numKvAttentionHeads * tpSize;
    auto const attnHeadSize = attentionHeadSize;

    std::vector<LoraModule> modules;

    for (auto const& moduleName : loraModuleNames)
    {
        auto const qOutSize = numHeads * attnHeadSize;
        auto const kvOutSize = numKvHeads * attnHeadSize;
        auto const t = toModuleType(moduleName);
        switch (t)
        {

        case ModuleType::kATTN_QKV:
        case ModuleType::kCROSS_ATTN_QKV:
            modules.emplace_back(t, hidden, (qOutSize + 2 * kvOutSize), false, true, -1, 0);
            break;
        case ModuleType::kATTN_Q:
        case ModuleType::kCROSS_ATTN_Q: modules.emplace_back(t, hidden, qOutSize, false, true, -1, 0); break;
        case ModuleType::kATTN_K:
        case ModuleType::kCROSS_ATTN_K:
        case ModuleType::kATTN_V:
        case ModuleType::kCROSS_ATTN_V: modules.emplace_back(t, hidden, kvOutSize, false, true, -1, 0); break;
        case ModuleType::kATTN_KV_A_MQA:
            if (!kvALoraOutFeatures.has_value())
            {
                throw std::runtime_error(
                    "attn_kv_a_mqa requires kv_lora_rank and qk_rope_head_dim in config to determine output dim");
            }
            modules.emplace_back(t, hidden, kvALoraOutFeatures.value(), false, true, -1, 0);
            break;
        case ModuleType::kATTN_KV_B_PROJ:
            if (!kvBLoraInFeatures.has_value() || !kvBLoraOutFeatures.has_value())
            {
                throw std::runtime_error(
                    "attn_kv_b_proj requires kv_lora_rank, qk_nope_head_dim, and v_head_dim in config");
            }
            modules.emplace_back(
                t, kvBLoraInFeatures.value(), kvBLoraOutFeatures.value(), false, true, -1, 0);
            break;
        case ModuleType::kATTN_WQ_B:
            if (!wqBLoraInFeatures.has_value() || !wqBLoraOutFeatures.has_value())
            {
                throw std::runtime_error(
                    "attn_wq_b requires q_lora_rank, qk_nope_head_dim, and qk_rope_head_dim in config");
            }
            modules.emplace_back(t, wqBLoraInFeatures.value(), wqBLoraOutFeatures.value(), false, true, -1, 0);
            break;
        case ModuleType::kATTN_DENSE:
        case ModuleType::kCROSS_ATTN_DENSE: modules.emplace_back(t, hidden, hidden, false, true, 1, -1); break;
        case ModuleType::kMLP_H_TO_4H: modules.emplace_back(t, hidden, mlpHidden, false, true, -1, 0); break;
        case ModuleType::kMLP_GATE: modules.emplace_back(t, hidden, mlpHidden, false, true, -1, 0); break;
        case ModuleType::kMLP_4H_TO_H: modules.emplace_back(t, mlpHidden, hidden, false, true, 1, -1); break;
        // TODO(TRTLLM-379): Support MOE LoRA weights
        case ModuleType::kMOE_H_TO_4H:
        case ModuleType::kMOE_GATE:
            modules.emplace_back(t, hidden * numExperts, mlpHidden * numExperts, false, true, -1, 0);
            break;
        case ModuleType::kMOE_4H_TO_H:
            modules.emplace_back(t, mlpHidden * numExperts, hidden * numExperts, false, true, 1, -1);
            break;
        case ModuleType::kMOE_ROUTER: modules.emplace_back(t, hidden, numExperts, false, true, -1, -1); break;
        case ModuleType::kMLP_ROUTER: modules.emplace_back(t, hidden, 1, false, true, -1, -1); break;
        case ModuleType::kMLP_GATE_UP: modules.emplace_back(t, hidden, 2 * mlpHidden, false, true, -1, 0); break;
        case ModuleType::kINVALID: throw std::runtime_error("Invalid LoRA module " + moduleName);
        }
    }
    return modules;
}
} // namespace tensorrt_llm::runtime

# VERL Debug Handoff

这份文档总结了本轮和 `verl` / `vllm_omni` 调试直接相关的信息，供下一个 coding agent 继续排查和收尾。

## 1. 当前调试目标

目标是在 `verl` 里跑通 diffusion 场景下的 `FlowGRPO`，并让 rollout 阶段返回 trainer 需要的 trajectory 数据：

- `all_latents`
- `all_log_probs`
- `all_timesteps`
- `prompt_embeds`
- `prompt_embeds_mask`
- `negative_prompt_embeds`
- `negative_prompt_embeds_mask`

最终要让 trainer 能完成：

- rollout
- reward / advantage 计算
- old log prob 计算
- actor loss 计算与参数更新

## 2. 已确认的执行路径

rollout phase 的 diffusion 请求，执行路径确认会走到 `vllm_omni.diffusion.diffusion_engine.DiffusionEngine.step()`。

主调用链如下：

```text
verl rollout http server
  -> AsyncOmni.generate(...)
  -> stage-0 diffusion worker
  -> AsyncOmniDiffusion.generate(...)
  -> DiffusionEngine.step(...)
```

因此：

- `DiffusionEngine.step()` 的 log 没出现时，不是没执行到
- 更可能是 logger level / handler 配置问题

## 3. 关于 logging 的结论

### 现象

最开始在 `DiffusionEngine.step()` 里手工加了很多 log，但 rollout 开始后看不到。

### 结论

原因不是执行路径错，而是 logger 没被正确放到 `INFO`。

`verl` 在 import 时会把全局 logging 设成 `WARNING`。  
而 `diffusion_engine.py` 里的 logger 没有显式 `setLevel(logging.INFO)`，所以 `logger.info(...)` 会被过滤。

### 建议

如果需要继续看 `diffusion_engine.py` 的 log，最直接的做法是：

```python
logger = init_logger(__name__)
logger.setLevel(logging.INFO)
```

或者把更上层 root logger 改成 `INFO`，但那样日志会非常吵。

## 4. 关于 custom pipeline 的结论

### 现象

尝试通过：

```bash
++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.custom_pipeline=...
```

让 rollout 走 `verl` 自己的 `QwenImagePipelineWithLogProb`。

但实际打印出的 pipeline 类型始终是原生：

```text
vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image.QwenImagePipeline
```

### 结论

当前本地安装的 `vllm_omni==0.14.0` 代码路径里，custom diffusion pipeline 支持并没有真正打通到底层 loader。

本轮曾尝试补几层 custom pipeline 透传，但用户后面决定放弃这条路，不再依赖 custom pipeline，而是直接改原生 pipeline。

### 当前策略

不要再继续围绕 custom pipeline 排查。

当前方案是：

- 直接修改原生 `vllm_omni` 的 `QwenImagePipeline`
- 让原生 pipeline 直接返回 FlowGRPO 所需字段

## 5. 已修改的关键文件

以下文件在本轮被实际修改过。

### A. `env/lib/python3.12/site-packages/vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`

这是本轮最核心的修改点。

已做的事情：

- 将原生 `QwenImagePipeline` 扩展到支持 FlowGRPO 所需的 trajectory 输出
- 在 pipeline 内部记录并返回：
  - `all_latents`
  - `all_log_probs`
  - `all_timesteps`
  - `prompt_embeds`
  - `prompt_embeds_mask`
  - `negative_prompt_embeds`
  - `negative_prompt_embeds_mask`
- 使用 `FlowMatchSDEDiscreteScheduler`
- 将 `req.prompts` 中已有的 embeddings / token ids 解析出来，而不是错误读取不存在的字段
- 从 `req.sampling_params.extra_args` 中读取：
  - `noise_level`
  - `sde_window_size`
  - `sde_window_range`
  - `sde_type`
- 修正空 `sde_window` 导致 `torch.stack([])` 报错的问题
- 修正过若干 shape / mask / request 字段访问错误

### B. `env/lib/python3.12/site-packages/vllm_omni/diffusion/data.py`

给原生 `DiffusionOutput` 增加了字段：

- `all_latents`
- `all_log_probs`
- `all_timesteps`
- `prompt_embeds`
- `prompt_embeds_mask`
- `negative_prompt_embeds`
- `negative_prompt_embeds_mask`

这样 pipeline 返回这些字段时，不会因为 dataclass 没定义而丢失。

### C. `verl/workers/rollout/vllm_rollout/vllm_omni_async_server.py`

这里修过两类问题。

1. earlier:

- 曾尝试加入 custom pipeline 透传逻辑
- 这部分现在不是主线，但改动仍在

2. currently important:

- 修正从 `multimodal_output` 里提取 trajectory 字段时的 shape 处理
- `_unwrap_first()` 现在会解包 leading batch=1 的 tensor
- `negative_prompt_embeds` 和 `negative_prompt_embeds_mask` 不再用错误的硬编码 `[0]`

这是为了解决 rollout server 把 `[1, seq, dim]` 错当成单样本 tensor 传给 trainer 的问题。

### D. `verl/experimental/agent_loop/agent_loop.py`

修正了 rollout 输出并入 batch 时的 embedding padding 逻辑。

之前的问题：

- `prompt_embeds` 还带着 batch=1 维
- postprocess 错把 `shape[0]` 当作 seq len 去 pad
- 导致不同样本最后变成不同长度
- 在 `DataProto.concat()` 阶段炸出类似：
  - `Expected size 1107 but got size 1109`

现在：

- 会先 squeeze 掉额外 batch=1 维
- 再按真实 seq len pad / truncate 到 `rollout.prompt_length`

### E. `verl/workers/config/model.py`

给 `DiffusersModelConfig` 新增了字段：

- `vae_scale_factor: int = 8`

原因：

- 后续为了让 trainer 和 rollout 用同一套 latent grid / timestep 设定，需要允许通过配置传 `actor_rollout_ref.model.vae_scale_factor=4`
- 否则 Hydra 实例化 `DiffusersModelConfig` 会报 unknown field

### F. `verl/utils/diffusers/utils.py`

修正训练侧 scheduler 初始化逻辑。

之前这里把：

```python
vae_scale_factor = 8
```

写死了。

现在会读取：

```python
vae_scale_factor = getattr(model_config, "vae_scale_factor", 8)
```

这是为了让训练侧 scheduler 的 timesteps / sigmas 与 rollout 侧保持一致。

## 6. 已定位并解决过的问题

### 问题 1：`custom pipeline` 没生效

现象：

- 实际 pipeline 类型仍是原生 `QwenImagePipeline`

结论：

- 当前本地 `vllm_omni` 版本底层未真正消费 custom diffusion pipeline 配置

处理：

- 放弃 custom pipeline 路线
- 直接修改原生 pipeline

### 问题 2：`DiffusionOutput` 没有 `all_latents`

现象：

```text
'DiffusionOutput' object has no attribute 'all_latents'
```

根因：

- 原生 `vllm_omni.diffusion.data.DiffusionOutput` 没有这些扩展字段

处理：

- 直接修改原生 `DiffusionOutput` dataclass

### 问题 3：request 里不存在 `prompt_ids`

现象：

```text
'OmniDiffusionRequest' object has no attribute 'prompt_ids'
```

根因：

- pipeline 误用了不存在的顶层 request 字段
- 实际数据在 `req.prompts` 里

处理：

- pipeline 改为从 `req.prompts` 提取 token ids / embeddings / masks

### 问题 4：`req.extra_args` 不存在

现象：

```text
'OmniDiffusionRequest' object has no attribute 'extra_args'
```

根因：

- 这些 rollout 参数其实在 `req.sampling_params.extra_args`

处理：

- pipeline 改为从 `req.sampling_params.extra_args` 读取 SDE 参数

### 问题 5：`all_log_probs` 为空导致 `torch.stack` 失败

现象：

```text
RuntimeError: stack expects a non-empty TensorList
```

根因：

- `sde_window` 没命中任何 step
- 特别是在 `num_inference_steps` 很小时更容易发生

处理：

- 对 `sde_window` 做了 clamp
- 对空列表做了 safe fallback

### 问题 6：`DataProto.concat` shape mismatch

现象：

```text
Expected size 1107 but got size 1109
```

根因：

- rollout server 返回的 `prompt_embeds` 仍是 `[1, seq, dim]`
- `agent_loop_postprocess` 误把 batch 维当作 seq len 去 pad

处理：

- server 端先 unwrap batch=1 tensor
- agent loop 端先 squeeze，再按真实 seq len pad/truncate

### 问题 7：`log_prob_micro_batch_size_per_gpu` 太大导致 ZeroDivisionError

现象：

```text
ZeroDivisionError: integer modulo by zero
```

根因：

- `prepare_micro_batches()` 用 `len(data) // micro_batch_size_per_gpu`
- 若 `micro_batch_size_per_gpu > len(data)`，则 chunks=0

结论：

- 小 batch 调试时，`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` 不能设太大

建议：

- 先用 `1` 或 `2`

### 问题 8：`4096 vs 1024` rotary embedding shape mismatch

现象：

```text
RuntimeError: The size of tensor a (4096) must match the size of tensor b (1024)
```

根因：

- rollout 实际使用的 latent grid 对应 `vae_scale_factor=4`
- trainer 侧却按默认 `8` 还原 `img_shapes`

处理：

- 给 `DiffusersModelConfig` 增加 `vae_scale_factor`
- 允许启动参数传：
  - `actor_rollout_ref.model.vae_scale_factor=4`
- 同时修正训练侧 scheduler 初始化使用真实 `vae_scale_factor`

### 问题 9：`index_for_timestep()` 找不到 timestep

现象：

```text
IndexError: index 0 is out of bounds for dimension 0 with size 0
```

报错位置在 scheduler 的：

- `index_for_timestep()`

根因：

- 训练侧 scheduler 初始化的 timestep 网格与 rollout 记录的 `all_timesteps` 不一致
- 本质上还是 `vae_scale_factor` 被写死成 8 导致 latent image seq len 不一致

处理：

- 已修改 `verl/utils/diffusers/utils.py`
- 训练侧 scheduler 现在会读取 `model_config.vae_scale_factor`

## 7. FlowGRPO 相关实现理解结论

这些是本轮已经确认的实现认知，下一个 agent 可以直接沿用。

### 1. rollout / old_log_prob / current log_prob 的关系

- rollout 先生成一批固定轨迹
- 然后旧策略对这批固定轨迹打分，得到 `old_log_prob`
- 再开始 actor update
- update 过程中当前 actor 会不断重算新的 `log_prob`

所以：

- `old_log_prob` 是 rollout 时策略快照的固定参考值
- `log_prob` 是当前正在训练的 actor 对相同步骤的重新评分

### 2. prompt group 和参数更新不是同一层概念

如果：

- 100 个 prompt
- 每个 prompt 10 个 request
- `sde_window_size=5`

那么 rollout 后有：

- 样本数 `B = 1000`
- trajectory steps `T = 5`
- 总 step-level transitions `B * T = 5000`

但：

- prompt group 只用于 reward/advantage 计算
- 参数更新按 batch / mini-batch / micro-batch 进行
- 不是“每个 prompt 更新一次”
- 也不是“每个 step 单独 optimizer.step 一次”

### 3. trainer 真正需要的数据

trainer 至少需要：

- `responses`
- `rm_scores`
- `index`
- `all_latents`
- `all_timesteps`
- `prompt_embeds`
- `prompt_embeds_mask`
- `negative_prompt_embeds`
- `negative_prompt_embeds_mask`

此外还会构造：

- `response_mask`
- `old_log_probs`
- `advantages`

## 8. 当前最重要的配置结论

对当前这条 Qwen-Image + FlowGRPO 路线，以下配置非常关键：

- `actor_rollout_ref.model.vae_scale_factor=4`

如果不加，trainer 侧很可能仍会按错误的 latent grid / scheduler timesteps 工作。

同时：

- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`
  在小 batch 调试时应先设小，例如 `1` 或 `2`

## 9. 当前仍需继续确认的事项

以下事项本轮没有完全闭环，需要下一个 agent 接着确认。

### A. 端到端是否已经完全跑通 old_log_prob + actor update

虽然已经修了：

- trajectory 暴露
- embedding shape
- `vae_scale_factor`
- scheduler timesteps

但还没有在本轮对话中看到“从 rollout 到 actor update 完整成功跑过一轮”的最终确认。

所以下一个 agent 应继续验证：

- old log prob 计算是否成功
- actor loss 是否成功
- optimizer step 是否成功

### B. `vllm_omni_async_server.py` 中 earlier custom pipeline 相关改动

这些改动现在不是主线，但还留在代码里。  
下一个 agent 需要判断：

- 是保留
- 还是在后续收尾时清理

### C. site-packages 改动和 repo 改动的边界

本轮直接改了大量：

- `env/lib/python3.12/site-packages/vllm_omni/...`

这些修改不是 repo 正常源码，而是当前环境内的安装包。  
下一个 agent 需要注意：

- 若要提交正式 patch，可能要把这些改动转回可维护的源码位置
- 若只是当前环境调试，继续沿着现有文件查即可

## 10. 建议下一个 agent 的优先排查顺序

1. 先重跑一次，确认 `vae_scale_factor=4` 和 scheduler patch 生效后的最新错误
2. 检查 old log prob 计算是否能完整通过
3. 检查 actor update 是否能走到 `optimizer.step()`
4. 若出现新 shape/timestep 问题，优先核对：
   - rollout side 的 `all_timesteps`
   - training side scheduler `timesteps`
   - `img_shapes`
   - `all_latents.shape`
5. 若端到端跑通，再收尾整理：
   - custom pipeline 相关残留改动
   - 日志插桩是否保留
   - site-packages 改动是否需要迁回 repo

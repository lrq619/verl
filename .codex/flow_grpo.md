# FlowGRPO

## 什么是 FlowGRPO？

FlowGRPO 可以看作是把 GRPO/PPO 风格的策略优化，从离散 token 生成扩展到 diffusion 轨迹上的一种方法。

在普通 LLM PPO 里：

- 状态是当前上下文
- 动作是输出一个 token
- `logprob` 是该 token 的对数概率

在 diffusion FlowGRPO 里：

- 状态是当前 latent
- 动作是从当前 latent 采样出下一个 latent
- `logprob` 是这一步连续状态转移的对数概率密度

因此，FlowGRPO 的训练对象不是 token 序列，而是 diffusion 轨迹。

### FlowGRPO训练基本流程

FlowGRPO 在 diffusion 场景下，可以理解为“先 rollout 出一批带 reward 的轨迹，再用 PPO/GRPO 的方式提升高 reward 轨迹中各个 step 的发生概率”。

它的训练流程可以按下面顺序理解：

1. 一个 prompt 会生成多个 request

假设 `rollout.n = 10`，那么一个 prompt 会生成 `10` 条 request，也就是 `10` 张图像样本。

每条 request：

- 对应一张最终图像
- 对应一条 diffusion trajectory

2. 每条 request 会记录一个局部 trajectory window

在当前实现里，FlowGRPO 并不总是记录完整 diffusion 全轨迹，而通常只记录 `sde_window` 内的一段 step。

如果：

- `sde_window_size = 5`

那么每条 request 会记录 `5` 个训练 step。

于是如果：

- 一共有 `100` 条 prompt
- 每条 prompt 生成 `10` 个 request
- 每条 request 记录 `5` 个 step

那么 rollout 后可供训练的 step-level transition 总数就是：

\[
100 \times 10 \times 5 = 5000
\]

也可以写成：

- 样本数 `B = 100 * 10 = 1000`
- 每条样本 trajectory 长度 `T = 5`
- 总 step 数 `B * T = 5000`

3. rollout 返回 reward 和 trajectory 数据

rollout 完成后，每条 request 不仅有最终图像和 reward，还会带回这条图像对应的训练轨迹数据，例如：

- `all_latents`
- `all_timesteps`
- `prompt_embeds`
- `prompt_embeds_mask`
- `negative_prompt_embeds`
- `negative_prompt_embeds_mask`

其中：

- `all_latents[:, step]`
  是第 `step` 个当前 latent 状态

- `all_latents[:, step + 1]`
  是 rollout 中真实发生的下一状态

- `all_timesteps[:, step]`
  是当前 step 的 diffusion timestep

4. 先按 prompt 分组计算 advantage

这里最重要的一点是：

- prompt group 是 reward/advantage 计算的基本单位
- 不是参数更新的基本单位

同一个 prompt 生成出来的 `10` 个 request 构成一个组。FlowGRPO 会在组内比较这些 request 的 reward，然后为每条 request 生成对应的 advantage。

因此：

- advantage 是“同一个 prompt 组内相对好不好”
- 不是“全局所有样本混在一起比”

5. 然后固定 rollout 数据，计算 old_log_prob

在开始 actor update 之前，会先用 rollout 时那份策略快照，对这批已经采样出的轨迹重新打分，得到：

- `old_log_probs`

这里的 `old_log_prob` 不是另外一个模型，而是 rollout 时旧策略对这些已发生轨迹 step 的 logprob。

6. actor update 时重新计算当前 log_prob

训练时会再次重放这些 trajectory step。对于每个 step，会根据：

- 当前 latent
- 下一 latent
- timestep
- prompt embeddings

重新计算当前 actor 对这一步状态转移的 logprob，也就是新的：

- `log_prob`

此时：

- `old_log_prob`
  是固定参考值

- `log_prob`
  是当前参数下重新算出来的值

7. 用 PPO/GRPO 风格的 loss 进行优化

FlowGRPO 的核心目标仍然是 PPO 风格的比值目标：

\[
r(\theta)=\exp(\log p_\theta - \log p_{\text{old}})
\]

然后用 `ratio` 和 `advantage` 共同决定 loss。

直觉上：

- 高 reward / 高 advantage 的轨迹，希望它们的 step logprob 变大
- 低 reward / 低 advantage 的轨迹，希望它们的 step logprob 变小

8. 参数更新不是按 prompt 单独进行，而是按 batch 进行

这是另一个容易混淆的点。

在 FlowGRPO 中：

- prompt group 用于算 advantage
- batch / mini-batch / micro-batch 用于做梯度更新

所以即使有 `100` 条 prompt，每条 prompt 有 `10` 个 request，训练时也不会“一个 prompt 更新一次权重”。

相反，trainer 会把所有 request 拼成一个 batch，然后：

- 先切 mini-batch
- 再切 micro-batch
- 在每个 micro-batch 内沿着 trajectory 的 step 维遍历
- 对这些 step 做 forward/backward
- 然后做一次 optimizer step

因此：

- `5000` 个 step-level 数据会参与 loss 计算
- 但不会发生 `5000` 次参数更新
- 真正的 optimizer step 次数由 mini-batch / micro-batch 数量决定

9. 一句话总结这条流程

FlowGRPO 的训练基本流程可以概括为：

- 一个 prompt 生成多个 request
- 每个 request 记录一段 trajectory step
- 同一个 prompt 组内比较 reward，得到 advantage
- 固定 rollout 数据并计算 old log prob
- 当前 actor 重放这些 step，得到新的 log prob
- 用 PPO/GRPO 风格 loss 优化 actor
- 按 batch 级别而不是按 prompt 级别更新模型参数

### 基本对象

- `latent`
  图像在隐空间中的表示，是一个固定 shape 的 tensor，不是最终像素图。

- `trajectory`
  一条图像生成路径，由多个 diffusion step 组成。每个 step 都对应一次状态转移，例如 `latent_t -> latent_{t-1}`。

- `logprob`
  不是词表上的离散概率，而是连续 latent 空间里，某一步从当前 latent 转移到下一个 latent 的对数概率密度。

- `reward`
  对最终图像的质量打分。

- `advantage`
  每条prompt相对于其组内平均reward的差值



### SDE window 的作用

FlowGRPO 通常不会拿完整 diffusion 全轨迹做 RL，而只拿一个窗口内的若干步。

- `num_inference_steps`
  决定整条 diffusion 轨迹总共有多少步。

- `sde_window_size`
  决定有多少步会被记录下来用于 FlowGRPO。

- `sde_window_range`
  决定这个窗口从哪一段范围里抽取。

因此：

- 一条图像生成会走完整个 diffusion 过程
- 但真正进入 FlowGRPO 训练的，往往只是其中一个局部 SDE window

## FlowGRPO 在 verl 上的实现

下面只讨论 diffusion 场景下的 FlowGRPO。

### trainer需要哪些数据

从 trainer 视角看，FlowGRPO 训练至少需要两类数据：

1. rollout outcome 数据

- `responses`
  最终生成的图像

- `rm_scores`
  reward model 或 reward function 对最终图像的打分

- `index`
  标识哪些样本属于同一个 prompt group，用于组内计算 advantage

2. trajectory 重放数据

- `all_latents`
  rollout 时记录下来的 latent trajectory

- `all_timesteps`
  每个 trajectory step 对应的 diffusion timestep

- `prompt_embeds`
- `prompt_embeds_mask`
- `negative_prompt_embeds`
- `negative_prompt_embeds_mask`

这些数据的作用是：

- `responses` 和 `rm_scores`
  用于算 reward 和 advantage

- `all_latents`、`all_timesteps`、`prompt_embeds`
  用于在训练时重放 rollout 中发生过的 diffusion step，并重新计算 logprob

此外，trainer 还会构造：

- `response_mask`

在 diffusion FlowGRPO 里，`response_mask` 实际上等价于“哪些 trajectory steps 是有效训练步”。它通常由 `all_latents` 或 `all_timesteps` 的时间维推导出来。

### vllm_omni是如何作为rollout引擎并入verl的

在 `verl` 里，`vllm_omni` 不是直接嵌进 trainer 主循环，而是作为 rollout server 接到 agent loop 体系里。

整体结构可以概括成：

```python
trainer
  -> async_rollout_manager.generate_sequences(...)
  -> agent loop
  -> vllm_omni async server
  -> AsyncOmni / diffusion engine
  -> image + multimodal trajectory data
```

其中：

- trainer 负责组织 prompt batch，并发起 rollout
- agent loop 负责把 prompt 变成 request
- `vllm_omni` 负责真正执行 diffusion generation
- rollout server 再把 diffusion 输出包装回 `verl` 可消费的数据结构

对 FlowGRPO 而言，最关键的不是仅仅拿到最终图片，而是让 rollout 引擎还能把中间 trajectory 数据一起带回来。

因此，`vllm_omni` 在这里承担两层职责：

1. 作为 rollout inference backend 生成图片
2. 作为 trajectory provider，把训练需要的 diffusion 中间量返回给 trainer

### 如何修改vllm_omni使其能够暴露trainer需要的数据

原始 `vllm_omni` 路径里，普通 diffusion 输出通常只关心最终图片，不一定会把 FlowGRPO 需要的中间数据完整暴露出来。

为了让 trainer 能训练 FlowGRPO，需要让 rollout 端额外暴露：

- `all_latents`
- `all_log_probs`
- `all_timesteps`
- `prompt_embeds`
- `prompt_embeds_mask`
- `negative_prompt_embeds`
- `negative_prompt_embeds_mask`

从实现上看，需要打通三层：

1. pipeline 层记录这些数据

伪代码可以理解为：

```python
for step in timesteps:
    noise_pred = transformer(...)
    latents, log_prob = scheduler.step(...)

    if step in sde_window:
        all_latents.append(latents)
        all_log_probs.append(log_prob)
        all_timesteps.append(step)

return DiffusionOutput(
    output=image,
    all_latents=stack(all_latents),
    all_log_probs=stack(all_log_probs),
    all_timesteps=stack(all_timesteps),
    prompt_embeds=prompt_embeds,
    prompt_embeds_mask=prompt_embeds_mask,
    negative_prompt_embeds=negative_prompt_embeds,
    negative_prompt_embeds_mask=negative_prompt_embeds_mask,
)
```

2. engine / request output 层把这些字段继续往上透传

如果 pipeline 返回了这些字段，但 engine 只包装最终图片，不把 `all_*` 和 `prompt_embeds` 带到输出对象里，trainer 仍然拿不到。

因此 diffusion engine 输出需要支持把这些字段挂到：

- diffusion output
- 或 multimodal output

总之要保证 rollout server 最终可见。

3. rollout server 层把这些字段转成 trainer batch 字段

伪代码如下：

```python
final_result = engine.generate(...)

extra_fields = {
    "all_latents": ...,
    "all_timesteps": ...,
    "prompt_embeds": ...,
    "prompt_embeds_mask": ...,
    "negative_prompt_embeds": ...,
    "negative_prompt_embeds_mask": ...,
}

return ImageOutput(
    image=image,
    log_probs=log_probs,
    extra_fields=extra_fields,
)
```

然后 agent loop / trainer 再把这些 `extra_fields` 并入训练 batch。

如果这条链任何一层断掉，就会出现典型问题：

- rollout 成功生成图片
- 但 trainer 侧 `all_latents` 或 `prompt_embeds` 是 `None`
- 随后 old log prob / actor loss 无法计算

### FlowGRPO训练数据在verl里如何组织

如果：

- 一共有 `100` 条 prompt
- 每条 prompt 生成 `10` 个 request
- `sde_window_size = 5`

那么 rollout 后会得到：

- 样本数 `B = 100 * 10 = 1000`
- 每条样本 trajectory 长度 `T = 5`
- 总 step-level transition 数 `B * T = 5000`

这里：

- 一个 request 对应一条样本
- 一条样本对应一条 trajectory
- trajectory 内部再包含多个 step

训练 batch 的关键 shape 可以理解为：

```python
all_latents:   [B, T + 1, ...]
all_timesteps: [B, T]
old_log_probs: [B, T]
advantages:    [B, T]
```

### FlowGRPO算法实现

FlowGRPO 在 `verl` 里的算法链可以概括成四步：

1. rollout 并计算 reward
2. 按 prompt group 计算 advantage
3. 计算 old log prob
4. actor 重放 trajectory 并计算当前 loss

#### 1. 按 prompt group 计算 advantage

FlowGRPO 不是把所有样本混在一起比较，而是同一个 prompt 生成出来的多个 request 构成一个 group。

伪代码如下：

```python
for each prompt_group:
    rewards = rewards_of_group
    baseline = group_mean_or_group_normalized_value(rewards)
    advantages = rewards - baseline
```

因此：

- prompt group 是 advantage 计算的基本单位
- 不是参数更新的基本单位

#### 2. 计算 old log prob

在 actor update 开始前，会先用 rollout 时的策略快照对这批固定轨迹重新打分，得到：

- `old_log_probs`

伪代码如下：

```python
old_log_probs = actor_rollout_policy.compute_log_prob(batch)
```

这里的关键点是：

- `old_log_prob` 是固定参考值
- 它代表 rollout 时旧策略对这批已发生 step 的打分

#### 3. 当前 actor 重新计算 log prob

训练时，actor 会对 rollout 中已经发生过的 trajectory step 重新前向：

```python
for step in trajectory:
    noise_pred = current_actor(latent_t, timestep_t, prompt_embeds, ...)
    log_prob = scheduler.sample_previous_step(
        sample=latent_t,
        prev_sample=latent_t_plus_1,
        model_output=noise_pred,
    )
```

这里得到的：

- `log_prob`

就是当前 actor 对相同步骤的重新评分。

#### 4. FlowGRPO 的 loss

FlowGRPO 的核心仍然是 PPO 风格的 ratio loss：

\[
r(\theta)=\exp(\log p_\theta - \log p_{\text{old}})
\]

伪代码如下：

```python
ratio = exp(log_prob - old_log_prob)
unclipped = -advantages * ratio
clipped = -advantages * clip(ratio, 1 - eps, 1 + eps)
pg_loss = mean(max(unclipped, clipped))
```

直觉上：

- advantage 高的轨迹，其 step logprob 会被鼓励增大
- advantage 低的轨迹，其 step logprob 会被鼓励减小

### 随后的更新是如何发生的

这部分很容易和 advantage 分组混淆。

需要区分两件事：

1. 按 prompt 分组的过程

- reward 比较
- advantage 计算

2. 按 batch 组织的过程

- actor forward/backward
- optimizer step

也就是说：

- prompt group 决定哪些样本相互比较来算 advantage
- 真正的参数更新不是“一个 prompt 更新一次”

后续训练更接近下面的伪代码：

```python
batch = all_requests_after_rollout

for mini_batch in split(batch):
    for micro_batch in split(mini_batch):
        zero_grad()
        for step in range(T):
            log_prob = current_actor.replay_step(micro_batch, step)
            loss = flow_grpo_loss(log_prob, old_log_prob, advantages)
            loss.backward()
        optimizer.step()
```

因此：

- 如果有 `5000` 个 step-level 数据
- 它们会进入 loss 计算
- 但不会对应 `5000` 次参数更新

真正的 optimizer step 次数由：

- mini-batch 数
- micro-batch 数
- PPO epoch 数

共同决定。

### 一句话总结这一层实现

可以用下面这句话概括 `verl` 中的 diffusion FlowGRPO：

- `vllm_omni` 负责 rollout 和暴露 trajectory
- trainer 负责收集 reward、构造 batch、按 prompt group 算 advantage
- actor 负责重放 trajectory、重新计算 logprob
- FlowGRPO loss 负责比较新旧 logprob，并把高 reward 轨迹的 step 概率往上推

# `update_weights_from_ipc` API Summary

## 结论

可以确认：按本次修改后的链路，`vllm-omni` 内部 diffusion worker 能真正执行 `update_weights_from_ipc`。

实际调用链如下：

1. 外部调用 `AsyncOmni.update_weights_from_ipc(...)`
2. `AsyncOmni` 向目标 diffusion stage 下发 `UPDATE_WEIGHTS_FROM_IPC` 任务
3. stage worker 调用 `AsyncOmniDiffusion.handle_update_weights_from_ipc_task(...)`
4. `AsyncOmniDiffusion` 调用 `DiffusionEngine.collective_rpc("update_weights_from_ipc", ...)`
5. diffusion multiprocess executor 广播 RPC 给 worker 进程
6. worker 通过 `WorkerWrapperBase.execute_method()` 把方法名解析到 `worker_extension_cls` 混入的方法
7. `vLLMOmniColocateWorkerExtension.update_weights_from_ipc(...)` 开始执行，并进入 `recv_comm_metadata`

## 重要实现约束

`AsyncOmniDiffusion` 内部不是用 `unique_reply_rank=None` 调 `collective_rpc`，而是：

```python
self.engine.collective_rpc(
    "update_weights_from_ipc",
    None,
    (),
    {
        "peft_config": peft_config,
        "base_sync_done": base_sync_done,
        "use_shm": use_shm,
    },
    0,
    True,
)
```

原因：

- 当前 `vllm_omni` 的 diffusion multiprocess executor 只有单 rank 回包通路适合直接等待
- 但 `update_weights_from_ipc` 必须在所有 diffusion worker 上执行
- 所以这里采用“所有 rank 执行，只有 rank0 回包”的模式：`unique_reply_rank=0, exec_all_ranks=True`

## 新增服务端 API

### 1. `AsyncOmni.update_weights_from_ipc`

签名：

```python
async def update_weights_from_ipc(
    self,
    stage_ids: list[int] | None = None,
    peft_config: dict | None = None,
    base_sync_done: bool = False,
    use_shm: bool = False,
) -> list[Any]
```

行为：

- `stage_ids=None` 时，默认选择所有 `stage_type == "diffusion"` 的 stage
- 如果显式传入了非 diffusion stage，会抛 `ValueError`
- 返回值是各目标 stage 的返回结果列表；单 stage 场景下通常是长度为 1 的列表

调用示例：

```python
results = await engine.update_weights_from_ipc(
    peft_config=peft_config,
    base_sync_done=base_sync_done,
    use_shm=use_shm,
)
```

如果只想对某个 diffusion stage 调用：

```python
results = await engine.update_weights_from_ipc(
    stage_ids=[0],
    peft_config=peft_config,
    base_sync_done=base_sync_done,
    use_shm=use_shm,
)
```

### 2. stage task envelope

`OmniStage` 下发到 stage worker 的 payload 结构：

```python
{
    "type": OmniStageTaskType.UPDATE_WEIGHTS_FROM_IPC,
    "peft_config": peft_config,
    "base_sync_done": base_sync_done,
    "use_shm": use_shm,
    "task_id": task_id,
}
```

stage worker 完成后回传：

```python
{
    "type": "rpc_result",
    "task_id": task_id,
    "stage_id": stage_id,
    "method": "update_weights_from_ipc",
    "result": result,
}
```

失败时：

```python
{
    "type": "rpc_result",
    "task_id": task_id,
    "stage_id": stage_id,
    "method": "update_weights_from_ipc",
    "error": "...",
}
```

## 给调用侧 agent 的建议

如果你在 `verl` 或 server adapter 侧接入这个能力，优先改成直接调用：

```python
await engine.update_weights_from_ipc(
    peft_config=...,
    base_sync_done=...,
    use_shm=...,
)
```

不要再假设 `AsyncOmni` 暴露了通用的 orchestrator 级 `collective_rpc("update_weights_from_ipc")`。

## 本次修改涉及文件

- `/workspace/vllm-omni/vllm_omni/entrypoints/async_omni.py`
- `/workspace/vllm-omni/vllm_omni/entrypoints/async_omni_diffusion.py`
- `/workspace/vllm-omni/vllm_omni/entrypoints/omni_stage.py`
- `/workspace/vllm-omni/vllm_omni/entrypoints/stage_utils.py`

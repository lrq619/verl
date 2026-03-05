# FlowGRPO Execution Flow (for `./run_flowgrpo.sh`)

## Scope
This diagram describes the actual runtime path triggered by:
- `/workspace/verl/run_flowgrpo.sh`
- `/workspace/verl/examples/flowgrpo_trainer/run_flowgrpo_nocfg.sh`
- `python3 -m verl.trainer.main_ppo --config-name ppo_diffusion_trainer ...`

It focuses on the FlowGRPO diffusion path (`RayFlowGRPOTrainer`), Ray actor orchestration, rollout/reward/agent-loop startup, and the per-step training loop.

## 1) Startup Sequence Diagram
```mermaid
sequenceDiagram
    autonumber
    participant U as User Shell
    participant S1 as run_flowgrpo.sh
    participant S2 as run_flowgrpo_nocfg.sh
    participant H as Hydra main_ppo.py
    participant R as Ray Driver
    participant T as TaskRunner (ray actor)
    participant P as RayFlowGRPOTrainer
    participant W as WorkerDict/RayWorkerGroup
    participant A as ActorRolloutRefWorker
    participant RM as RewardLoopManager
    participant AM as AgentLoopManager
    participant VS as vLLM/vLLM-Omni Servers

    U->>S1: ./run_flowgrpo.sh
    S1->>S1: set -euo pipefail, tee logs
    S1->>S2: bash examples/flowgrpo_trainer/run_flowgrpo_nocfg.sh + overrides
    S2->>H: python -m verl.trainer.main_ppo --config-name ppo_diffusion_trainer ...

    H->>H: auto_set_device + migrate_legacy_reward_impl
    H->>R: run_ppo(config)
    R->>R: ray.init(runtime_env from get_ppo_ray_runtime_env)
    R->>T: create remote TaskRunner and call run(config)

    T->>T: add_actor_rollout_worker(use_legacy_worker_impl=disable -> engine_workers.ActorRolloutRefWorker)
    T->>T: add_critic_worker / add_ref_policy_worker / validate_config
    T->>T: build tokenizer/processor + datasets/sampler
    T->>P: instantiate RayFlowGRPOTrainer

    P->>W: init_workers() -> create_resource_pool + create_colocated_worker_cls + spawn WorkerDict
    W->>A: init_model() (actor/ref/rollout/checkpoint-engine wiring)
    P->>RM: init RewardLoopManager (optional RewardModelManager + RewardLoopWorker[])
    P->>AM: init AgentLoopManager
    AM->>VS: init rollout replicas + launch vLLM/vLLM-Omni HTTP servers
    P->>P: CheckpointEngineManager.sleep_replicas()

    T->>P: fit()

    loop each train step
        P->>AM: generate_sequences()
        AM->>VS: inference via agent loop workers
        P->>RM: compute_rm_score() (if reward model enabled)
        P->>A: compute old_log_prob / ref_log_prob / values
        P->>P: compute advantage on driver
        P->>A: update_actor (and update_critic if enabled)
        P->>A: checkpoint_manager.update_weights()
        P->>VS: rollout servers sleep/wake as needed
    end
```

## 2) Control-Flow Graph (condensed)
```mermaid
flowchart TD
    A[run_flowgrpo.sh] --> B[run_flowgrpo_nocfg.sh]
    B --> C[main_ppo.main via Hydra]
    C --> D[run_ppo]
    D --> E[ray.init]
    E --> F[TaskRunner.run]

    F --> G[build role_worker_mapping + resource pools]
    F --> H[load tokenizer/processor + create RL datasets]
    F --> I{model_type == diffusion_model?}
    I -- yes --> J[RayFlowGRPOTrainer]
    I -- no --> K[RayPPOTrainer]

    J --> L[init_workers]
    L --> M[spawn WorkerDict in RayWorkerGroup]
    L --> N[ActorRolloutRefWorker.init_model]
    L --> O[RewardLoopManager init]
    L --> P[AgentLoopManager init]
    P --> Q[launch vLLM/vLLM-Omni rollout servers]

    J --> R[fit training loop]
    R --> S[generate_sequences]
    S --> T[reward + logprob + values + advantage]
    T --> U[update_actor/update_critic]
    U --> V[checkpoint_manager.update_weights]
    V --> R
```

## 3) Important Runtime Details (from this path)
- `run_flowgrpo.sh` only forwards overrides; real default bundle lives in `examples/flowgrpo_trainer/run_flowgrpo_nocfg.sh`.
- `run_flowgrpo_nocfg.sh` ends with `$@`, so values passed from `run_flowgrpo.sh` override earlier defaults if duplicated.
- `main_ppo.py` selects trainer class by `config.actor_rollout_ref.model.model_type`:
  - `diffusion_model` -> `RayFlowGRPOTrainer`
  - otherwise -> `RayPPOTrainer`
- Worker colocation is done by `create_colocated_worker_cls(...)`, which builds a `WorkerDict` wrapper and binds registered methods of inner workers.

## 4) Key Source Anchors
- Entry scripts:
  - `/workspace/verl/run_flowgrpo.sh`
  - `/workspace/verl/examples/flowgrpo_trainer/run_flowgrpo_nocfg.sh`
- Main entry and task runner:
  - `/workspace/verl/verl/trainer/main_ppo.py`
- FlowGRPO trainer lifecycle:
  - `/workspace/verl/verl/trainer/ppo/ray_diffusion_trainer.py`
- Colocated WorkerDict generation:
  - `/workspace/verl/verl/single_controller/ray/base.py`
- Engine worker init/update:
  - `/workspace/verl/verl/workers/engine_workers.py`
- Reward loop and reward model server manager:
  - `/workspace/verl/verl/experimental/reward_loop/reward_loop.py`
  - `/workspace/verl/verl/experimental/reward_loop/reward_model.py`
- Agent loop and rollout server launching:
  - `/workspace/verl/verl/experimental/agent_loop/agent_loop.py`
  - `/workspace/verl/verl/workers/rollout/replica.py`
  - `/workspace/verl/verl/workers/rollout/vllm_rollout/vllm_omni_async_server.py`

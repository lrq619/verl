# Docker + uv：共享可写 Python 环境（任何用户可修改）

## 目标
- 在镜像构建时使用 `uv` 创建一个共享虚拟环境。
- Dockerfile 后续所有 `pip install` / `uv pip install` 都安装到该虚拟环境。
- 容器运行后，任意用户都可以使用并修改该环境。

## 基线做法
1. 使用固定路径作为虚拟环境目录，例如 `/opt/venv`。
2. 通过 `ENV` 设置 `VIRTUAL_ENV` 并把其 `bin` 放到 `PATH` 最前，不依赖 `source activate`。
3. 优先使用 `uv pip --python /opt/venv/bin/python` 安装，避免解释器歧义。
4. 若必须“任何用户可写”，直接给 `/opt/venv` 设置全员可写权限。

## Dockerfile 参考片段
```dockerfile
FROM python:3.12-slim

ARG VENV=/opt/venv
ENV VIRTUAL_ENV=${VENV}
ENV PATH="${VENV}/bin:/root/.local/bin:${PATH}"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && curl -LsSf https://astral.sh/uv/install.sh | sh \
 && uv venv "${VENV}" \
 && chmod -R a+rwX "${VENV}"

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --python "${VENV}/bin/python" -r /tmp/requirements.txt

# Dockerfile 中后续安装默认进入 /opt/venv：
# RUN pip install <pkg>
# RUN uv pip install --python "${VENV}/bin/python" <pkg>
```

## 运行时说明
- `ENV PATH=...` 会写入镜像元数据，因此不同运行用户仍会优先使用 `/opt/venv/bin` 下的 `python` / `pip`。
- `chmod -R a+rwX /opt/venv` 可实现真正意义上的“任意用户可修改”。
- 该方案便利但会降低隔离性、安全性与可复现性，仅在确实需要共享可变环境时使用。

## 可选加固（若不强制要求“任意用户可写”）
- 优先采用“同组可写”（`chgrp` + `chmod g+rwX` + 目录 setgid），不要全员可写。
- 把基础环境设为只读，运行时扩展包安装到单独的可写目录。

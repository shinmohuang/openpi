# Multi-venv image that co-locates RoboCasa (Py3.10) + OpenPI policy server (Py3.11).
# Build:
#   docker build -f examples/robocasa/Dockerfile -t openpi-robocasa:multi .
# Run (mount repo for live code reload):
#   docker run --rm -it --network=host --gpus all \
#     -e DISPLAY=$DISPLAY -e MUJOCO_GL=${MUJOCO_GL:-egl} \
#     -v $PWD:/app openpi-robocasa:multi /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

ARG INSTALL_APT_DEPS=1
RUN if [ "$INSTALL_APT_DEPS" = "1" ]; then \
    set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git git-lfs build-essential clang \
        libosmesa6-dev libgl1-mesa-glx libglew-dev libglfw3-dev libgles2-mesa-dev \
        libglib2.0-0 libsm6 libxrender1 libxext6; \
    git lfs install --skip-smudge; \
    git config --global filter.lfs.required true; \
    git config --global filter.lfs.smudge "git-lfs smudge --skip -- %f"; \
    git config --global filter.lfs.process "git-lfs filter-process --skip"; \
    rm -rf /var/lib/apt/lists/*; \
    else \
    echo "Skipping apt dependency install (INSTALL_APT_DEPS=0)"; \
    fi

WORKDIR /app

ENV UV_LINK_MODE=copy
ENV OPENPI_VENV=/.venv-openpi
ENV ROBOCASA_VENV=/.venv-robocasa

# Create dedicated virtual environments.
RUN uv venv --python 3.11 $OPENPI_VENV
RUN uv venv --python 3.10 $ROBOCASA_VENV

# Install OpenPI policy dependencies into the 3.11 venv (no project install so host mount stays authoritative).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock,readonly \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,readonly \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml,readonly \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src,readonly \
    bash -c "UV_PROJECT_ENVIRONMENT=$OPENPI_VENV GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev"
# Minimal test dependencies imported at runtime (e.g., pytest from gemma implementation)
RUN uv pip install --python $OPENPI_VENV/bin/python pytest

# Bring simulator sources + deps into the 3.10 venv.
COPY ./examples/robocasa/requirements.txt /tmp/robocasa-requirements.txt
COPY ./third_party/robosuite /third_party/robosuite
COPY ./third_party/robocasa /third_party/robocasa

RUN set -eux; \
    $ROBOCASA_VENV/bin/python -m ensurepip --upgrade; \
    $ROBOCASA_VENV/bin/python -m pip install --no-cache-dir -U pip setuptools wheel; \
    $ROBOCASA_VENV/bin/python -m pip install --no-cache-dir -r /tmp/robocasa-requirements.txt; \
    $ROBOCASA_VENV/bin/python -m pip install --no-cache-dir -e /third_party/robosuite -e /third_party/robocasa; \
    uv --python $ROBOCASA_VENV/bin/python cache clean -a || true

# Copy RoboCasa macros to allow runtime overrides without importing at build time.
RUN set -eux; \
    MACROS_SRC=/third_party/robocasa/robocasa/macros.py; \
    MACROS_DST=/third_party/robocasa/robocasa/macros_private.py; \
    if [ -f "$MACROS_SRC" ]; then cp -f "$MACROS_SRC" "$MACROS_DST"; fi

# Environment wiring: expose both interpreters through PATH + helpful env hints.
ENV PATH=$ROBOCASA_VENV/bin:$OPENPI_VENV/bin:$PATH
ENV PYTHONPATH=/app/src:/app/packages/openpi-client/src:/third_party/robosuite:/third_party/robocasa
ENV MUJOCO_GL=egl
ENV OPENPI_ACTIVATE="source $OPENPI_VENV/bin/activate"
ENV ROBOCASA_ACTIVATE="source $ROBOCASA_VENV/bin/activate"

# Helper script to activate both envs is optional; default to shell for flexibility.
CMD ["/bin/bash"]

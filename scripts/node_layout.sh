# shellcheck shell=bash disable=SC2034
# (SC2034: the variables are read by the scripts that source this file.)
# Where everything lives on a lease node. Sourced by setup.sh on the node, and streamed ahead of
# node_ctl.sh by deploy.sh / stop_engines.sh / stop_scheduler.sh, so every script agrees on the layout.
# /mydata is the node's local scratch disk (not in fstab: setup.sh remounts it after a reboot).

MYDATA=/mydata
RB_ROOT=$MYDATA/rb
RB_HF_HOME=$MYDATA/hf                 # HF_HOME of every process: weights, tokenizers, encoders
ENGINE_VENV=$RB_ROOT/venv-engine      # official vLLM (its own torch); never shared with the scheduler
SCHED_VENV=$RB_ROOT/venv-scheduler    # this repository, pip install -e (its own torch)
TEST_VENV=$RB_ROOT/venv-test          # setup.sh --with-tests: the same pins plus the dev extras, for the test suite only
AP_DIR=$RB_ROOT/AvengersPro           # the pinned upstream AvengersPro checkout (scheduler only)
BR_DIR=$RB_ROOT/best-route-llm        # the pinned upstream BEST-Route checkout (scheduler only, D79)
RB_TIKTOKEN=$RB_ROOT/tiktoken         # TIKTOKEN_CACHE_DIR of the scheduler (upstream AvengersPro's cl100k_base)
RB_LOGS=$RB_ROOT/logs                 # engine and scheduler stdout/stderr, one file per launch
RB_RUN=$RB_ROOT/run                   # pid files, launch commands, the scheduler's copied inputs
RB_DOWNLOADS=$RB_ROOT/downloads       # the CUDA local-repo installer
CUDA_DIR=/usr/local/cuda-12.8
DRIVER_VERSION=570.86.10
VLLM_VERSION=0.18.1                   # == the pyproject `serve` extra and every engine args file (tests check)
# vLLM's VLLM_HTTP_TIMEOUT_KEEP_ALIVE for every engine (stock 5 s): above the clients' 60 s keep-alive (vLLM bench's stock
# connector), as v3 set its scheduler's server (600 s, "Was 5 — caused localhost RST race vs bench keepalive=60s");
# owner 09-29, D138
ENGINE_KEEP_ALIVE_S=600
# the environment every engine and the scheduler start with (env -i: nothing inherited from login files)
RB_PATH=$CUDA_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

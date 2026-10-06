"""Every fixed value of the vLLM-SR runs, each with its source (the owner's value, upstream's stock, or
v2/v3's)."""

# upstream, as released: a fresh deployment with the least configuration tuning
VSR_VERSION = "0.4.0"                                   # pip vllm-sr==0.4.0, released 2026-09-27
VSR_COMMIT = "6573123715e214a9eaa3ad23372ee5157411e519"  # tag v0.4.0
# the images the 0.4.0 CLI pulls (the router at its own release tag, Envoy fixed in its consts.py; the stores the
# Balance policy needs), pinned by the digests pulled on dev-gpu 2026-09-30
IMAGES = {
    "router": ("ghcr.io/vllm-project/semantic-router/vllm-sr:v0.4.0",
               "sha256:f6037c06796d6b6f678b2289b4fd618c6fa22b1487b276241f9382f9557ac690"),
    "envoy": ("envoyproxy/envoy:v1.35.3", "sha256:4d496918618a7ebd6c71ae8285e31ebff092f3a0a5ad642d50decf4a54eb2456"),
    "postgres": ("postgres:16-alpine", "sha256:721873c34ceb9f8d8fc265984940dc982404c105f19ad51be9fdc5970a6080ea"),
    "redis": ("redis:7-alpine", "sha256:858f009f9709ce576febc734aa78b8f6d624b82571f9ddb6bda4377c833b3499"),
}
CONTAINERS = ("vllm-sr-envoy-container", "vllm-sr-router-container", "vllm-sr-postgres", "vllm-sr-redis")
VOLUMES = ("vllm-sr-postgres-data", "vllm-sr-redis-data")
# the host variables upstream's CLI forwards into its containers (cli/commands/runtime_support.py, container_start.py):
# the stack runs under a clean environment, so none may appear in a container (live.container_problems)
FORWARDED_ENV = ("HF_ENDPOINT", "HF_TOKEN", "HF_HOME", "HF_HUB_CACHE", "VLLM_SR_DETERMINISTIC_EMBEDDINGS", "SR_LOG_LEVEL",
                 "SR_LOG_ENCODING", "SR_LOG_DEVELOPMENT", "SR_LOG_ADD_CALLER", "VLLM_SR_ENVOY_LOG_LEVEL", "ENVOY_LOG_LEVEL",
                 "VLLM_SR_STACK_NAME", "VLLM_SR_PLATFORM", "VLLM_SR_NOFILE_LIMIT", "VLLM_SR_DNS")
STACK_NETWORK = "vllm-sr-network"               # the CLI's container network (Envoy, the router)
REPLAY_TABLE = "public.router_replay_records"   # the only table a fresh 0.4.0 stack creates (probed 30 Sep)

# the policy: the built-in Mixture-of-Models Balance, as shipped, behind one entrypoint name
BUNDLE, RECIPE = "mom-v1", "balance"
ENTRYPOINT = "routebalance/mom-balance"
SHIPPED_INDICES = ("vllm-sr/general@1.0.0", "vllm-sr/reasoning@1.0.0")   # the pools' quality.index as shipped
POOLS = ("simple", "medium", "reasoning")

# the pools, by size as neighbouring pairs (v2's assignment)
LADDER = {
    "qwen": {"simple": ("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"),
             "medium": ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B"),
             "reasoning": ("Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-72B")},
    "gemma": {"simple": ("google/gemma-3-1b-it", "google/gemma-3-4b-it"),
              "medium": ("google/gemma-3-4b-it", "google/gemma-3-12b-it"),
              "reasoning": ("google/gemma-3-12b-it", "google/gemma-3-27b-it")},
}

# the quality evidence: our DeepEval score on the training split (tuning on training data is allowed,
# as for BEST-Route; "yes" to our DeepEval index), through v0.4.0's documented custom evaluation
BENCHMARK_ID = "routebalance/deepeval-train@1.0.0"
INDEX_ID = "routebalance/deepeval-quality@1.0.0"
PROFILE = "train-split"
JUDGE_KEY = "deepeval-llama3.1-8b-it_reference"         # the corpus's judge (the labels every estimator trains on)
# the training split's judged rows each family's means come from, pinned: the 28 Sep re-collection for Qwen;
# Gemma's is added when its re-collection exists. Any other file is refused, the evaluation corpora first
TRAINING_ROWS_SHA256 = {"qwen": "0a624b7f4e88d53e494dd1af00c445283d2c01450bc88a94901f1818e7dcec96"}

# the owner's values
LISTENER_TIMEOUT_S = 21600    # the route and stream-idle timeouts: the client's bound, vLLM bench's stock 6 h
MESSAGE_TIMEOUT_S = 3600      # ext_proc message_timeout: Envoy's maximum ([0s, 1h])
CIRCUIT_BREAKER = 200000      # connections, pending, requests on every cluster: v2's value
LISTENER_PORT = 8899          # the CLI's default routed listener
TEMPLATE_SHA256 = "3397dd85a2689e5fff01db569478a69de6c3a94cba064d936238ee9a2e107f36"   # templates/ (stack.tree_digest)
ENVOY_ADMIN_PORT = 9901       # the CLI template's admin port (read only, for the live checks)

# the request carries only the OpenAI chat fields v0.4.0 accepts; no stop (the stop string has no effect)
REQUEST_FIELDS = ("model", "messages", "temperature", "repetition_penalty", "frequency_penalty",
                  "max_completion_tokens", "stream", "stream_options")

# the test profile (dev-gpu) is marked, and a lease config carrying any mark is refused
# (the test setup is isolated from the evaluation)
TEST_MARK = "routebalance-test-profile"
# Docker's default bridge gateway (a Docker constant, not a host of ours): the dev-gpu host as the test stack
# reaches its engines; assembled from parts, as the hygiene gate asks of any private address in the tree
TEST_GATEWAY = ".".join(("172", "17", "0", "1"))
TEST_TOKENS = (TEST_MARK, "host.docker.internal", TEST_GATEWAY, "m-small", "m-large")

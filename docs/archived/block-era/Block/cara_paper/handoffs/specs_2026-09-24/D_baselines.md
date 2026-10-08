# Spec D — Baseline routers, dispatchers and filters (port spec for the from-scratch rebuild)

Status: read-only extraction, 2026-09-24. Nothing in any source tree was modified.

## 0. Sources, versions, conventions

| label used below | what | exact ref |
|---|---|---|
| **v3** (source of truth) | `/home/wd312/Code/llm/Block`, branch `v3-campaign` | HEAD `984ea18` (file mtimes Jul 26 – Aug 3 2026). Paths below are relative to `block/global_scheduler/cara/` unless absolute. |
| **master** | same repo, branch `master` (read via `git show`) | `f9d612e` (2026-07-25). Contains the June "v2" router code incl. `batch_scoring` (commit `ed4ee5b`, 2026-06-10) and AP off-loop (commit `51e2aa5`). |
| **RB-HEAD** | `/home/wd312/Code/llm/route-balance` (public repo, working tree intentionally empty) | `671db9f` (2026-06-18); paths `route_balance/global_scheduler/route_balance/...` |
| **v1exact** | submitted-paper tree copy | `cara_paper/handoffs/window_setup_work/v1exact_tree/tree/block/global_scheduler/cara/` |
| **AP upstream** | `/home/wd312/Code/llm/AvengersPro` (origin `https://github.com/ZhangYiqun018/AvengersPro.git`) | `dfb60b156c4b6011b89edfc8516aa16d8e4f4d38` (2025-12-11) **plus one uncommitted local edit** (see §6) |
| banked v3 cells | `/home/wd312/Code/llm/rb-exp/v3-exp/cells/` | S2 lease `asdwb-312083`, 2026-08-03/04 |

"cara" = internal codename of RouteBalance (RB). "Pipeline mode" = `cara_serve --scheduling pipeline`; every baseline in scope runs in pipeline mode. RB itself runs `--scheduling cara` (native batch scheduler) and does NOT use the router/dispatcher plugins at all (payloads with no `router`/`dispatch` key; `p1_pareto.sh:125`, `l6_retune.sh:37`, `s2_q1.sh:92-93`).

### 0.1 How a pipeline request executes in v3 (the glue the plugins depend on)

`cara_serve.py`:

1. `/v1/completions` (`:411`) → `prompt_text = request_json["prompt"]` (`:474`; the benchmark sends the Qwen chat-wrapped prompt, e.g. `<|im_start|>system\n...<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`). If `scheduling=="pipeline"` and router/dispatcher unset → `RuntimeError` (`:612-619`). Any exception in the request → HTTP 500 with `success:false` (`:637-656`); the cell's RULE-0 post gate then marks it INVALID.
2. `_select_via_pipeline` (`:2657-2837`):
   - `pool = _unique_model_pool(all_instances)` (`:2686`; def `:2644-2654`) = model names de-duplicated in **instance order**. Instances are built in `model_deployment.json` key order (`:3488-3547`), instance id = `f"{deployment_key}_{idx}"`. S2 order: `Qwen-2.5-14B_0..2, Qwen-2.5-3B_0..2, Qwen-2.5-72B_0..1, Qwen-2.5-7B_0..4` (13 instances) ⇒ `pool = ["Qwen/Qwen2.5-14B","Qwen/Qwen2.5-3B","Qwen/Qwen2.5-72B","Qwen/Qwen2.5-7B"]`.
   - `rreq = RouterRequest(prompt=_estimator_prompt(prompt_text), ...)` (`:2691-2700`). `_estimator_prompt` (`:107-120`) extracts the Qwen user turn with `<\|im_start\|>user\n(.*?)<\|im_end\|>` (DOTALL) or the Gemma user turn, then `.strip()`; raw prompts are only `.strip()`-ed. No switch (FAIRNESS comment `:2687-2690`: BR, AP and RB's KNN all consume the unwrapped text).
   - `decision = await router.choose_model(rreq, pool)` (`:2702`); `breakdown["router_ms"]` measured around it.
   - If `decision.model_name == "__ALL__"` (literal string compare, `:2709`) → `candidates = list(all_instances)`; else `candidates = [i for i in all_instances if i._model_name == chosen]`, empty → `ValueError` (`:2712-2717`).
   - `predicted_output = max_output_tokens` (`:2725`) — no length estimator on the pipeline path.
   - `needs_stats = getattr(dispatcher,"requires_stats",True) or (slo_filter is not None and getattr(slo_filter,"requires_stats",True))` (`:2732-2734`). If true: default path `stats_map = await _fetch_all_instance_stats(candidates)` (`:2748`, fail-loud); if `slo_defaults.stats_blind_absorb` is truthy: per-candidate `_get_instance_stats` gathered with `return_exceptions=True`, keep only non-exception truthy results (`:2742-2746`). Else `stats_map = {}` (`:2751`). **Stats are fetched only for `candidates`, not all instances.**
   - Filter block (`:2755-2819`) builds `InstanceState`s from `stats_map` and calls `slo_filter.get_eligible`; for `none` it is a no-op. The whole block is wrapped in `try/except Exception: logger.debug(...)` (`:2817-2818`).
   - `dreq = DispatchRequest(prompt=prompt_text  # NOT unwrapped, ...)` (`:2821-2829`); `inst = await dispatcher.choose_instance(candidates, dreq, stats_map=stats_map)` (`:2831-2833`). Dispatchers return the **Instance object**, not a `DispatchDecision`.
3. Router `decision.score` / `decision.reason` are **discarded** (only `.model_name` is used, `:2704`); the pipeline path writes no decision-log record. Per-request `scheduling_overhead_breakdown` (router_ms, estimator_ms, stats_fetch_ms, filter_ms, dispatcher_ms, total_sched_ms) is returned in the response (`:587`).
4. Stats fetch: `_get_instance_stats` (`:963-998`) GET `http://{ip}:{backend_port}/instance_stats`, `aiohttp.ClientTimeout(total=3)`, pooled session (`_get_http_session`, `:909-926`, limit 4096 / per-host 256, session total 30 s). Non-200 / exception / empty body → raise `InstanceStateUnavailable` (`:928-945`) unless `stats_blind_absorb` → return `{}`. `_fetch_all_instance_stats` (`:1001-1019`) gathers with `return_exceptions=True` and then **raises if any result is an exception or falsy** (so `{}` from the absorb-mode base function still raises here; absorb is honoured only by the pipeline's own branch at `:2742`).
5. `/instance_stats` payload (vLLM fork `vllm/v1/core/sched/scheduler.py:1281-1365` in `/home/wd312/Code/llm/vllm-cara`; fleet pin `AKafakA/vllm@2894a98722733c76cdb549c8b5e5068594f12bef`): `num_running, num_waiting, num_active_decode_seqs, decode_ctx_{mean,p50,p95,max}, total_decode_context_tokens, pending_prefill_tokens, pending_decode_tokens, kv_cache_utilization, kv_free_blocks, token_budget_per_iter, max_num_seqs, num_preempted, timestamp`. In-scope dispatchers read only `num_running` and `num_waiting`.
6. Runtime reconfiguration: `POST /v1/config` (`:764-887`). `"router"` → `create_router(data["router"], model_estimator=<global>, scoring_weights=<global dict>)`; on exception → HTTP 400 `{"where":"router","router_still_live":applied_router_config}` and the **previous router stays installed** (`:788-803`). `"dispatch"` → `create_dispatcher(data["dispatch"], scoring_weights=...)` (`:806-818`); `"filter"` → `create_filter(data["filter"])` (`:820-829`). `"slo_defaults"` is a dict **merge** (`:780-781`). Response echoes `router_config` and `router_sha` (`:881-885`); `_router_sha` = `sha256(json.dumps(cfg, sort_keys=True, separators=(",",":"))).hexdigest()[:16]` (`:707-720`). `GET /v1/router` (`:723`) and `GET /v1/config` (`:735-761`, includes `stats_blind_absorb`). Router and dispatcher objects are **re-constructed on every POST that carries the key** (so RNG/counters reset per cell).
7. The S2 scheduler config (`block/config/cara/s2/scheduler_config_s2.json`, md5 `002413dae0ae8c1534e90d31e7ecdd12`) has **no** `scheduling_pipeline` block, so router/dispatcher/filter are always installed by the runner's POST. `restart_v3` launches with `CUDA_VISIBLE_DEVICES=""` in every mode (`lib_serve_v3.sh:158`).

---

## 1. Base interfaces (exact v3 contracts)

### 1.1 `routers/base.py`

```python
@dataclass
class RouterRequest:                      # :23-35
    prompt: str
    num_prompt_tokens: int = 0
    max_output_tokens: int = 256
    budget_tokens: int = 256
    ttft_slo_ms: float = 10000.0
    tpot_slo_ms: float = 200.0
    quality_min: float = 0.0
    request_id: str = ""
    extra: dict = field(default_factory=dict)   # never populated by cara_serve

@dataclass
class RouterDecision:                     # :38-48
    model_name: str        # must be in model_pool, or the sentinel "__ALL__"
    score: float = 1.0
    reason: str = ""

class RouterBase(ABC):                    # :51-67
    @abstractmethod
    async def choose_model(self, req: RouterRequest, model_pool: List[str]) -> RouterDecision
```
Docstring contract (`:62-65`): must return a name in `model_pool`; "abstention" = most permissive model + `reason='abstain'` (no in-scope router uses it; v3 routers raise instead).

### 1.2 `dispatch/base.py`

```python
@dataclass
class DispatchRequest:                    # :14-24
    prompt: str
    num_prompt_tokens: int = 0
    max_output_tokens: int = 256
    predicted_output_tokens: Optional[int] = None
    ttft_slo_ms: float = 10000.0
    tpot_slo_ms: float = 200.0
    request_id: str = ""
    extra: dict = field(default_factory=dict)

@dataclass
class DispatchDecision:                   # :27-32  -- DEFINED BUT UNUSED by every dispatcher
    instance_id: str
    score: float = 1.0
    reason: str = ""

class DispatchBase(ABC):                  # :35-64
    requires_stats: bool = True           # :42  class attr; caller skips the stats fan-out when False
    @abstractmethod
    async def choose_instance(self, candidates: List[Any], req: DispatchRequest, *,
                              stats_map: Optional[Dict[str, dict]] = None) -> Any   # returns an Instance
```
Instance attributes the in-scope dispatchers touch: `_instance_id`, `_model_name`, `total_request` (cumulative count incremented in `cara_instance/Instance.py:141` at `query_instance` entry; **never decremented** — it is not an outstanding-request count).

### 1.3 `filters/base.py` + `filters/factory.py`

```python
@dataclass class SLOConstraints:  # :17-30  ttft_ms=10000.0, tpot_ms=200.0, budget_tokens=256, quality_min=0.0,
                                  #          budget_cost=0.0, num_prompt_tokens=0, max_output_tokens=256
@dataclass class InstanceState:   # :33-52  instance_id, model_name, gpu_type, predicted_ttft_ms=0.0,
                                  #          predicted_tpot_ms=0.0, predicted_e2e_ms=0.0, num_running=0, num_waiting=0,
                                  #          kv_cache_utilization=0.0, ttft_std_ms=0.0, tpot_std_ms=0.0,
                                  #          cost_per_input_token=0.0, cost_per_output_token=0.0
@dataclass class FilterResult:    # :55-64  instance_id, accepted, reason="", predicted_ttft_ms=0.0,
                                  #          predicted_tpot_ms=0.0, confidence=1.0
class SLOFilter(ABC):             # :67-98
    @abstractmethod
    def filter(self, slo, instances, predicted_output_tokens: int = 100) -> List[FilterResult]
    def get_eligible(self, slo, instances, predicted_output_tokens=100) -> List[InstanceState]
        # accepted_ids from filter(); returns instances (input order) whose id is accepted
```
`SLOFilter` has **no** `requires_stats` attribute (callers default it to True via `getattr`). `create_filter(cfg)` (`factory.py:19-69`): `type` default `"cara_tiered"` (so a filter dict without `type` silently becomes the CDF filter); `"none"` → `_NoFilter()`; unknown → `ValueError`. `_NoFilter` (`factory.py:72-88`): `requires_stats = False`; `filter()` returns `FilterResult(instance_id, accepted=True, reason="no_filter", predicted_ttft_ms=inst.predicted_ttft_ms, predicted_tpot_ms=inst.predicted_tpot_ms)` for every instance. The factory module imports all five real filters at import time (`factory.py:12-16`; `cara_cdf_filter` needs numpy).
`_NoFilter` is shared with the RB-native path (`cara_serve.py:1602-1636`), because RB payloads also post `"filter":{"type":"none"}`; the rebuild's `none` filter must satisfy both callers.

### 1.4 Factories

`routers/factory.py:create_router(router_config, *, model_estimator=None, scoring_weights=None)` (`:25-113`): `rtype = router_config.get("type", "cara_native")`. In-scope branches: `"cara_score"` → `CaraScoreRouter(model_estimator, scoring_weights)` (`:44-48`); `"cara_native"` → deprecated alias `CaraNativeRouter` (logs a warning) (`:49-54`); `"best_route_4way"` → `BestRoute4WayRouter(**router_config.get("kwargs", {}))` (`:86-90`); `"passthrough"` → `PassthroughRouter()` (kwargs ignored) (`:91-96`); `"avengers_pro"` → `AvengersProRouter(**router_config.get("kwargs", {}))` (`:103-105`). Unknown → `ValueError` (`:113`). Module-level imports: `cara_native`, `null_fixed`, `random_router` (`:20-22`); all others lazy.

`dispatch/factory.py:create_dispatcher(dispatch_config, *, scoring_weights=None, latency_predictor=None)` (`:24-72`): `dtype = dispatch_config.get("type", "cara_native")` (`:39`). `"round_robin"` → `RoundRobinDispatch()`; `"random"` → `RandomDispatch(seed=cfg.get("seed"))`; `"shortest_queue"` → `ShortestQueueDispatch(seed=cfg.get("seed"), blind_policy=cfg.get("blind_policy","counter"))`; `"po2_queue"` → `Po2QueueDispatch(seed=..., blind_policy=...)`. NB: dispatcher knobs are **top-level keys** of the dispatch dict (not under `kwargs`), unlike routers. Unknown → `ValueError`.

---

## 2. Per-component specification

### 2.1 `passthrough` (`routers/passthrough.py`)

- `PASSTHROUGH_SENTINEL = "__ALL__"` (`:16`). `choose_model`: empty pool → `ValueError("model_pool is empty")` (`:25-26`); else `RouterDecision("__ALL__", score=1.0/max(len(pool),1), reason="passthrough:no_routing")` (`:27-31`). No model, no RNG, no stats, never reads the prompt.
- Effect: dispatcher sees all 13 instances in deployment order. Factory kwargs: none (payload sends `"kwargs":{}` in `p1_pareto.sh:153`, none in `gemma_pt.sh:37-38`).
- The pipeline compares the literal `"__ALL__"` (`cara_serve.py:2709`), not the imported constant.

### 2.2 `best_route_4way` (`routers/best_route_4way.py`, v3)

Constructor `BestRoute4WayRouter(*, checkpoint_path="models/cara/best_route_4way_qwen", confidence_threshold=0.0, fallback_model=None, label_to_model=None, max_length=512, device=None)` (`:23-32`). Any other kwarg (e.g. master's `batch_scoring`) → `TypeError` → POST 400.

Construction, in order:
1. `_threshold=float(t)`, `_fallback=fallback_model`, `_max_length=int(max_length)` (`:33-35`).
2. **Raise #1** (`:42-49`): `t > 0 and fallback_model is None` → `RuntimeError("best_route_4way: confidence_threshold=... requires an explicit fallback_model ...")`.
3. **Raise #2** (`:51-58`): checkpoint dir missing or no `config.json` → `RuntimeError("BEST-Route-4way checkpoint not found ...")`. Path is relative to the process cwd (`cara_serve` is launched from the tree root).
4. `label_to_model`: explicit kwarg wins; else `training_results.json["label_to_model"]` (`:60-65`). **Raise #3** (`:66-70`): still falsy → `RuntimeError`. Keys normalised with `int(k)` (`:72-74`). Shipped mapping: `{0:"Qwen/Qwen2.5-3B",1:"Qwen/Qwen2.5-7B",2:"Qwen/Qwen2.5-14B",3:"Qwen/Qwen2.5-72B"}`.
5. `AutoTokenizer.from_pretrained(ckpt, use_fast=False)` (slow SentencePiece `DebertaV2Tokenizer`; `tokenizer.json` in the dir is ignored), `AutoModelForSequenceClassification.from_pretrained(ckpt).to(device or "cpu").eval()` (`:76-85`). No `torch.set_num_threads`, no dtype change (float32).

`choose_model(req, pool)` (`:115-159`), step by step:
1. Empty pool → `ValueError("model_pool is empty")` (`:120-121`).
2. `enc = tok(req.prompt, max_length=512, truncation=True, padding=False, return_tensors="pt").to(device)` (`:123-129`) — single prompt, no padding.
3. `with torch.no_grad(): logits = model(**enc).logits.squeeze(0); probs = softmax(logits, -1); top_idx = int(argmax(probs)); top_prob = float(probs[top_idx])` (`:130-134`). Runs **inline in the coroutine** (blocks the event loop; measured router_ms p50 ≈ 49–57 ms, p99 ≈ 175–195 ms in banked v3 cells, e.g. `s2_p1_l6/br4_t00_rr_l6_n3534.json`). Tie-break among equal probabilities = `torch.argmax` (first maximal index).
4. `chosen = label_to_model.get(top_idx)`; if `None` or `chosen not in pool` → `_resolve_fallback(pool, "pool_miss")` and return `RouterDecision(chosen, score=top_prob, reason=f"best_route_4way:pool_miss:label={top_idx}:fallback={chosen}")` (`:136-143`). **The pool-miss check precedes the threshold check**, and it applies even at t = 0.
5. If `t > 0 and top_prob < t` (strict `<`) → `_resolve_fallback(pool, "low_conf")`, return `reason=f"best_route_4way:low_conf:p={top_prob:.3f}<{t}:fallback={chosen}"` (`:146-153`).
6. Else `RouterDecision(chosen, score=top_prob, reason=f"best_route_4way:argmax_label={top_idx}:p={top_prob:.3f}")` (`:155-159`).

`_resolve_fallback(pool, why)` (`:100-113`): **Raise #4** fallback `None` → `RuntimeError("best_route_4way: {why} needs an explicit fallback_model ...")` (reachable only via pool_miss at t=0 with no fallback, since t>0 without fallback already failed at construction); **Raise #5** fallback not in live pool → `RuntimeError("... fallback_model=... is not in the live pool ... Refusing to substitute a different tier.")`. Never infers "smallest".

Knobs and ranges: `confidence_threshold` float; campaign values {0.0, 0.3, 0.5, 0.7} (`p1_pareto.sh:51`). With 4 classes `top_prob ≥ 0.25` always, so any t ≤ 0.25 is pure argmax. `fallback_model` must be an exact pool name; campaign value `"Qwen/Qwen2.5-3B"` (`payloads_v3.sh:14`). `max_length` 512 (= training, `train_best_route_wrapper.py:323`; also the model's `max_position_embeddings`). `device` default `"cpu"`. No randomness. Stats: none.

Reference behaviour (banked v3, λ6, rr; tier mix 3B/7B/14B/72B %): t0.0 50.3/21.9/15.5/12.3; t0.3 53.8/20.9/13.3/12.0; t0.5 91.1/0.2/0.6/8.1; t0.7 95.2/0.0/0.1/4.7. Per-prompt decisions are identical across all rates and dispatchers (3534/3534 prompts matched between `s2_p1_l6` and `s2_p1_l12`/`s2_p1_l30` cells and between rr and po2 cells).

### 2.3 `avengers_pro` (`routers/avengers_pro.py`, v3)

Upstream location: module constant `_AVENGERS_REPO = os.environ.get("CARA_AVENGERS_REPO", str(Path.home()/"AvengersPro"))` (`:37-40`) — on the fleet `~asdwb/AvengersPro`, shipped from the VPS checkout `/home/wd312/Code/llm/AvengersPro` (window log `2026-07-25_window_execution_log.md:211`). `_ensure_upstream_on_path()` (`:43-45`) inserts **the module constant** (not the `upstream_repo` kwarg) at `sys.path[0]`. Upstream modules are imported as top-level names: `simple_cluster_router`, `config`, `embedding_cache`.

Constructor `AvengersProRouter(*, artifact_dir="models/cara/avengers_pro_qwen", embedder_model="sentence-transformers/all-MiniLM-L6-v2", fallback_model=None, upstream_repo=_AVENGERS_REPO)` (`:84-108`); sets `self._loaded = self._try_load()`. `fallback_model` is accepted but **dead in v3** (only `_fallback_choice`, `:228-237`, reads it, and nothing calls `_fallback_choice` any more). The default `artifact_dir` does not exist in the tree; every campaign payload passes `models/cara/avengers_4model/avengers_pro_qwen_pw{pw}` (`p1_pareto.sh:166`).

`_try_load()` (`:110-226`) — returns False (router constructed, but every request will raise) on: any of `kmeans.joblib`, `cluster_rankings.json`, `config.json` missing (`:111-116`); `upstream_repo` path missing (`:119-124`); import of `joblib` / `simple_cluster_router` failing (`:127-135`); embedder init failing (`:188-192`); kmeans load failing (`:195-200`). It **propagates** (constructor raises → POST 400) on: malformed `config.json`/`cluster_rankings.json`, `SimpleClusterConfig(**cfg_kwargs)` validation failure, `normalizer.joblib` load failure. Steps:
1. `upstream = SimpleClusterRouter.__new__(SimpleClusterRouter)` — upstream `__init__` (which builds the HTTP `EmbeddingCache`) is bypassed (`:138`).
2. Build `SimpleClusterConfig` from `config.json` keys that are dataclass fields (`:141-145`); `setdefault("top_k",3)`, `setdefault("beta",5.0)`, `setdefault("max_router",1)` (`:147-149`); every remaining required field without default gets a type dummy ("" / 0 / 0.0 / False / None) (`:153-169`); `data_path` pointed at `config.json` if unset/missing, to pass upstream `__post_init__` existence check (`:172-173`). For the Qwen artifacts this yields `n_clusters=32, top_k=4` (from config.json), **`beta=5.0` (wrapper default; upstream default is 9.0)**, `max_router=1`, `max_workers=4`, `max_tokens=7500`, `performance_weight`/`cost_sensitivity` from config.json (validated in [0,1] by upstream, never read at route time).
3. `upstream.config = cfg`; `upstream.logger = logging.getLogger("SimpleClusterRouter")`; `upstream.tokenizer = tiktoken.get_encoding("cl100k_base")`, or `None` on any exception (`:176-185`).
4. `upstream.embedder = _LocalEmbedderShim(embedder_model)` (`:188-192`): `SentenceTransformer(model_name, device="cpu")` (`:61`); `get(query)` → `encode([query], normalize_embeddings=False)[0]` (`:69-78`); `get_embeddings(list)` kept for the older API (`:64-67`).
5. `kmeans_model = joblib.load(kmeans.joblib)`, `cluster_centers = kmeans_model.cluster_centers_` (`:195-200`); `normalizer = joblib.load(normalizer.joblib)` if present, else a fresh `Normalizer("l2")` fit on a zero row (`:202-210`; the Gemma artifact has no normalizer file).
6. `cluster_rankings = {int(k): {"ranking": list}}` from either a list or a `{"ranking": [...]}` dict; anything else → `{"ranking": []}` (`:212-221`). `available_models = config.json["models"]` (`:223`), i.e. `["Qwen/Qwen2.5-3B","Qwen/Qwen2.5-7B","Qwen/Qwen2.5-14B","Qwen/Qwen2.5-72B"]`.

`choose_model(req, pool)` (`:239-295`):
1. Empty pool → `ValueError` (`:244-245`).
2. **Raise #1** (`:246-259`): `not self._loaded` → `RuntimeError("avengers_pro: artifact/upstream failed to load ... Refusing to fall back ...")` — per request, not at construction.
3. `results = self._upstream.route_queries_batch([req.prompt])` **inline** (`:265-266`); any exception → **Raise #2** `RuntimeError("avengers_pro: upstream route_queries_batch failed (...)")` (`:267-272`). Measured router_ms p50 ≈ 17–20 ms, p99 ≈ 36 ms.
4. **Raise #3** (`:274-278`): `not results or not results[0]` → `RuntimeError("... upstream returned an empty ranking ...")`.
5. Return the first name of `results[0]` that is in `pool` with `score=1.0, reason=f"avengers_pro:upstream:rank=1:model={name}"` (`:281-288`). With `max_router=1`, `results[0]` has exactly one element.
6. **Raise #4** (`:290-295`): none in pool → `RuntimeError("... none of upstream's ranked models ... are in the live pool ...")`.

Upstream algorithm actually executed (`simple_cluster_router.py`):
- `route_queries_batch(queries)` (`:523-572`): `_generate_embeddings_concurrent(queries)` (`:344-359`) creates a **new `ThreadPoolExecutor(max_workers=4)` and a `tqdm` bar per call**, batch size `max(1, len//4)` = 1; each worker runs `_get_embedding_batch` (`:332-342`): `_truncate_text` (`:299-330`, tiktoken count; >7500 tokens → truncate to 7500; if `tokenizer` is None the `except` path logs an error and char-truncates at 22,500 chars) → `np.array(embedder.get(text))` → `normalizer.transform([e])[0]` (first normalisation). Back in `route_queries_batch`: `normalizer.transform(...)` again (second, idempotent for L2); `distances = 1 - E @ cluster_centers.T`; per query: `closest = np.argsort(d)[:top_k]` (top_k=4 of 32), `logits = -beta * d[closest]`, softmax; for each cluster in that order, for each model in `available_models` order, `score[m] += p * 1/(rank_in_cluster(m)+1)`; missing models get 0.0; `sorted(scores.items(), key=score, reverse=True)[:max_router]` — Python's stable sort, so **ties go to the earlier-inserted model = `available_models` order (3B first)**.
- The `pw` knob does not exist at runtime: it is baked into `cluster_rankings.json` at build time (`score = pw*norm_acc + cs*(1-norm_cost)`, `cs = 1-pw`, `norm_cost` from model size in B for the Qwen artifacts; see `v3_scripts/build_qwen_ap_realprice.sh:5-35`). The five Qwen artifacts share byte-identical `kmeans.joblib` (md5 `ea9dc3b9f597e63fd09f4976c17f09b3`) and `normalizer.joblib` (`8c814878f85ad424001dcdd1bf80a8f2`); only `cluster_rankings.json`/`config.json` differ. Valid `pw` = the five built artifacts {0.25, 0.39, 0.53, 0.7, 0.8} (`p1_pareto.sh:50`); Gemma: `avengers_pro_gemma_pw0.7` only (`w1b_cells.sh:48`).
- No randomness at route time (kmeans already fitted). No stats.

Reference behaviour (banked v3, λ6, rr; 3B/7B/14B/72B %): pw0.25 36.8/7.9/55.3/0.0; pw0.39 17.0/1.4/81.7/0.0; pw0.53 12.4/0.9/86.7/0.0; pw0.7 13.6/1.7/49.3/35.4; pw0.8 9.4/0.7/30.0/60.0. Per-prompt decisions identical across rates/dispatchers (3534/3534).

### 2.4 `cara_native` / `cara_score` router (`routers/cara_native.py`) — in scope by order, not executed by v3 payloads

`CaraScoreRouter(model_estimator=None, scoring_weights=None)` (`:29-42`); `CaraNativeRouter` = deprecated alias that logs a warning (`:98-108`). Weights: the **live global `scoring_weights` dict object** from cara_serve (mutated in place by later `POST scoring_weights`), or default `{"w_quality":0.6,"w_cost":0.3,"w_length":0.1}` if falsy. Keys read: `w_quality` (default 0.6), `w_cost` (0.3), `w_length` (0.1; absent from the global dict, so always 0.1 in practice).
`choose_model` (`:44-94`): empty pool → `ValueError`; **Raise** (`:52-63`, v3 FIX D30) `model_estimator is None` → `RuntimeError`. `estimates = model_estimator.estimate(req.prompt, req.budget_tokens)` — **synchronous, inline** (E1 applies). For each `name` in pool order with an estimate: `quality = est.score or 0.0`, `length = est.length_expected or 0.0`, `cost_proxy = size(name)/max(size(n) for n in pool)` where `size` parses `(\d+(?:\.\d+)?)[Bb]` from the name (default 1.0) (`:111-114`); `score = wq*quality - wc*cost_proxy - wl*(length/1024)`; strict `>` → first in pool order wins ties. **If no pool model has an estimate, it silently returns `_largest_model(pool)` with `score=-inf`** (`:87-88`) — a remaining silent fallback not covered by D30. reason `"cara_score:multi_obj"`. The estimator handle is back-filled after startup load (`cara_serve.py:3222-3223`).
`cara_native` is also the **factory default** for both a missing router type (`routers/factory.py:42`, startup `cara_serve.py:3155`) and a missing dispatch type (`dispatch/factory.py:39`). The `cara_native`/`cara_score` dispatcher (`dispatch/cara_native.py:23-72`): `score = w_latency*(running+waiting)/max(1,max_num_seqs) + w_balance*kv_cache_utilization`, argmin, first wins ties, missing stats → zeros. Not executed in v3.

### 2.5 `round_robin` (`dispatch/round_robin.py`)

`requires_stats = False` (`:8`) → pipeline skips the stats fan-out. State `_counters: Dict[str,int]` keyed by `candidates[0]._model_name` (`:24`; `"_default"` if missing). `idx = counters.get(key,0) % len(candidates); counters[key] = idx+1; return candidates[idx]` (`:25-27`). Empty → `ValueError`. With passthrough, the key is always `"Qwen/Qwen2.5-14B"` (first of 13) and one counter rotates over all 13. No RNG, no stats. **Warm-up offset:** each cell's dispatcher is freshly built at POST, then 5 `warmup_cell` requests (`lib_serve_v3.sh:210-216`) and the benchmark's initial single-prompt test (`benchmark_serving.py:632-676`) pass through it before the measured run, so the measured sequence starts at counter 6 for passthrough (verified: `pt_rr_l6` matches `order[(i+6)%13]` for 3486/3534 requests; residue = arrival reordering under concurrency).

### 2.6 `shortest_queue` (`dispatch/shortest_queue.py`, v3)

`__init__(seed=None, blind_policy="counter")` (`:19-25`): `_rng = random.Random(1 if seed is None else seed)`; `blind_policy ∉ {"counter","random"}` → `ValueError`; counters `blind_dispatches`, `partial_blind_dispatches` (not exposed by any endpoint). `requires_stats` = True (inherited). Empty candidates → `ValueError`.
Lookup `sighted_q(inst) = (stats_map or {}).get(inst._instance_id)` (`:31-32`).
- `"counter"` (default; the only policy used by grid/Q1 cells) (`:51-63`): `best=candidates[0]; best_q=inf`; per candidate in list order: `q = inst.total_request` if the entry is `None` (absent) else `s.get("num_running",0) + s.get("num_waiting",0)` (no int cast, no None guard); `if q < best_q` (strict) update. **Ties → first in candidate order.** Note: an entry that is present but `{}` gives q = 0.
- `"random"` (`:34-49`): sighted = candidates whose entry is **truthy**; none → `blind_dispatches += 1; return _rng.choice(candidates)`; some missing → `partial_blind_dispatches += 1`; min over sighted, strict `<`, first wins ties.
Under default fail-loud, `stats_map` always contains every candidate (or the request already raised), so both policies reduce to "min(running+waiting), first on tie" and the RNG is never used. The blind branches are reachable only when `stats_blind_absorb` is on (brfill cells).
Author ruling (commit `866a0f0`, 2026-08-03): first-on-equal-depth is the accepted choice for SQ.

### 2.7 `po2_queue` (`dispatch/po2_queue.py`, v3)

`__init__(seed=None, blind_policy="counter")` (`:22-28`), same RNG/validation/counters as SQ. `requires_stats` True.
`choose_instance` (`:30-74`): empty → `ValueError`. `pool = candidates if len(candidates) < 2 else self._rng.sample(candidates, 2)` (`:40`) — **consumes the RNG whenever there are ≥2 candidates** (every tier in S2 has ≥2 instances). `rng.sample` order is random, so "first sampled wins ties" is a random tie-break. Then the same `counter` (`:62-74`) or `random` (`:45-60`, `rng.choice(pool)` when both sampled are blind) logic as SQ, applied to the sampled pair. One RNG per dispatcher object shared across all tiers ⇒ the draw for request k depends on the global order of all dispatches since the POST (including warm-up and the benchmark test request).

### 2.8 `random` dispatch (`dispatch/random_dispatch.py`)

`RandomDispatch(seed=None)`: `_rng = random.Random(seed)` if seed not None **else the global `random` module** (`:10`). `choose_instance` → `_rng.choice(candidates)`; empty → `ValueError`. `requires_stats` is the inherited **True**, so the pipeline fetches (and fail-loud-validates) stats it never reads. The campaign payload (`p1_pareto.sh:153`, `pt_rand` cells) passes no seed ⇒ unseeded, not reproducible, and the global stream is shared with other `random.*` users in cara_serve (`:496,520,1664,2607,2896,2923`, none on the pipeline path).

### 2.9 Filter `none`

See §1.3. On the pipeline path it still executes the `InstanceState` build (`cara_serve.py:2764-2779`, reads `num_running`, `num_waiting`, `kv_cache_utilization` with `int()/float()` casts, `gpu_type` from `instance_meta`) and `get_eligible`; result is always all candidates. Because `_NoFilter.requires_stats=False`, `needs_stats` depends only on the dispatcher.

---

## 3. Version diffs (v3 vs master vs RB-HEAD vs v1exact)

### 3.1 `best_route_4way.py`

| aspect | v1exact (submitted) | master `f9d612e` / RB-HEAD `671db9f` | v3 `984ea18` |
|---|---|---|---|
| scoring execution | inline tokenizer+forward in coroutine | `choose_model` → `_score_one` (master `:119-128`). `batch_scoring=False` (default): `await loop.run_in_executor(None, _score_batch_sync, [prompt])` — **off-loop, batch of 1, `padding=True`**. `batch_scoring=True`: prompt queued with a Future; lazily-started `_drain_loop` (`:100-117`) sleeps `batch_window_ms` (10), takes ≤ `batch_max` (64), runs one padded forward in the default executor, fans results/exception out | inline (reverted to v1exact) (`:123-134`) |
| inline path | live | `_legacy_choose_model` (`:160-204`) exists but is **never called** | live |
| kwargs | ckpt, t, fallback, label_to_model, max_length, device | + `batch_scoring`, `batch_window_ms`, `batch_max` (`:32-34`) | v1exact set (batch kwargs → `TypeError`) |
| fallback | `self._fallback or model_pool[0]` (pool_miss and low_conf) — with nothing ever setting fallback this is the first-listed model (14B) (master `:141,148`) | same | raise-only via `_resolve_fallback` (`:100-113`); construction raise for t>0 w/o fallback (`:42-49`) |
| reason strings | `...:pool_miss:label=N:fallback`, `...:low_conf:p=..<t` | same | adds `:fallback={model}` suffix |
| default ckpt | `models/cara/...` | master `models/cara/...`; RB-HEAD `models/route_balance/best_route_4way_qwen` | `models/cara/best_route_4way_qwen` |

Documentation mismatch a porter must know: commit `ed4ee5b`'s message and defect-inventory D17 (`2026-07-30_V3_DEFECT_INVENTORY.md:103`) say "`batch_scoring=False` preserves the original serial/inline path". The master code does not do that: with False it already runs the forward off-loop per request (only `_legacy_choose_model`, which nothing calls, is inline). So master/June/RB-HEAD BEST-Route execution ≠ v1/v3 execution even at the default.

### 3.2 `avengers_pro.py`

| aspect | master / RB-HEAD | v3 |
|---|---|---|
| upstream call | `await loop.run_in_executor(None, upstream.route_queries_batch, [prompt])` (master `:263-267`, commit `51e2aa5`) — off-loop | inline `route_queries_batch([prompt])` (`:266`) |
| failure handling | four paths return `_fallback_choice(pool)` = `fallback_model` if in pool else **max-by-parsed-size (72B)**: not loaded (`:255-259`), upstream exception (`:268-273`), empty result (`:275-279`), top model not in pool (`:291-295`) | all four raise `RuntimeError` (D30) |
| repo discovery | master: first existing of `/users/asdwb/AvengersPro`, `/home/wd312/Code/llm/AvengersPro`, `~/AvengersPro` (`:41-48`). RB-HEAD: `~/AvengersPro` else the **literal string `"${HOME}/AvengersPro"`** (never expanded by Python) | env `CARA_AVENGERS_REPO`, default `~/AvengersPro` |
| shim | CPU pin + `get()` already present (commit `b79d6d4`) | same behaviour, D9 comments |
| tiktoken | injected after embedder, logs warning on failure | injected before embedder, silent on failure |
| default artifact_dir | master `models/cara/avengers_pro_qwen`; RB-HEAD `models/route_balance/avengers_pro_qwen` | `models/cara/avengers_pro_qwen` (non-existent; payloads always override) |

### 3.3 `shortest_queue.py` / `po2_queue.py` / stats semantics

| aspect | master / RB-HEAD | v3 |
|---|---|---|
| SQ constructor | no args, no RNG | `seed`, `blind_policy` (counter/random), blind counters |
| SQ algorithm | counter policy only (`total_request` when entry is `None`), first on tie | identical under `counter`; `random` policy added |
| Po2 | master: `seed` only, counter policy; **RB-HEAD has no `po2_queue` at all** (factory lists rr/random/sq/num_waiting/native/llumnix/block_style) | + `blind_policy`, counters |
| stats failure semantics feeding the dispatcher | `_get_instance_stats` returns `{}` on failure and `_fetch_all_instance_stats` stores `{}` for the instance (master/RB-HEAD serve `:696-724`) → SQ/Po2 read `{}` → q = 0 → **a failed instance looks idle and attracts traffic** | fail-loud raise (default); under absorb the failed instance is **absent** (→ `total_request` for counter, excluded for random) |
| dispatch history on v3 | `0498563` (random ties, counter removed) → `866a0f0` (first-on-tie restored) → `45ec7da` (reverted to running version) → `d6da961` (config-gated `blind_policy`, default `counter`) | grid/Q1 cells ran the pre-`d6da961` tree, which is behaviour-identical to `counter` under fail-loud; brfill ran `d6da961` |

### 3.4 `cara_native.py` router

master/RB-HEAD: estimator `None` → returns largest model (`reason="..._score:no_estimator:fallback_largest"`). v3: raises (D30). RB-HEAD renames to `route_balance_native.py` / `RouteBalanceScoreRouter`.

### 3.5 Pipeline glue

RB-HEAD `_select_via_pipeline` passes the **raw** `prompt_text` to routers; v3 passes `_estimator_prompt(prompt_text)` (chat unwrap + strip). Same `needs_stats` gating in both.

---

## 4. Executed vs dead (evidence)

S2 head chain (the scripts actually launched on the head, `v3-exp/head_scripts/`): `_run_p1_l{6,12,18,24,30}.sh` → `p1_pareto.sh`; `_run_q1.sh` → `s2_q1.sh`; `_run_qb_prod.sh`/`_run_qb_rerun_s1.sh` → `s2_q1_burst.sh`; `_run_brfill.sh` → `br_fill_rerun.sh`; `_run_acell.sh` → `a_cell.sh`; `_run_l6retune.sh` → `l6_retune.sh`. `s2_driver.sh:908,929` invokes `s2_q1.sh` and `p1_pareto.sh`.

**Executed (pipeline mode):**

| component | evidence |
|---|---|
| router `best_route_4way` (t ∈ {0.0,0.3,0.5,0.7}, fallback 3B) | `p1_pareto.sh:136-147` (payload `:141`); `br_fill_rerun.sh:47`; `a_cell.sh:35`; constants `payloads_v3.sh:14-15` |
| router `avengers_pro` (pw ∈ {0.25,0.39,0.53,0.7,0.8}) | `p1_pareto.sh:161-172` (payload `:166`) |
| router `passthrough` | `p1_pareto.sh:150-158` (payload `:153`) |
| dispatch `round_robin`, `shortest_queue`, `po2_queue` | `p1_pareto.sh:52` (`DISPATCHERS`), `:53` (`DISP_ONLY`) |
| dispatch `random` | `p1_pareto.sh:53` (`DISP_ONLY` = rr sq **random** po2; label `rand`, `:65`) |
| `blind_policy` = `random` (B) / `counter` (A) + `slo_defaults.stats_blind_absorb=true` | `br_fill_rerun.sh:47,68,82,92`; `a_cell.sh:24,35` (sq/po2 only) |
| filter `none` | every payload, e.g. `p1_pareto.sh:125,141,153,166` |

**Executed but not via plugins:** RB native (`--scheduling cara`, `assignment_strategy="scoring"` in all 16 occurrences in `v3_scripts/`). "simple-SQ" / "simple-llumnix" ablations are **not** the `dispatch/llumnix_minus.py` plugin: they are `slo_defaults.assignment_strategy ∈ {"shortest_queue","llumnix_minus"}` inside RB's batch scorer (`cara_serve.py:1577-1580, 1805-1830`), and no v3 script sets them (they appear only as planned items in `2026-09-01_NEXT_CAMPAIGN_HANDOFF.md:61-62`). They belong to the RB-native spec, not this one.

**Present in `v3_scripts/` but outside the S2 head chain** (earlier v3-window runners; same in-scope set only): `phase2_cells.sh:45-46` (AP/BR + po2), `w1b_cells.sh:48` (AP Gemma + po2), `gemma_pt.sh:37-38` (passthrough + sq/po2), `rebuttal_fix_phase1.sh:85-87` (BR/AP + po2), `br_fairness_ablation.sh:25` (BR + sq), `probe_v3.sh:40` (negative test: BR t=0.5 without fallback must 400), `gate_v3.sh:32` (`$BR4T5_SQ`). `payloads_v3.sh`'s exported `BR4*` variables are used only by `probe_v3.sh` and `gate_v3.sh`; `p1_pareto.sh` builds its own identical JSON inline at `:141`.

**Dead in v3 (no hit in `payloads_v3.sh`, `lib_post_config.sh`, head scripts or any `v3_scripts/*.sh`; grep of `"type"` values in `v3_scripts/*.sh *.py` returns only best_route_4way, avengers_pro, passthrough, shortest_queue, po2_queue, round_robin (via `$disp`), random (via `$disp`), none):** routers `omnirouter`, `qlm`, `routellm`, `vllm_sr`, `dual_pool`, `null_fixed`, `random`, `best_route`, `best_route_wrapper`, `cara_native`/`cara_score` (factory default only); dispatchers `llumnix_minus`, `num_waiting`, `predicted_latency`, `cara_native`/`cara_score` (factory default only); filters `cara_tiered`, `cara_hard_reject`, `slos_serve`, `polyserve`, `qlm`, `timebill`. `lib_post_config.sh` sets no types (it only posts and verifies the router digest / scoring weights). The 2026-09-01 handoff also excludes vLLM-SR and RouteLLM from the next campaign (`:70`).

---

## 5. Bug-fix markers that MUST survive (behaviour and rationale)

| where (v3) | marker | meaning to preserve |
|---|---|---|
| `routers/best_route_4way.py:37-49` | v3 FIX (construction) | t>0 without explicit `fallback_model` fails at construction (POST 400), not mid-cell. |
| `routers/best_route_4way.py:88-99, 100-113` | v3 FIX (D12′) | Never `model_pool[0]`, never inferred "smallest"; fallback must be stated and in the live pool, else raise. D12′: v1 t0.5/t0.7 sent 98.4%/99.9% of traffic to 14B (first-listed). Mentions D13/D25 (hard-coded-ladder bug class) as the reason not to infer from names. |
| `routers/best_route_4way.py:9-11` | stale docstring | Still says "default: smallest in pool" — false in every version; do not re-implement the docstring. |
| `routers/avengers_pro.py:54-61` | v3 FIX (D9, half 1) | Embedder pinned to CPU (CUDA auto-select caused contention / OOM → silent flood). |
| `routers/avengers_pro.py:70-78` | v3 FIX (D9, half 2) | Shim exposes `get(single)->1D` (current upstream API) and `get_embeddings(list)`. Missing `get` was the all-72B flood root cause. D9 also requires a scheduler restart after any router code change (inventory `:94`). |
| `routers/avengers_pro.py:150-173` | `[v1exact-v3 PORT 2026-07-27, author-approved]` | Dummy-fill required dataclass fields; `data_path` → existing file. Load-path only. |
| `routers/avengers_pro.py:178-185` | `[v1exact-v3 cont.]` | Inject `tiktoken cl100k_base` for upstream `_truncate_text`. |
| `routers/avengers_pro.py:247-259, 261-295` | v3 FIX (D30) | No route-to-largest on any failure path: not loaded, upstream exception, empty ranking, pool mismatch all raise. |
| `routers/cara_native.py:53-63` | v3 FIX (D30) | Estimator `None` raises. |
| `dispatch/factory.py:21,49` | `[v1exact-v2 PORT 2026-07-27]` | `po2_queue` registered (seeded Po2 over queue depth). |
| `dispatch/shortest_queue.py:3-10`, `po2_queue.py:7-13` | author-ordered A/B, 2026-08-03 | `blind_policy` counter(A, default) / random(B); ties first (SQ ruling, `:9`); blindness counted. |
| `cara_serve.py:210-216` (+`:796, 802, 881-885`) | v3 FIX | `applied_router_config` + `router_sha` echo; failed POST reports `router_still_live`. |
| `cara_serve.py:612-619` | guard | Pipeline mode with no router/dispatcher raises (no silent random). |
| `cara_serve.py:928-945, 947-960` | AUTHOR DECISION 2026-08-02 | `InstanceStateUnavailable`: telemetry failure raises, never `{}`. |
| `cara_serve.py:968-975` | ERA-MODE MASKING 2026-08-04 | `stats_blind_absorb` reverts the base fetch to `{}`-on-failure. |
| `cara_serve.py:1008-1011` | DO NOT MASK | `_fetch_all_instance_stats` re-raises every per-instance failure. |
| `cara_serve.py:2687-2690` | FAIRNESS | Routers receive the chat-unwrapped prompt, same as RB's estimator. |
| `cara_serve.py:2719-2724` | (perf fix) | No length estimator forward on the pipeline path; `predicted_output = max_output_tokens`. |
| `cara_serve.py:2728-2731` | (perf fix) | Skip stats fan-out when neither dispatcher nor filter needs it. |
| `cara_serve.py:2737-2741` | stats_blind_absorb, 2026-08-03 | Default OFF; pipeline-only. |
| `cara_serve.py:846-868` | RE-EMIT | EFFECTIVE_CONFIG re-logged on every POST (merge semantics of `slo_defaults` deliberately unchanged). |
| `v3_scripts/lib_post_config.sh:5-19, 37-111` | reconfigure contract | POST must be 2xx and the live router digest must equal the digest of what was sent. |
| `v3_scripts/lib_serve_v3.sh:240-252` | `router_needs_full_fleet` | BR/AP cells refused up front on a fleet missing any Qwen tier. |
| `v3_scripts/p1_pareto.sh:56-72` | exhaustive `dlabel` | Unknown dispatcher is a hard error (return code), not a default label. |
| inventory D17 (`2026-07-30_V3_DEFECT_INVENTORY.md:103`) | "VERIFIED — do not fix" | v3 keeps BEST-Route serial/inline; off-loop scoring is an owner decision (= E1). |

---

## 6. External dependencies

**AvengersPro upstream.** Checkout `/home/wd312/Code/llm/AvengersPro` @ `dfb60b156c4b6011b89edfc8516aa16d8e4f4d38`, with one **uncommitted** edit in `simple_cluster_router.py:30-33`: `from datasets import Dataset` wrapped in `try/except ImportError: Dataset = None`. The VPS checkout was shipped to the fleet as `~/AvengersPro` (window log `:211`); no commit/md5 of the fleet copy was recorded in the v3 manifests. Upstream imports needed at route time: `numpy`, `sklearn` (KMeans/Normalizer pickles), `joblib`, `tqdm`, `tiktoken`, `embedding_cache` (imported but unused because `__init__` is bypassed), `config.SimpleClusterConfig` (`config.py:15-72`). API used: `SimpleClusterRouter.__new__`, attributes `config, logger, tokenizer, embedder, kmeans_model, cluster_centers, normalizer, cluster_rankings, available_models`, method `route_queries_batch(list[str]) -> list[list[str]]` (plus its internal `_generate_embeddings_concurrent`, `_get_embedding_batch`, `_truncate_text`).

**AvengersPro artifacts** (`/home/wd312/Code/llm/Block/models/cara/avengers_4model/`, May 2 build, 32 clusters, top_k 4, MiniLM, judge `deepeval-llama3.1-8b-it_reference`, trained on `train_scored_filtered.jsonl`, 14,891 prompts, size-proxy costs):

| dir | cluster_rankings.json md5 | config.json md5 |
|---|---|---|
| `avengers_pro_qwen_pw0.25` | `aa4c467259b86d408c4dded03f85c44e` | `a95adae830eadb3bb01788cb916b8667` |
| `avengers_pro_qwen_pw0.39` | `44e9fa43d9668ea75686d4ee98d0d099` | `33974a0b0834a5e0b70d5bd6de006ca4` |
| `avengers_pro_qwen_pw0.53` | `e5e4b9879a726ab82be4da1e07ee9caf` | `85bccef5ebcea948ea219811aff8cdeb` |
| `avengers_pro_qwen_pw0.7` | `2b86ce2aab1c26383382caefaec84a27` | `6e9a1bd1fa2a16f1758ae7a1a81056af` |
| `avengers_pro_qwen_pw0.8` | `56638f312805597052b28ca348af265d` | `a4a297e85bc60c12e8513bbd7c97374d` |

All five: `kmeans.joblib` `ea9dc3b9f597e63fd09f4976c17f09b3` (109,443 B), `normalizer.joblib` `8c814878f85ad424001dcdd1bf80a8f2` (132 B). Gemma: `avengers_pro_gemma_pw0.7` (no normalizer). Do **not** confuse with the flat `models/cara/avengers_pro_qwen_pw*` (Apr 15, 2-model, 16 clusters) or `avengers_pro_gemma_pw0.7_LEGACY_costdominated` (quarantined; routed 100% to 4B). `s2_driver.sh:749-754` stages both `avengers_4model/` and the flat dirs.

**Embedder:** `sentence-transformers/all-MiniLM-L6-v2` (384-d), `device="cpu"`, loaded from HF hub/cache at construction; revision not pinned anywhere (no local cache on openclaw).

**BEST-Route checkpoint:** `/home/wd312/Code/llm/Block/models/cara/best_route_4way_qwen/` (552 MB; May 2 build; `best_val_acc 0.3923`, epoch 3, per-class acc 0.597/0.289/0.192/0.226). Files (md5): `model.safetensors` 567,604,704 B `01164d29c2db06b569897da2ee58b775`; `config.json` `ae4b77636d053baa8038849eea4763b8` (DebertaV2ForSequenceClassification, 6 layers, hidden 768, vocab 128,100, max_position 512, float32, saved with transformers 4.50.3); `spm.model` `1613fcbf3b82999c187b09c9db79b568`; `tokenizer.json` `a825fad6a7ab8ee30eaf4338fd1f5be9` (unused: `use_fast=False`); `tokenizer_config.json` `e1c5c36050e65902c8c1a6f7c7eef490`; `special_tokens_map.json` `6257cdebfe0656344ea0a28bb96a39d9`; `added_tokens.json` `88d1944d382d2aef86430924f6d65eeb`; `training_results.json` `6af015b31c202a43139fe5f51313f786`. Base model `microsoft/deberta-v3-small` (`train_best_route_wrapper.py:302-304`), trained with `padding=True, truncation=True, max_length=512`. Device CPU (default and `CUDA_VISIBLE_DEVICES=""`), max_length 512, slow tokenizer.

**Head-node runtime (S2, `v3-exp/env/`):** Python 3.10.12 system interpreter (no venv); torch 2.9.0, transformers 4.56.0, sentence-transformers 5.4.1, sentencepiece 0.2.2, scikit-learn 1.5.2, joblib 1.5.3, numpy 2.2.6, tiktoken 0.13.0, aiohttp 3.10.11. The scheduler runs on c4130 `.23`, **co-located with the 14B_0 vLLM worker** (`ENVIRONMENT.md` fleet table), so BR/AP CPU inference shares the host with a serving instance.

---

## 7. Proposed rebuild interfaces

Principle: the default configuration reproduces v3 execution exactly; E1 and E2 are separate, explicit switches whose default is the v3 behaviour until the owner rules. Every switch is echoed in `/v1/config` and in the per-cell provenance.

### 7.1 Core types (drop-in equivalents)

```python
@dataclass(frozen=True)
class RouterRequest:  # identical fields/defaults to v3 §1.1
@dataclass(frozen=True)
class RouterDecision:
    model_name: str            # pool member or ALL_INSTANCES = "__ALL__"
    score: float = 1.0
    reason: str = ""

class Router(ABC):
    type_name: ClassVar[str]
    @abstractmethod
    async def choose_model(self, req: RouterRequest, model_pool: Sequence[str]) -> RouterDecision: ...
    async def aclose(self) -> None: ...          # stops OffLoopScorer drain tasks / executors

@dataclass(frozen=True)
class DispatchRequest:  # identical to v3 §1.2
class Dispatcher(ABC):
    requires_stats: ClassVar[bool] = True        # keep v3 values: rr False, sq/po2/random True
    @abstractmethod
    async def choose_instance(self, candidates: Sequence[Instance], req: DispatchRequest, *,
                              stats_map: Mapping[str, dict]) -> Instance: ...
    def counters(self) -> dict: ...              # blind_dispatches etc., exposed in /v1/config

class SLOFilter(ABC): ...                        # identical to v3 §1.3; NoFilter.requires_stats=False
```
Factories keep v3 config shapes exactly (router knobs under `kwargs`, dispatcher knobs top-level) and the `_router_sha` digest, because `lib_post_config.sh` recomputes it client-side. Proposal needing owner approval (behaviour-visible only for malformed payloads): reject a missing `type` instead of defaulting to `cara_native`/`cara_tiered`.

### 7.2 E1 — `OffLoopScorer` (every router model forward off the event loop)

What E1 touches in v3: `routers/best_route_4way.py:123-134` (tokenize + forward); `routers/avengers_pro.py:265-266` (`route_queries_batch`, which internally also blocks on `future.result()` from its per-call ThreadPoolExecutor, `simple_cluster_router.py:344-359`); `routers/cara_native.py:65` (`model_estimator.estimate`, if cara_score is ever run). Reference pattern already used by RB: `loop.run_in_executor(None, ...)` on the default executor (`cara_serve.py:1311` TPOT boosters; `:2136`, `:2361` `estimate_batch`). The master implementation of the same idea: `best_route_4way.py@master:85-128`, `avengers_pro.py@master:263-267`.

```python
class OffLoopScorer(Generic[I, O]):
    """Hosts a synchronous batch function fn(list[I]) -> list[O]."""
    def __init__(self, fn: Callable[[list[I]], list[O]], *,
                 mode: Literal["inline", "executor", "microbatch"] = "inline",   # "inline" == v3 exact
                 executor: Optional[Executor] = None,     # None == loop default executor (RB parity)
                 window_ms: float = 10.0, max_batch: int = 64,  # microbatch only (master defaults)
                 name: str = ""): ...
    async def submit(self, item: I) -> O
        # inline:     return fn([item])[0]                      (blocks the loop; v3)
        # executor:   return (await loop.run_in_executor(executor, fn, [item]))[0]
        # microbatch: enqueue (item, Future); drain task sleeps window_ms, pops <= max_batch,
        #             runs fn(batch) in executor, resolves futures; exception -> set_exception on all
    def stats(self) -> dict   # calls, batches, max_batch_seen, queue_wait_ms p50/p99, fn_ms p50/p99
    async def aclose(self)
```
Router wiring: BR `fn = lambda prompts: [(top_idx, top_prob), ...]` using the v3 tokenizer call; for batch size 1 use `padding=False` exactly as v3 (a 1-row batch is unpadded either way, but keep the call identical); for microbatch use `padding=True` (master) and treat it as a different arm. AP `fn = upstream.route_queries_batch`. The fallback / raise logic (§2.2 steps 4-6, §2.3 steps 2-6) runs on the loop after `submit` returns, unchanged. Config: router kwarg `scoring_mode ∈ {inline, executor, microbatch}` (+ `batch_window_ms`, `batch_max`); because the router digest covers kwargs, cells with different modes get different `router_sha` automatically.

### 7.3 E2 — `StatsCache` (shared, asynchronously refreshed load view for SQ/Po2)

What E2 touches in v3: the `needs_stats` block `cara_serve.py:2732-2752` (both the fail-loud call `:2748` and the absorb branch `:2742-2746`); the fetch primitives `_get_instance_stats` (`:963-998`) and `_fetch_all_instance_stats` (`:1001-1019`), which the refresher would reuse; `_get_http_session` (`:909-926`); `breakdown["stats_fetch_ms"]`. Dispatchers themselves need no change (they consume a `stats_map` dict). Not touched unless the owner extends E2 to RB: RB-native's per-batch fetches `cara_serve.py:2235, 2442` (and `:3018`), and RB's in-batch optimistic accounting `_update_local_stats` (`:1097-1111`).

```python
@dataclass(frozen=True)
class StatsEntry:
    stats: Mapping[str, Any]      # raw /instance_stats body
    fetched_at: float             # time.monotonic() at response
    server_ts: Optional[float]    # body["timestamp"]
    error: Optional[str]          # last error if the latest refresh failed

class StatsReader(Protocol):
    async def get_many(self, instance_ids: Sequence[str]) -> dict[str, dict]:
        """Returns {iid: stats}. Missing/stale entries: raise InstanceStateUnavailable when
        on_missing='raise' (fail-loud, v3 default); omit the key when on_missing='absorb'
        (v3 stats_blind_absorb semantics, then the dispatcher's blind_policy decides)."""
    def snapshot(self) -> dict[str, StatsEntry]      # for provenance / staleness metrics

class PerRequestStats(StatsReader):   # v3-exact: fresh fan-out of exactly the requested ids per call,
                                      # 3 s timeout, same raise/absorb rules as cara_serve.py:2742-2748
class SharedStatsCache(StatsReader):  # E2: one background task refreshes ALL instances every
    def __init__(self, instances, fetch_one, *, refresh_ms: float, max_age_ms: float,
                 on_missing: Literal["raise", "absorb"], timeout_s: float = 3.0): ...
    async def start(self); async def stop(self)
```
Pipeline change: `stats_map = await stats_reader.get_many([i._instance_id for i in candidates])` when `needs_stats`; `breakdown["stats_fetch_ms"]` becomes read time and a new `breakdown["stats_age_ms_max"]` records staleness. Owner decisions that E2 cannot make silently: `refresh_ms`, `max_age_ms`, whether a stale entry counts as missing, whether the cache adds optimistic local increments between refreshes (RB does this within a batch; without it, SQ's first-on-tie plus a stale view sends every request in a refresh interval to the same instance), and whether RB-native reads the same cache.

---

## 8. Golden-test recipe (v3 vs rebuild, deterministic)

Harness: a test package that imports the v3 modules from a read-only checkout of `Block@984ea18` (`block.global_scheduler.cara.routers|dispatch|filters`) and the rebuild's modules side by side, in the same process. Pin `torch.set_num_threads(1)` (and the same value for both sides), `PYTHONHASHSEED=0`, CPU only, the §6 library versions. Record checkpoint/artifact md5s in the fixture header and fail if they differ.

Fixtures:
- `FakeInstance(_instance_id, _model_name, total_request=0)` × 13 in S2 order (§0.1), plus reduced topologies (drop 72B; single-instance tier).
- Prompts: all 3534 records of `data/cara/best-route-v3-test-3534-eval.jsonl` (md5 `15ed0e67ffa50e0d17955127c144a566` per cell provenance), wrapped with the Qwen chat template exactly as the banked `response_details[*].prompt` shows, then passed through a ported `_estimator_prompt`; also a direct test that `_estimator_prompt(banked_prompt)` equals the rebuild's unwrap for all 3534 banked prompts.
- Banked references: `/home/wd312/Code/llm/rb-exp/v3-exp/cells/s2_p1_l6/{br4_t00,br4_t03,br4_t05,br4_t07}_rr_l6_n3534.json`, `ap_pw{025,039,053,07,08}_rr_l6_n3534.json`, `pt_rr_l6_n3534.json`, with `(prompt → model, instance_id)` per request.

Router tests:
1. **BR replay (exact):** for each t ∈ {0.0,0.3,0.5,0.7} with fallback 3B, v3 router, rebuild router (each `scoring_mode`), and the banked cell must agree on `model_name` for 3534/3534 prompts; `reason` string equal; `|score_v3 − score_rebuild| ≤ 1e-6` for inline/executor. `microbatch` (padded) is compared separately: report argmax flips and any decision within 1e-4 of the threshold — expected ≈ 0 but not guaranteed bit-exact, so it is an arm, not a golden.
2. **BR raises:** t=0.5 without fallback → construction `RuntimeError` (mirrors `probe_v3.sh:40`); fallback `"Qwen/Qwen2.5-1B"` with a low-confidence prompt → per-request `RuntimeError`; pool without 72B and a prompt whose argmax is label 3 → pool_miss path returns the fallback (and raises when fallback is None at t=0); missing checkpoint / missing `label_to_model` → construction `RuntimeError`; empty pool → `ValueError`; unknown kwarg → `TypeError`.
3. **AP replay (exact):** each pw artifact; v3 vs rebuild vs banked, 3534/3534 `model_name` equality; also assert the rebuild calls upstream with `beta=5.0, top_k=4, max_router=1`. Tier-mix sanity against §2.3 reference mixes (a flood detector: fail if any tier exceeds its banked share by >2 pp, the twin-gate threshold in `twin_gate_v2.py`).
4. **AP raises:** artifact dir missing → constructor succeeds, every request raises (v3 semantics; a construction-time raise would be an owner-approved change and then this test flips); upstream monkeypatched to raise / return `[]` / return `["Qwen/Qwen2.5-72B"]` against a pool without 72B → each raises `RuntimeError`.
5. **passthrough:** returns `("__ALL__", 1/len(pool), "passthrough:no_routing")`; empty pool raises.
6. **cara_score:** fake estimator with fixed `(score, length_expected)` per model; compare decisions over a grid of weights; estimator None raises; estimator returning no pool models returns the largest model with `score=-inf` (v3 behaviour, §2.4).

Dispatcher tests (synthetic, seeded):
7. Generate a trace of 20,000 dispatch events with `numpy.random.default_rng(20260924)`: candidate set = one tier (sequence drawn from the BR t0.0 banked tier sequence) or all 13 (passthrough); per-event `stats_map` for candidates with `num_running, num_waiting ∈ [0,8]` (ties frequent); `total_request` counters incremented after each dispatch like `Instance.query_instance`. Absorb variant: drop each entry with p=0.2 (and occasionally all).
8. Feed the identical trace to v3 and rebuild instances built with identical config: `round_robin`; `shortest_queue` × {counter, random} × seed {None, 0, 7}; `po2_queue` × same; `random` with seed 123 (and seed None after `random.seed(k)` on both sides, run in separate processes because the global stream is shared). Assert the instance-id sequences are identical and the blind counters equal.
9. Prefix the trace with the 6 pre-measurement dispatches (5 warm-up + 1 benchmark test request) and check the RR offset against `pt_rr_l6` (≥98.5% positional match is the banked ceiling; the rebuild must equal the v3 module exactly on the synthetic trace).
10. Edge cases: 1 candidate (Po2 must not consume the RNG), 2 candidates (sample order decides ties), `{}` entries vs absent entries (counter: `{}` → 0; random: `{}` → blind).

Pipeline tests:
11. Port `_select_via_pipeline` against a fake instance set and a `StatsReader` stub: `PerRequestStats` must reproduce v3 exactly — stats fetched only for post-router candidates, RR fetches nothing, `random` fetches but ignores, any failure raises `InstanceStateUnavailable` unless absorb, absorb drops failed entries. `SharedStatsCache` with `refresh_ms=0`/read-through must reproduce the same decisions on the synthetic trace; with a real refresh interval it is the E2 arm and is only characterised (staleness distribution, blind counts), not goldened.
12. E1 loop-lag test: a 1 ms heartbeat coroutine while 64 concurrent BR `choose_model` calls run; inline shows lag ≈ N × forward time, executor/microbatch keep lag < 5 ms; decisions identical to test 1.
13. `/v1/config` contract: POST BR t=0.5 without fallback returns 400 with `router_still_live` and the previous digest; `router_sha` equals the client-side digest in `lib_post_config.sh:25-35` for every §4 payload.

---

## 9. Hazards for a porter

1. **Order is behaviour.** SQ breaks ties by candidate order, BR/AP pools and AP tie-breaks follow list order, RR walks list order. Instance order comes from `model_deployment.json` key order (14B, 3B, 72B, 7B). Sorting instances or pools in the rebuild changes decisions.
2. **RNG streams.** SQ/Po2 default to `Random(1)`, re-seeded on every POST that carries `dispatch`; Po2 consumes two draws per multi-candidate dispatch; warm-up and the benchmark's test request advance RNG and RR counters before measurement. `random` dispatch with no seed uses the unseeded global stream (pt_rand cells are not reproducible).
3. **`total_request` is cumulative**, not in-flight; the counter blind policy therefore favours the least-used-since-start instance.
4. **`{}` vs absent.** master/RB-HEAD map a failed fetch to `{}` → SQ/Po2 read q=0 → the failed instance attracts traffic. v3 fails loudly, or under absorb omits the entry. Do not port master's stats semantics.
5. **BEST-Route determinism depends on the slow tokenizer** (`use_fast=False`), max_length 512, no padding, float32 CPU, and thread count; transformers on the head was 4.56.0 while the checkpoint was saved with 4.50.3.
6. **BR check order:** pool-miss before threshold; strict `<` threshold; for 4 classes t ≤ 0.25 = argmax.
7. **Raise timing differs by router:** BR raises at construction (400, previous router stays live); AP defers most load failures to per-request raises (HTTP 500 per request, cell INVALID after it has run). The rebuild should keep this unless the owner approves moving AP's checks to construction.
8. **AP load quirks:** `beta=5.0` overrides upstream's 9.0; `top_k` comes from the artifact config (4), not upstream's 1; `tiktoken.get_encoding` may need network/cache (on failure truncation silently switches to 22,500 chars and logs an error per request); upstream creates a thread pool and a tqdm bar per call (stderr noise, per-call cost); normalisation is applied twice; `_ensure_upstream_on_path` uses the env/module path even when `upstream_repo` is passed; upstream is imported as top-level `config`/`embedding_cache`, which can collide with other modules named `config`; the fleet copy's commit and the `datasets` patch are unrecorded; the artifact path must be the `avengers_4model/` one; MiniLM revision is unpinned.
9. **Prompt input:** routers get `_estimator_prompt(prompt)` (unwrapped and stripped); dispatchers get the raw prompt. RB-HEAD passed the raw prompt to routers.
10. **Config lifecycle:** router, then dispatch, then filter are applied in that order; a dispatch failure after a successful router swap leaves a partially applied config (400 with `where:"dispatch"`). `slo_defaults` merges, so `stats_blind_absorb:true` persists for the rest of a scheduler session unless reset (brfill restarts the scheduler per cell; the rebuild's runner must not rely on that implicitly).
11. **`random` dispatch has `requires_stats=True`**: it pays the stats fan-out and inherits fail-loud without reading the stats. Changing that flag changes the pt_rand execution.
12. **Filter `none` is not free:** the pipeline still builds `InstanceState`s and swallows any exception in that block (`cara_serve.py:2817-2818`).
13. **cara_native leftovers:** silent largest-model return when no estimate matches the pool (`cara_native.py:87-88`); weights read from the live global dict; synchronous estimator call; factory default for missing types.
14. **E1 and E2 change measured results, so they are arms, not refactors.** Banked v3 evidence: BR router_ms p50 ≈ 49–57 ms inline (the ~18.7 req/s serial ceiling recorded in D17); BR sq/po2 `stats_fetch_ms` p50 25 ms at λ6, ≈145 ms at λ12 and ≈2.3 s at λ24 (brfill), versus 41 ms for passthrough SQ at λ30. The 3 s stats timeout is measured on the event loop, so time the loop spends in an inline forward counts against it; the 24 λ≥18 BR sq/po2 grid cells were INVALID under fail-loud, which `v3-exp/EXPERIMENTS.md` attributes to the per-request stats path exhausting under load (the author's ruling in `project_dispatch_determinism_bugs.md` names the combination of serial scoring and the per-request stats fan-out). Per the owner's first rule, before any E1/E2 campaign list each execution difference between RB and the baselines and either share it or run the control arm. Known items for this component: (a) RB's estimator/predictors already run in the default executor while BR/AP run inline (E1 closes this); (b) RB fetches stats once per batch with in-batch optimistic updates, while SQ/Po2 fetch per request without them (E2 changes this, in a direction the owner must choose); (c) RB's scorer breaks ties by queue then LRU (`cara_serve.py:2028-2043`), SQ breaks them by list order (owner-ruled acceptable, `866a0f0`); (d) the scheduler shares its CPU with the 14B_0 worker, which affects inline CPU routers more than off-loop ones.
15. **Documentation drift:** the BR docstring ("smallest in pool"), the commit message of `ed4ee5b` and D17's description of `batch_scoring=False` all describe behaviour the code does not have (§3.1, §5). Port from the code and the banked behaviour, not from those texts.
16. **Instrumentation gaps worth closing without changing decisions:** router `score`/`reason` are dropped on the pipeline path and the SQ/Po2 blind counters are never exposed; logging both per request would let a rebuild prove parity cell-by-cell. This adds output only.

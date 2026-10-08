review comments

> **📖 Design Document**: See [IMPLEMENTATION_TASKS_V2_DESIGN.md](IMPLEMENTATION_TASKS_V2_DESIGN.md) for detailed designs addressing all the issues below.
> **Status**: All issues have been analyzed and designs created (January 26, 2026). Pending implementation.

---



state_manager.py
LGTM, but would be better also tracking the time spent for sync as part of metrics later
predictor_registry.py
LGTM
scheduling_policy
	The question I noticed so far, we have simple two mode. 
Summary: Cara and bootstrap mode and the bootstrap mode will automatically converted into Cara mode. Both of the mode can use route_batch to do, and current cara_scheduling policy only implement the simple best_latency approaches
	So, 1) the rr/random and least_load scheduling besides act as the bootstrap for training data of latency predictor, it will also act as the baseline over cara during evaluation, and always suppose to be dispatched in streaming way (one by one) for fairness per-batch is not their requirment
2) the automatical conversation between bootstrapping and cara can not be required, as in our current design/plan, for the latency prediction, no real prompts/responses is needed, so we just generate the dummy prompts (always with dummy token) with given trace length and limited its response length as max_tokens similar like burstgpt did (btw, these also need to be updated in the benchmark serving accordingly to support such bootstrap mode with 1) given trace of data 2) keep shuffle the data and uses its length information only to construct the dummy requests 3) send the requests with dummy prompts under given qps 4) stop until specified number_training data reached. need to discuss the design of this later). And for the real evaluation, we are using the real testing dataset which has never been played for training purposes. 
3) both the scheduling policy should be extendable via a interface and implementation and configurable through a scheduler config. For the heuristic scheduling policy, round-robin, random and least load and also other scheduler could implemented later. And it should also be allowed to leverage the model quality and cost estimation as filter (before scheduling) or not, as the baseline

 For the Cara, scheduler, it should fit our current design in the paper (rechecking the design!) so multiple object such as  relevance, cost, cluster memory balance, latency with configurable weights? And to apply the LPT, we should sort first (also configurable to allow ablation study later)? for now, I see just a simple based on the least latencies. Are we plan to have it later? 

service_rate_tracker.py
self.alpha not convince me, it means we need to manually set it before the data collection and then could affect the ground truth of training data? That is hard to debugging e.g poor relevance finally showed and we cannot identify if this is the data issues (due to miss setting of alpha) or the model issues (no enough features) can this to be decoupled two ground truth feature of states to let the model self-learned?

xgboost predictor (other predictor is optional, so have not checked it yet)
	
"target_metric": float(prediction),
            "latency_prediction": float(prediction),
	this can be wrong if the predicted metrics is ttft instead of e2e latency, we should have the target_feature only only and also have target_feature_type as part of shared predictor config fields
	we can iterating the model/feature design later. But we should also have the option to use either 1) real decode length logged in the call back or 2) predicted length in the request for the latency predictions as we discussed before. Also, we should track the overhead of different predictor (optional, if enabled) as part of our measurement
training xgboost
    python -m block.predictor.cara.offline_training.train_xgboost \
        --input training_data/cara/*.jsonl \
        --output-dir models/cara/xgboost
	So, are we supposed to train all the instances with one commend and produce the number of instance types model separately? or a single model for all instances? If not this first one, how we can achieve the first, run the training script for each instance type 1 by 1? can we have a shell scripts to automatically handle the whole e2e training pipeline similar to the exp script? (so auto deploy instances, cara serve, run the benchmark serving with right parameters and configs, auto-collect the trainable data from all instances, split the dataset and run the training pipeline and evaluation pipeline finally)
         Besides, similar as above the training label(predicted metrics) should be configurable and use e2e latency as default. So,we can also explore other metrics in the futures.
	Also, have us also report the learning curve for the training for tuning?
evaluate_xgboost
	we may also need the cdf about the error distribution plot, And besides the distribution of predicted and real metrics.
	Also, the metrics should be allow to be different and read from config.
cara_serve
I see the implementation of batch completion, but that seems not our design, our design is still allow the client to send request individually but for cara batch scheduler, it will append it to a global request queue and scheduling it in batch by itself (with dynamic batch size as we emphased in the design) and return any response immediately if completed (so no need to wait to the slowest request in current batches), your current implementation missed all these and also will introduce significant waiting delay for the no slowest requests in the batch, which is bad and wrong. (again recheck the all the design documentations and paper / code)
Also, recheck if it current works align with the bootstrapping benchmark mentioned above.
DummyCARAPredictor
this also not fit our design, so after agreeing to move the predictor from instance side car to the scheduler side, the predictor instance and the data collection serve is decoupled, and this class is used for the data collection purpose but still using the predict interface and api which is incorrect and need a rename/refactoring.
Missing part
with all these update, we used to agree to have another major revision and updates over the deploy scripts (deploy_cara.py) but seems missed here.

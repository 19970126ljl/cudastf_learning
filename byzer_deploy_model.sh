byzerllm deploy --pretrained_model_type saas/openai \
--cpus_per_worker 0.01 \
--gpus_per_worker 0 \
--num_workers 1 \
--worker_concurrency 1 \
--infer_params saas.api_key=token saas.model=qwen2.5:latest saas.base_url="http://localhost:11434/v1/" \
--model ollama_qwen2_5_chat

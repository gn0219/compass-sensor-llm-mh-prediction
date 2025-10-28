import sys
import subprocess

models = [
    "llama-3.1-8b",
    "mistral-7b",
    "gemma2-9b"
]

combos = [
    {"strategy": "none",            "reasoning": "direct",        "n_shot": 0, "load_prompts": "globem_compass_zeroshot_none_direct_seed42"},
    {"strategy": "none",            "reasoning": "cot",           "n_shot": 0, "load_prompts": "globem_compass_zeroshot_none_cot_seed42"},
    {"strategy": "cross_random",    "reasoning": "cot",           "n_shot": 4, "load_prompts": "globem_compass_4shot_crossrandom_cot_seed42"},
    {"strategy": "cross_retrieval", "reasoning": "cot",           "n_shot": 4, "load_prompts": "globem_compass_4shot_crossretrieval_cot_seed42"},
    {"strategy": "personal_recent", "reasoning": "cot",           "n_shot": 4, "load_prompts": "globem_compass_4shot_personalrecent_cot_seed42"},
    {"strategy": "hybrid_blend",    "reasoning": "cot",           "n_shot": 4, "load_prompts": "globem_compass_4shot_hybridblend_cot_seed42"}
]

SEED = 42
LLM_SEED = 42
CHECKPOINT_EVERY = 5
CONTINUE_ON_ERROR = True  # Continue on error

try:
    for model in models:
        for combo in combos:
            cmd = [
                "python",
                "run_evaluation.py",
                "--mode", "batch",
                "--load-prompts", combo["load_prompts"],
                "--n_shot", str(combo["n_shot"]),
                "--strategy", combo["strategy"],
                "--seed", str(SEED),
                "--model", model,
                "--reasoning", combo["reasoning"],
                "--llm_seed", str(LLM_SEED),
                "--checkpoint-every", str(CHECKPOINT_EVERY),
                "--verbose"
            ]

            print("🚀", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed (exit {e.returncode}): {' '.join(cmd)}")
                if not CONTINUE_ON_ERROR:
                    raise
            # Add short sleep if needed
except KeyboardInterrupt:
    print("\nStopped by user.")

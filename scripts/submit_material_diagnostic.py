"""Submit the reviewed configuration to LlamaBoard's Start-training endpoint."""

import json
from pathlib import Path

import wandb
from gradio_client import Client


def main():
    # Online monitoring is required for this run; fail before starting a GPU job.
    viewer = wandb.Api(timeout=30).viewer
    if not viewer:
        raise RuntimeError("Log in with wandb login before starting this experiment")
    args = json.loads(Path("/root/autodl-tmp/logs/material_webui_api_inputs.json").read_text())
    client = Client("http://127.0.0.1:7860")
    print(client.predict(*args, api_name="/preview_train"), flush=True)
    job = client.submit(*args, api_name="/run_train")
    for result in job:
        # LlamaBoard keeps running even when the user's SSH terminal closes.
        text = str(result[0])
        print(text[-1600:], flush=True)
    print("WEBUI_JOB_FINISHED", flush=True)


if __name__ == "__main__":
    main()

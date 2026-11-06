"""CLI for the single-epoch Qwen outcome/action experiment."""
from pathlib import Path


def add_sft_parser(subparsers):
    parser = subparsers.add_parser("sft", help="prepare, train and evaluate full-episode LoRA")
    stages = parser.add_subparsers(dest="sft_command", required=True)
    prepare = stages.add_parser("prepare", help="verify the release and encode complete episodes on CPU")
    prepare.add_argument("--stage", choices=("all", "reconstruct", "encode", "select"), default="all")
    prepare.add_argument("--dataset-dir", type=Path, help="local snapshot of the pinned HF release")
    prepare.add_argument("--prepared-dir", type=Path, required=True, help="external working directory")
    prepare.add_argument("--model-path", type=Path, help="local snapshot of the pinned base model")
    for name, help_text in (("train", "run GPU checks, one epoch, then held-out evaluation"),
                            ("evaluate", "evaluate the saved adapter on held-out layouts")):
        stage = stages.add_parser(name, help=help_text)
        stage.add_argument("--prepared-dir", type=Path, required=True)
        stage.add_argument("--model-path", type=Path, required=True)
        stage.add_argument("--run-dir", type=Path, required=True)
        if name == "train":
            stage.add_argument("--resume", action="store_true", help="continue from the committed optimizer step")
            stage.add_argument("--fused-kernels", action="store_true", default=None,
                               help="verify and use fused language RMSNorm and SwiGLU")


def run_sft(args):
    prepared = args.prepared_dir.expanduser().resolve()
    model = args.model_path.expanduser().resolve() if args.model_path else None
    if args.sft_command == "prepare":
        if args.stage == "select":
            from robodata.sft.encoding import select_eligible
            manifest = select_eligible(prepared)
            return {"manifest": str(prepared / "eligible_manifest.json"), "fingerprint": manifest["fingerprint"],
                    "counts": manifest["counts"], "excluded": manifest["excluded"]}
        if args.stage != "reconstruct" and model is None:
            raise ValueError("--model-path is required for encoding")
        if args.stage != "encode":
            if args.dataset_dir is None:
                raise ValueError("--dataset-dir is required for reconstruction")
            from robodata.sft.archive import reconstruct
            reconstruct(args.dataset_dir.expanduser().resolve(), prepared)
        if args.stage != "reconstruct":
            from robodata.sft.encoding import encode_all
            encode_all(prepared, model)
        return {"stage": args.stage, "prepared_dir": str(prepared)}
    from robodata.sft.training import evaluate, train
    run = args.run_dir.expanduser().resolve()
    if args.sft_command == "train":
        train(prepared, model, run, resume=args.resume, fused_kernels=args.fused_kernels)
    else:
        evaluate(prepared, model, run)
    return {"stage": args.sft_command, "run_dir": str(run)}

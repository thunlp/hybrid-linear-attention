from pathlib import Path

from accelerate.utils import set_seed
import torch
from transformers import AutoModelForCausalLM

from modeling.hypenet import HypeNetForCausalLM, HypeNetConfig
from modeling.hypenet import build_hybrid_from_ckpt
from trainer.trainer import LMTrainer
from preparation import (
    get_args,
    get_accelerator,
    get_dataloaders,
    prepare_optimizers,
    load_train_config,
)


def main():
    args = get_args()
    load_train_config(args)
    set_seed(args.seed)
    accelerator = get_accelerator(args)

    accelerator.print("================ args ================")
    accelerator.print(args)
    accelerator.print("======================================")

    # Make output dir and dump args.
    output_dir = Path(args.output_dir, args.proj_name, args.run_name)
    if accelerator.is_main_process:
        output_dir.mkdir(exist_ok=True, parents=True)
        args.save(str(output_dir / "args.json"))

    # This is the actual batch size
    tokens_per_iter = (
        accelerator.num_processes
        * args.grad_accum_steps
        * args.batch_size
        * args.max_len
    )
    accelerator.print(f"Tokens per batch: {tokens_per_iter:,}")
    accelerator.print(f"Process: {accelerator.num_processes}")
    accelerator.print(f"Grad accum: {args.grad_accum_steps}")
    accelerator.print(f"Batch size: {args.batch_size}")
    accelerator.print(f"Max len: {args.max_len:,}")

    accelerator.print("Building student model...")

    if args.init_method == 'build_from_stage2':
        model = build_hybrid_from_ckpt(args, accelerator, args.init_from)
    elif args.init_method == 'HypeNetForCausalLM.from_pretrained':
        assert args.init_from is not None
        model = HypeNetForCausalLM.from_pretrained(args.init_from)
    elif args.init_method == 'state_dict':
        assert args.model_config is not None
        assert args.init_from is not None
        config = HypeNetConfig.from_json_file(Path(args.model_config))
        model = HypeNetForCausalLM(config=config)
        print("Loading state dict")
        state_dict = torch.load(args.init_from)
        print("Loading state dict into model")
        model.load_state_dict(state_dict)
    elif args.init_method == 'AutoModelForCausalLM.from_pretrained':
        model = AutoModelForCausalLM.from_pretrained(args.init_from, trust_remote_code=True)
    else:
        raise ValueError
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model = model.to(torch.bfloat16)  # type: ignore

    n_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Model parameters: {n_params:,}")

    accelerator.print("================ model ================")
    accelerator.print(model)
    accelerator.print("=======================================")

    if accelerator.is_main_process:
        # save model config
        # model_config: PretrainedConfig = model.config  # type: ignore
        config_path = output_dir / "config.json"
        accelerator.print(f"Saving model config to {config_path}")
        model.config.to_json_file(config_path)

    accelerator.print("Preparing optimizers...")
    optimizer, lr_scheduler = prepare_optimizers(
        model=model,
        args=args,
        accelerator=accelerator,
    )

    # Compile with PyTorch 2.0, very powerful
    if bool(args.compile):
        assert args.device != "mps", "torch.compile not supported on MPS"
        accelerator.print("compiling the model... (takes a ~minute)")
        # unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0  # type: ignore

    accelerator.print("Preparing dataloaders...")
    train_loader, val_loader = get_dataloaders(
        args=args,
        tok_path=args.tok_path,
        accelerator=accelerator,
    )

    accelerator.print("Preparing LMTrainer...")
    trainer = LMTrainer(
        args=args,
        output_dir=output_dir,
        accelerator=accelerator,
        model=model,  # type: ignore
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    accelerator.print("===== Start training =====")
    trainer.train()
    accelerator.print("===== Done training =====")


if __name__ == "__main__":
    main()


from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import set_seed
import torch
from modeling.tfm2rnn.modeling_hybrid import HypeNetForCausalLM
from transformers import AutoTokenizer

from arguments import Args
from modeling.tfm2rnn.distillation_model import DistillationModel, build_hybrid_from_ckpt
from trainer.trainer import LMTrainer
from preparation import get_args, get_accelerator, get_dataloaders, prepare_optimizers, load_train_config
from utils import print_trainable_parameters


def main():
    torch.set_default_dtype(torch.bfloat16)
    args: Args = get_args()
    load_train_config(args)
    set_seed(args.seed)
    accelerator: Accelerator = get_accelerator(args)

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

    # Load teacher model
    assert args.teacher_model is not None and args.teacher_model != '', "Teacher model is not specified"
    accelerator.print(f"Loading teacher model from {args.teacher_model}...")

    teacher_model = HypeNetForCausalLM.from_pretrained(args.teacher_model)
    print_trainable_parameters(teacher_model)
    accelerator.print(f"Loading tokenizer from {args.tok_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path, pad_token_id=2)

    # Build the student model from previous step
    accelerator.print(f"Building student model...")
    student_model = build_hybrid_from_ckpt(
        args=args,
        accelerator=accelerator,
        model_name=args.model_name,
        checkpoint_path=args.init_from,
    )

    accelerator.wait_for_everyone()

    assert args.loss_fn is not None, "Loss function is not specified"
    model = DistillationModel(
        teacher_model,
        student_model,
        loss_fn=args.loss_fn,
        # train_mlp=student_model.config.train_mlp,
    )
    n_student_params = sum(p.numel() for p in student_model.parameters())
    n_teacher_params = sum(p.numel() for p in teacher_model.parameters())
    accelerator.print(f"Student model parameters: {n_student_params:,}")
    accelerator.print(f"Teacher model parameters: {n_teacher_params:,}")
    n_distillation_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Distillation model parameters: {n_distillation_params:,}")
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Trainable parameters: {n_trainable_params:,} ({n_trainable_params / n_student_params * 100:.2f}% of student)")

    accelerator.print("================ model ================")
    accelerator.print(model)
    accelerator.print("=======================================")

    if accelerator.is_main_process:
        # save model config
        # model_config: PretrainedConfig = model.config  # type: ignore
        student_config_path = output_dir / "student_config.json"
        student_model.config.to_json_file(student_config_path)
        teacher_config_path = output_dir / "teacher_config.json"
        teacher_model.config.to_json_file(teacher_config_path)

    accelerator.print("Preparing optimizers...")
    optimizer, lr_scheduler = prepare_optimizers(model=model, args=args, accelerator=accelerator)

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

    # Print memory usage
    accelerator.print("Printing memory usage...")
    accelerator.print(f"Memory usage: {torch.cuda.memory_summary(device=args.device)}")

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
    accelerator.print('===== Done training =====')


if __name__ == '__main__':
    main()

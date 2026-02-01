from pathlib import Path
from functools import partial
from copy import deepcopy

from accelerate.utils import set_seed
import torch
from transformers import AutoTokenizer

from arguments import Args
from modeling.tfm2rnn.hidden_state_aligner import (
    HiddenStateAligner,
)
from modeling.tfm2rnn.lightning_attn import (
    build_lightning_attn_with_attn,
)
from modeling.tfm2rnn.kda import build_kda_with_attn
from modeling.tfm2rnn.gdn import build_gdn_with_attn
from modeling.tfm2rnn.modeling_hybrid import HypeNetForCausalLM
from modeling.tfm2rnn.configuration_hybrid import HybridConfig
from trainer.trainer import LMTrainer
from preparation import get_args, get_accelerator, get_dataloaders, prepare_optimizers, load_train_config


def main():
    # torch.set_float32_matmul_precision('high')  # wtf is this?
    torch.set_default_dtype(torch.bfloat16)
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
        assert len(list(output_dir.glob('ckpt_*'))) == 0, f"Output directory {output_dir} already exists, please choose another the output directory."
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

    assert args.loss_fn is not None, "loss_fn must be specified"

    if args.model_name in ["LA", "lightning-attn"]:
        # Build student config from orig config.
        student_config = HybridConfig.from_json_file(args.model_config)
        layer_init_fn = partial(
            build_lightning_attn_with_attn,
            config=student_config,
        )
        accelerator.print("==== Student layer config ====")
        accelerator.print(student_config)
        accelerator.print("=======================")
    elif args.model_name in ["kda"]:
        student_config = HybridConfig.from_json_file(args.model_config)
        layer_init_fn = partial(
            build_kda_with_attn,
            config=student_config,
        )
        accelerator.print("==== Student layer config ====")
        accelerator.print(student_config)
        accelerator.print("=======================")
    elif args.model_name in ["gdn"]:
        student_config = HybridConfig.from_json_file(args.model_config)
        layer_init_fn = partial(
            build_gdn_with_attn,
            config=student_config,
        )
        accelerator.print("==== Student layer config ====")
        accelerator.print(student_config)
        accelerator.print("=======================")
    else:
        raise ValueError(f"Model name {args.model_name} not supported")
    accelerator.print("Preparing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)

    accelerator.print("Preparing teacher model...")
    # model: nn.Module = get_model(accelerator, args)
    if 'qwen3' in args.init_from:
        accelerator.print(f"Loading model from {args.init_from}")
        orig_model = HypeNetForCausalLM.from_pretrained(args.init_from).to(
            args.device,
            dtype=torch.bfloat16,
        )
        accelerator.print(orig_model)
        orig_config = orig_model.config
    elif 'minicpm' in args.init_from:
        from modeling.mch.modeling_mch import MCHForCausalLM
        accelerator.print(f"Loading model from {args.init_from}")
        orig_model = MCHForCausalLM.from_pretrained(args.init_from).to(
            args.device,
            dtype=torch.bfloat16,
        )
        if student_config.lightning_use_rope:
            # We need to re-initialize the position embeddings in MCHModel, which is used by
            # lightning attention layers.
            
            orig_model.model.config.lightning_rope_scaling = student_config.lightning_rope_scaling
            orig_model.model.config.lightning_head_dim = student_config.lightning_head_dim
            orig_model.model.config.lightning_use_rope = student_config.lightning_use_rope

            # if student_config.lightning_head_dim is not None:
            #     # Change the head dim of RoPE to match lightning attn's head dim.
            #     orig_model.model.config.lightning_head_dim = student_config.lightning_head_dim
            # else:
            #     orig_model.model.config.lightning_head_dim = orig_model.model.config.head_dim
            # orig_model.model.config.lightning_head_dim = orig_model.model.config.head_dim
            orig_model.model._init_rope()

        accelerator.print(orig_model)
        orig_config = orig_model.config
    else:
        raise ValueError(f"Model {args.init_from} not supported")

    n_layers = len(orig_model.model.layers)
    mixer_types = student_config.mixer_types
    assert len(mixer_types) == n_layers, "mixer_types must have the same length as the number of layers"
    # convert_layer_idxs = [i for i, mixer_type in enumerate(mixer_types) if mixer_type == "lightning-attn"]
    convert_layer_idxs = None
    accelerator.print(f"Converting model to hidden state aligner...")
    model = HiddenStateAligner(
        model=orig_model,
        layer_init_fn=layer_init_fn,
        loss_fn=args.loss_fn,
        convert_layer_idxs=convert_layer_idxs,
    ).to(args.device, dtype=torch.bfloat16)

    accelerator.print('=' * 100)
    prompt = 'My name is'
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = inputs.to(args.device)
    with torch.no_grad():
        orig_model.eval()
        outputs = orig_model.generate(**inputs, max_new_tokens=10, do_sample=False)
        orig_model.train()
    accelerator.print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    accelerator.print('=' * 100)
    orig_model.config.use_cache = False
    # orig_model.gradient_checkpointing_enable()

    # accelerator.print(model.model.model.layers[0].self_attn.student_layer.q_proj.weight)
    # accelerator.print(model.model.model.layers[0].self_attn.teacher_layer.q_proj.weight)
    # accelerator.print(model.model.model.layers[0].self_attn.student_layer.q_norm.weight)
    # accelerator.print(model.model.model.layers[0].self_attn.teacher_layer.q_norm.weight)
    # accelerator.print(model.model.model.layers[0].self_attn.student_layer.o_proj.weight)
    # accelerator.print(model.model.model.layers[0].self_attn.teacher_layer.o_proj.weight)

    accelerator.print("================ model ================")
    accelerator.print(model)
    accelerator.print("=======================================")
    n_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Number of parameters: {n_params:,}")
    n_student_params = sum(p.numel() for n, p in model.named_parameters() if "student" in n)
    accelerator.print(f"Student parameters: {n_student_params:,}")
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Trainable parameters: {n_trainable_params:,}")

    if accelerator.is_main_process:
        # save model config
        # model_config: PretrainedConfig = model.config  # type: ignore
        orig_config_path = output_dir / "orig_config.json"
        orig_config.to_json_file(orig_config_path)
        student_config_path = output_dir / "student_config.json"
        student_config.to_json_file(student_config_path)

    accelerator.print("Preparing optimizers...")
    optimizer, lr_scheduler = prepare_optimizers(model=model, args=args, accelerator=accelerator)

    # # Compile with PyTorch 2.0, very powerful
    # if bool(args.compile):
    #     assert args.device != "mps", "torch.compile not supported on MPS"
    #     accelerator.print("compiling the model... (takes a ~minute)")
    #     # unoptimized_model = model
    #     model = torch.compile(model)  # requires PyTorch 2.0  # type: ignore

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
        model=model,
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

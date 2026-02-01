from pathlib import Path
import json
from typing import Tuple
from datetime import datetime

import torch
from accelerate import Accelerator
from torch import nn, Tensor
from torch.utils.data import DataLoader
# from accelerate import DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import DataLoaderConfiguration
from torch.optim import AdamW
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
# from data.modelbest_sdk import ModelBestSDKDatasetBuilder
from optim.lr_scheduler import WSDScheduler

from arguments import Args
from data import get_data


def load_train_config(args: Args):
    """
    Load from `args.train_config` if it exists, and use the values
    when the argument is not provided.
    """
    if args.train_config is not None and Path(args.train_config).exists():
        print(f"Loading train config from {args.train_config}")
        config = json.load(open(args.train_config, "r"))
        for k, v in config.items():
            if getattr(args, k) is None:
                setattr(args, k, v)
    else:
        print(f"WARNING: train config {args.train_config} does not exist.")
        print("It is highly recommended to provide a train config.")


def get_accelerator(args: Args, find_unused_parameters: bool = False) -> Accelerator:
    assert args.grad_accum_steps is not None
    assert args.run_name is not None
    assert args.proj_name is not None

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=find_unused_parameters,
    )

    dataloader_config = DataLoaderConfiguration(
        use_stateful_dataloader=False,
    )

    # ds_plugin = DeepSpeedPlugin(
    #     gradient_accumulation_steps=args.grad_accum_steps,
    #     hf_ds_config="configs/accelerate/zero3_config.json",
    # )

    project_dir = f"./results/{args.proj_name}"
    log_with = args.report_to.split(",")
    accelerator = Accelerator(
        project_dir=project_dir,
        log_with=log_with,  # type: ignore
        gradient_accumulation_steps=args.grad_accum_steps,
        # This argument allows us to step the LR scheduler manually.
        # This is needed because internally, accelerator expects the
        # LR scheduler to step more frequently when using multiple
        # processes or when using gradient accumulation (e.g.,
        # the warmup steps should be scaled by
        # num_processes * gradient_accumulation).
        step_scheduler_with_optimizer=False,
        # dataloader_config=dataloader_config,
        kwargs_handlers=[ddp_kwargs] if find_unused_parameters else [],
        # deepspeed_plugin=ds_plugin,
    )
    accelerator.print(f"Project directory: {project_dir}")
    accelerator.print(f"Reporting to: {log_with}")
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d-%H%M%S")
    hps = args.as_dict()
    run_name = f"{args.run_name}_{formatted_time}"
    accelerator.init_trackers(
        project_name=args.proj_name,
        config=hps,
        init_kwargs={
            "wandb": {
                "name": run_name,
            },
            "swanlab": {
                "experiment_name": run_name,
            },
        },
    )
    return accelerator


def cu_seqlens_collate_fn(
    batch: dict[str, Tensor],
) -> dict[str, Tensor]:
    seqlens = torch.tensor([it["input_ids"].size(0) for it in batch], dtype=torch.int32)

    cu = torch.empty(seqlens.numel() + 1, dtype=torch.int32)
    cu[0] = 0
    torch.cumsum(seqlens, dim=0, out=cu[1:])

    input_ids: Tensor = torch.cat([it["input_ids"] for it in batch], dim=0).to(
        dtype=torch.long
    )
    labels: Tensor = torch.cat([it["labels"] for it in batch], dim=0).to(
        dtype=torch.long
    )

    if input_ids.numel() == 0:
        raise ValueError("input_ids is empty after concatenation, unexpected.")

    return {
        "input_ids": input_ids.view(1, -1),  # (1, total_tokens)
        "labels": labels.view(1, -1),  # (1, total_tokens)
        "cu_seqlens": cu.view(1, -1),  # (1, B + 1)
    }


class ModelBestArgs:
    def __init__(
        self,
        data_path=None,
        train_data_path=None,
        valid_data_path=None,
        test_data_path=None,
        micro_batch_size=32,
        seq_length=512,
        use_fixed_length_segment=False,
        no_load_data_state=False,
        ckpt_step=None,
        load=None,
        clear_sampler_state=False,
    ):
        self.data_path = data_path
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.test_data_path = test_data_path
        self.micro_batch_size = micro_batch_size
        self.seq_length = seq_length
        self.use_fixed_length_segment = use_fixed_length_segment
        self.no_load_data_state = no_load_data_state
        self.ckpt_step = ckpt_step
        self.load = load
        self.clear_sampler_state = clear_sampler_state


def get_dataloaders(
    args: Args, tok_path: str, accelerator: Accelerator
) -> Tuple[DataLoader, DataLoader | None]:
    assert args.data_name is not None
    assert args.data_path is not None
    assert args.max_len is not None
    assert args.batch_size is not None
    if args.use_ulysses_attn is not None and bool(args.use_ulysses_attn):
        assert not args.shift_labels_in_model

    if args.data_name == "modelbest":
        # TODO: Should move this to `data/__init__.py`.
        from modelbest_sdk.dataset.thrift_wrapper.dataset_context import DatasetContext

        modelbest_args = ModelBestArgs(
            # data_path=["0.5","/home/wangshuo/wangshuo04/project/data/c4_dedup","0.5","/home/wangshuo/wangshuo04/project/linear_attention/zyx_infra/data"], #args.data_path,
            data_path=[item.strip() for item in args.mb_sdk_data_list.split(",")],
            # data_path=["1.0","/home/wangshuo/wangshuo04/project/linear_attention/zyx_infra/data"], #args.data_path,
            # train_data_path='path/to/train_data',
            # valid_data_path='path/to/valid_data',
            # test_data_path='path/to/test_data',
            micro_batch_size=args.batch_size,
            seq_length=args.max_len,
            use_fixed_length_segment=True,
            no_load_data_state=True,
            ckpt_step=args.save_interval,
            # load='path/to/checkpoints',
            clear_sampler_state=True,
        )
        print(f"modelbest_args = {vars(modelbest_args)}")
        config = DatasetContext(
            # rank=mpu.get_data_parallel_rank(),
            # world_size=mpu.get_data_parallel_world_size(),
            # tp_rank=mpu.get_tensor_model_parallel_rank(),
            # tp_size=mpu.get_tensor_model_parallel_world_size(),
            # pp_rank=mpu.get_pipeline_model_parallel_rank(),
            # pp_size=mpu.get_pipeline_model_parallel_world_size(),
            # num_workers=args.num_workers,
            dataset_config_path="",
            # dataset_checkpoint_path=args.save,
            # Use a fixed seed that doesn't depend on parallel configuration
            # to ensure same data sampling across different context parallel sizes
            # seed=args.seed if hasattr(args, 'seed') and args.seed is not None else 1234,
            # seed = mpu.get_data_parallel_world_size() + mpu.get_data_parallel_rank(),
        )
        train_ds, valid_dataloader, test_dataloader = ModelBestSDKDatasetBuilder(
            config
        ).build(modelbest_args)
        # train_loader = DataLoader(
        #     train_ds,  # type: ignore
        #     batch_size=args.batch_size,
        #     pin_memory=True,
        #     num_workers=1 if args.data_n_workers is None else args.data_n_workers,
        #     collate_fn=None,
        # )
        return train_ds.dataloader, None

    if bool(args.use_cp):
        assert not bool(args.shift_labels_in_model)
        shift_labels_in_dataset = False
        dp_size = accelerator.num_processes // args.sp_size
        # We want each GPU to see ctx_len = max_len / sp_size.
        # But, we need to pass sp_size * dp_size to
        # UlyssesSPDataLoaderAdapter, so the max_len need to be multiplied
        # by dp_size. Otherwise, the sequence length observed by each SP
        # rank will be shorter.
        max_len = args.max_len * dp_size
    else:
        shift_labels_in_dataset = not bool(args.shift_labels_in_model)
        max_len = args.max_len

    train_ds = get_data(
        tokenizer_name=tok_path,
        data_name=args.data_name,
        data_path=args.data_path,
        max_len=max_len,
        is_main_process=accelerator.is_main_process,
        shift_labels=shift_labels_in_dataset,
        is_seq2seq=bool(args.is_seq2seq),
        stage=args.stage,
    )  # type: ignore

    if args.use_cu_seqlens:
        collate_fn = cu_seqlens_collate_fn
    else:
        collate_fn = None

    train_loader = DataLoader(
        train_ds,  # type: ignore
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=1 if args.data_n_workers is None else args.data_n_workers,
        collate_fn=collate_fn,
    )


    if args.validation_data_path is not None and args.validation_data_name is not None:
        val_ds = get_data(
            tokenizer_name=tok_path,
            data_name=args.validation_data_name,
            data_path=args.validation_data_path,
            max_len=max_len,
            is_main_process=accelerator.is_main_process,
            shift_labels=shift_labels_in_dataset,
            is_seq2seq=bool(args.is_seq2seq),
            stage=args.stage,
        )  # type: ignore
        if args.validation_data_size is not None:
            val_ds = val_ds.take(args.validation_data_size)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, collate_fn=collate_fn  # type: ignore
        )
    else:
        val_loader = None

    return train_loader, val_loader


def get_args() -> Args:
    args = Args().parse_args()

    if args.compile == 1:
        if args.grad_ckpt == 1:
            print(
                "Cannot use grad checkpoint and compile mode together, setting compile to 0."
            )
            args.compile = 0

        if args.model_name in ["gated_deltanet", "gated-deltanet"]:
            print(
                "Gated DeltaNet does not support torch.compile, setting compile to 0..."
            )
            args.compile = 0

        if args.model_name in ["rabbit"]:
            print("Rabbit does not support compilation, turning it off...")
            args.compile = 0

        if args.model_name in ["mamba2"]:
            print("Turning off compilation for Mamba2")
            args.compile = 0

    return args


def prepare_optimizers(model: nn.Module, args: Args, accelerator: Accelerator):
    """
    Returns: (optimizer, lr_scheduler)
    """
    assert args.lr is not None
    assert args.beta1 is not None
    assert args.beta2 is not None
    assert args.weight_decay is not None
    assert args.n_warmup_steps is not None
    assert args.n_train_steps is not None
    assert args.lr_scheduler is not None

    decay = set()
    no_decay = set()
    for name, param in model.named_parameters():
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.add(name)
        else:
            decay.add(name)

    trainable_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    frozen_params = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    accelerator.print(f"Trainable parameters: {len(trainable_params)}")
    accelerator.print(f"Trainable parameters: {list(trainable_params.keys())}")
    # if not model.student_model.config.train_mlp:
    #     rm_names = [n for n, p in trainable_params.items() if '.mlp.' in n]
    #     for n in rm_names:
    #         del trainable_params[n]
    # accelerator.print(f"Trainable parameters: {len(trainable_params)}")
    # accelerator.print(f"Trainable parameters: {list(trainable_params.keys())}")

    param_groups = [
        {
            "params": [p for n, p in trainable_params.items() if n in decay],
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in trainable_params.items() if n in no_decay],
            "lr": args.lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in frozen_params.items()],
            "lr": 0.0,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        param_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    if args.lr_scheduler == "wsd":
        assert args.n_drop_steps is not None
        lr_scheduler = WSDScheduler(
            optimizer=optimizer,
            lr=args.lr,
            n_decay_iters=args.n_drop_steps,
            n_warmup_iters=args.n_warmup_steps,
            n_train_iters=args.n_train_steps,
            min_lr=args.min_lr,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.n_warmup_steps,
            num_training_steps=args.n_train_steps,
            min_lr=args.min_lr,
        )
    else:
        raise ValueError(f"Invalid LR scheduler: {args.lr_scheduler}")
    return optimizer, lr_scheduler

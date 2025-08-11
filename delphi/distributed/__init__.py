from . import backend, ddp, single

BACKEND_TYPE_TO_MODULE_MAP = {
    "nccl": ddp.DataParallelDistributedBackend,
    None: single.SinlgeNodeBackend,
}


def make_backend_from_args(args) -> backend.DistributedBackend:
    return BACKEND_TYPE_TO_MODULE_MAP[args.distributed_backend](args)


def registered_backends():
    return BACKEND_TYPE_TO_MODULE_MAP.keys()

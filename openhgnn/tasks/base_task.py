from abc import ABC, ABCMeta, abstractmethod
import argparse
import atexit
import os
import torch


# class LoadFrom(ABCMeta):
#     def __call__(cls, *args, **kwargs):
#         obj = type.__call__(cls, *args, **kwargs)
#         obj.load_from_pretrained()
#         if hasattr(obj, "models") and hasattr(obj, "device"):
#             obj.model.set_device(obj.device)
#         return obj


class BaseTask(ABC):

    def __init__(self, args):
        super(BaseTask, self).__init__()
        os.makedirs("./checkpoints", exist_ok=True)
        self.loss_fn = None
        self.evaluator = None

        self.load_from_checkpoint = hasattr(args, "checkpoint") and args.checkpoint
        if self.load_from_checkpoint:
            self._checkpoint = os.path.join("checkpoints", f"{args.model}_{args.dataset}.pt")
            atexit.register(self.save_checkpoint)
        else:
            self._checkpoint = None

    def load_from_pretrained(self):
        if self.load_from_checkpoint:
            try:
                ck_pt = torch.load(self._checkpoint)
                self.model.load_state_dict(ck_pt)
            except FileNotFoundError:
                print(f"'{self._checkpoint}' doesn't exists")
        return self.model

    def save_checkpoint(self):
        if self._checkpoint and hasattr(self.model, "_parameters()"):
            torch.save(self.model.state_dict(), self._checkpoint)

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
        self.model.set_loss_fn(self.loss_fn)

    def set_evaluator(self, dataset):
        self.evaluator = dataset.get_evaluator()

    def get_trainer(self, model, args):
        if hasattr(args, "trainer") and args.trainer is not None:
            if "self_auxiliary_task" in args.trainer and not hasattr(model, "get_embeddings"):
                raise ValueError("Model ({}) must implement get_embeddings method".format(args.model))
            return build_trainer(args)
        elif model.get_trainer(None, args) is not None:
            return model.get_trainer(None, args)(args)
        else:
            return None

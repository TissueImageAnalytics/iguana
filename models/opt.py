import torch.optim as optim

from run_utils.callbacks.base import (
    AccumulateRawOutput,
    CheckpointSaver,
    ProcessAccumulatedRawOutput,
    ScalarMovingAverage,
    ScheduleLr,
    TrackLr,
    TriggerEngine,
)
from run_utils.callbacks.logging import LoggingEpochOutput
from run_utils.engine import Events
from .net_desc import create_model
from .run_desc import proc_valid_step_output, train_step, valid_step


train_config = {
    # ------------------------------------------------------------------
    # phases are run sequentially from index 0 to N
    "phase_list": [
        {
            "run_info": {
                # may need more dynamic for each network
                "net": {
                    # model name and number of features retrieved from config
                    "desc": lambda: create_model(
                        model_name=None, 
                        nr_features=None,
                        node_degree=None),
                    "optimizer": [
                        optim.Adam,
                        {  # should match keyword for parameters within the optimizer
                            "lr": 5.0e-3,  # initial learning rate,
                            "betas": (0.9, 0.999),
                        },
                    ],
                    # learning rate scheduler
                    "lr_scheduler": lambda x: optim.lr_scheduler.MultiStepLR(x, milestones=[24], gamma=0.2),
                    
                    "extra_info": {"loss": {"ce": 1},},
                    # path to load, -1 to auto load checkpoint from previous phase,
                    # None to start from scratch
                    "pretrained": None,
                },
            },
            "nr_types": None,
            "target_info": {"gen": None, "viz": None},
            "batch_size": {"train": 64, "valid": 64,},  # engine name : value
            "nr_epochs": 60,
        },
    ],
    # ------------------------------------------------------------------
    # TODO: dynamically for dataset plugin selection and processing also?
    # all enclosed engine shares the same neural networks
    # as the on at the outer calling it
    "run_engine": {
        "train": {
            # TODO: align here, file path or what? what about CV?
            "dataset": "",  # what about compound dataset ?
            "nr_procs": 10,  # number of threads for dataloader
            "run_step": train_step,  # TODO: function name or function variable ?
            "reset_per_run": False,
            # callbacks are run according to the list order of the event
            "callbacks": {
                Events.STEP_COMPLETED: [
                    ScalarMovingAverage(),
                ],
                Events.EPOCH_COMPLETED: [
                    TrackLr(),
                    CheckpointSaver(),
                    LoggingEpochOutput(),
                    TriggerEngine("valid"),
                    ScheduleLr(),
                ],
            },
        },
        "valid": {
            "dataset": "",  # whats about compound dataset ?
            "nr_procs": 10,  # number of threads for dataloader
            "run_step": valid_step,
            "reset_per_run": True,  # * to stop aggregating output etc. from last run
            # callbacks are run according to the list order of the event
            "callbacks": {
                Events.STEP_COMPLETED: [AccumulateRawOutput(),],
                Events.EPOCH_COMPLETED: [
                    ProcessAccumulatedRawOutput(proc_valid_step_output),
                    LoggingEpochOutput(),
                ],
            },
        },
    },
}

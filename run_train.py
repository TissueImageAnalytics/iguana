"""run_train.py

Main training script for graph-based networks.

Usage:
  run_train.py [--gpu=<id>]
  run_train.py (-h | --help)
  run_train.py --version

Options:
  -h --help             Show this string.
  --version             Show version.
  --gpu=<id>            Comma separated GPU list. 
"""

import dataloader.graph_loader as graph_loader
from misc.utils import get_pna_deg, rm_n_mkdir, get_local_feat_stats
from run_utils.utils import check_manual_seed, colored
from run_utils.engine import RunEngine
from config import Config

import torch_geometric
from torch.nn import DataParallel  # TODO: switch to DistributedDataParallel
import torch
from tensorboardX import SummaryWriter
import json
import os
import inspect
import glob
import numpy as np
from docopt import docopt
import cv2
import pandas as pd

cv2.setNumThreads(0)


#### have to move outside because of spawn
# * must initialize augmentor per worker, else duplicated rng generators may happen

def worker_init_fn(worker_id):
    # ! to make the seed chain reproducible, must use the torch random, not numpy
    # the torch rng from main thread will regenerate a base seed, which is then
    # copied into the dataloader each time it created (i.e start of each epoch)
    # then dataloader with this seed will spawn worker, now we reseed the worker
    worker_info = torch.utils.data.get_worker_info()
    return


class TrainManager(Config):
    """
    Either used to view the dataset or
    to initialise the main training loop.
    """

    def __init__(self, args):
        super().__init__()
        self.model_config = self.model_config_file.__getattribute__("train_config")
        self._parse_args(args)

        if self.model_name == "pna" and self.compute_deg:
            get_pna_deg(self.dataset.all_dir_list, self.dataset.file_ext, self.data_path)

        return

    def _parse_args(self, run_args):
        """Parse command line arguments and set as instance variables."""
        for variable, value in run_args.items():
            self.__setattr__(variable[2:], value)
        return


    def get_datagen(
        self,
        batch_size,
        run_mode,
        nr_procs=0,
        fold_idx=0,
        compute_stats=None
    ):

        file_list = []

        train_files = []
        for dir_name in self.dataset.train_dir_list:
            train_files.extend(glob.glob("%s/*%s" % (dir_name, self.dataset.file_ext)))
        valid_files = []
        for dir_name in self.dataset.valid_dir_list:
            valid_files.extend(glob.glob("%s/*%s" % (dir_name, self.dataset.file_ext)))
        
        # wsi_info = pd.read_excel("data/CoBi_WSI_diagnosis.xlsx")
        # train_files = refine_files(train_files, wsi_info)
        # valid_files = refine_files(valid_files, wsi_info)

        all_data_list = train_files + valid_files
        if run_mode == "train":
            file_list = train_files
        else:
            file_list = valid_files
        file_list.sort()  # to always ensure same input ordering

        assert len(file_list) > 0, (
            "No %s found for `%s`, please check `%s` in `config.py`"
            % (self.dataset.file_ext, run_mode, "%s_dir_list" % run_mode,)
        )
        print("Dataset %s: %d" % (run_mode, len(file_list)))

        if compute_stats:
            local_mean, local_median, local_std, local_perc_25, local_perc_75 = get_local_feat_stats(all_data_list)
            np.save(f"{self.data_path}/local_mean.npy", local_mean)
            np.save(f"{self.data_path}/local_median.npy", local_median)
            np.save(f"{self.data_path}/local_std.npy", local_std)
            np.save(f"{self.data_path}/local_perc_25.npy", local_perc_25)
            np.save(f"{self.data_path}/local_perc_75.npy", local_perc_75)
        else:
            print('Loading feature statistics...')
            local_mean = np.load(f"{self.data_path}/local_mean.npy")
            local_median = np.load(f"{self.data_path}/local_median.npy")
            local_std = np.load(f"{self.data_path}/local_std.npy")
            local_perc_25 = np.load(f"{self.data_path}/local_perc_25.npy")
            local_perc_75 = np.load(f"{self.data_path}/local_perc_75.npy")

        feat_stats = [
            local_mean, local_median, local_std, local_perc_25, local_perc_75
        ]

        input_dataset = graph_loader.FileLoader(
            file_list, feat_stats=feat_stats, norm="standard", data_clean="std"
        )

        dataloader = torch_geometric.loader.DataLoader(
            input_dataset,
            num_workers=nr_procs,
            batch_size=batch_size,
            shuffle=run_mode == "train",
            drop_last=run_mode == "train",
            worker_init_fn=worker_init_fn,
        )
        return dataloader

    def run_once(self, opt, run_engine_opt, log_dir, prev_log_dir=None, fold_idx=0):
        """
        Simply run the defined run_step of the related method once
        """
        check_manual_seed(self.seed)

        log_info = {}
        if self.logging:
            # check_log_dir(log_dir)
            rm_n_mkdir(log_dir)

            tfwriter = SummaryWriter(log_dir=log_dir)
            json_log_file = log_dir + "/stats.json"
            with open(json_log_file, "w") as json_file:
                json.dump({}, json_file)  # create empty file
            log_info = {
                "json_file": json_log_file,
                "tfwriter": tfwriter,
            }

        def get_last_chkpt_path(prev_phase_dir, net_name):
            stat_file_path = prev_phase_dir + "/stats.json"
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            epoch_list = [int(v) for v in info.keys()]
            last_chkpts_path = "%s/%s_epoch=%d.tar" % (
                prev_phase_dir,
                net_name,
                max(epoch_list),
            )
            return last_chkpts_path

        # TODO: adding way to load pretrained weight or resume the training
        # parsing the network and optimizer information
        net_run_info = {}
        net_info_opt = opt["run_info"]
        for net_name, net_info in net_info_opt.items():
            assert inspect.isclass(net_info["desc"]) or inspect.isfunction(
                net_info["desc"]
            ), "`desc` must be a Class or Function which instantiate NEW objects !!!"
            net_desc = net_info["desc"]()

            # TODO: customize print-out for each run ?
            # summary_string(net_desc, (3, 270, 270), device='cpu')

            pretrained_path = net_info["pretrained"]
            if pretrained_path is not None:
                if pretrained_path == -1:
                    # * depend on logging format so may be broken if logging format has been changed
                    pretrained_path = get_last_chkpt_path(prev_log_dir, net_name)
                    net_state_dict = torch.load(pretrained_path)["desc"]
                    # conversion to single mode if its saved in parallel mode
                    variable_name_list = list(net_state_dict.keys())
                    is_in_parallel_mode = all(
                        v.split(".")[0] == "module" for v in variable_name_list
                    )
                    if is_in_parallel_mode:
                        colored_word = colored("WARNING", color="red", attrs=["bold"])
                        print(
                            (
                                "%s: Detect checkpoint saved in data-parallel mode."
                                " Converting saved model to single GPU mode."
                                % colored_word
                            ).rjust(80)
                        )
                        net_state_dict = {
                            ".".join(k.split(".")[1:]): v
                            for k, v in net_state_dict.items()
                        }
                else:
                    net_state_dict = dict(np.load(pretrained_path))
                    net_state_dict = {
                        k: torch.from_numpy(v) for k, v in net_state_dict.items()
                    }

                colored_word = colored(net_name, color="red", attrs=["bold"])
                print(
                    "Model `%s` pretrained path: %s" % (colored_word, pretrained_path)
                )

                # load_state_dict returns (missing keys, unexpected keys)
                load_feedback = net_desc.load_state_dict(net_state_dict, strict=False)
                # * uncomment for your convenience
                print("Missing Variables: \n", load_feedback[0])
                print("Detected Unknown Variables: \n", load_feedback[1])

            # currently we only support parallel mode for standard pytorch (not pytorch geometric)
            # net_desc = DataParallel(net_desc)
            net_desc = net_desc.to("cuda")
            # print out trainable parameters
            # for name, param in net_desc.named_parameters():
            #     if param.requires_grad:
            #         print(name)
            # print(net_desc) # * dump network definition or not?
            optimizer, optimizer_args = net_info["optimizer"]
            optimizer = optimizer(net_desc.parameters(), **optimizer_args)
            scheduler = net_info["lr_scheduler"](optimizer)
            net_run_info[net_name] = {
                "desc": net_desc,
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                # TODO: standardize API for external hooks
                "extra_info": net_info["extra_info"],
            }

        # parsing the running engine configuration
        assert (
            "train" in run_engine_opt
        ), "No engine for training detected in description file"

        # initialize runner and attach callback afterward
        # * all engine shared the same network info declaration
        runner_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            if runner_name == "train" and self.compute_stats:
                compute_stats=True
            else:
                compute_stats=False
            runner_dict[runner_name] = RunEngine(
                dataloader = self.get_datagen(
                    opt["batch_size"][runner_name],
                    runner_name,
                    nr_procs=runner_opt["nr_procs"],
                    fold_idx=fold_idx,
                    compute_stats=compute_stats,
                ),
                engine_name=runner_name,
                run_step=runner_opt["run_step"],
                run_info=net_run_info,
                log_info=log_info,
            )

        for runner_name, runner in runner_dict.items():
            callback_info = run_engine_opt[runner_name]["callbacks"]
            for (event, callback_list,) in callback_info.items():
                for callback in callback_list:
                    if callback.engine_trigger:
                        triggered_runner_name = callback.triggered_engine_name
                        callback.triggered_engine = runner_dict[triggered_runner_name]
                    runner.add_event_handler(event, callback)

        # retrieve main runner
        main_runner = runner_dict["train"]
        main_runner.state.logging = self.logging
        main_runner.state.log_dir = log_dir
        # start the run loop
        main_runner.run(opt["nr_epochs"])

        print("\n")
        print("########################################################")
        print("########################################################")
        print("\n")
        return

    def run(self):
        """
        Define multi-stage run or cross-validation or whatever in here
        """
        phase_list = self.model_config["phase_list"]
        engine_opt = self.model_config["run_engine"]

        prev_save_path = None

        for phase_idx, phase_info in enumerate(phase_list):
            save_path = self.log_dir
            self.run_once(
                phase_info, engine_opt, save_path, prev_log_dir=prev_save_path
            )
            prev_save_path = save_path


if __name__ == "__main__":
    # args = docopt(__doc__, version="GCN v1.0")
    # trainer = TrainManager()
    args = {}
    args["--gpu"] = "0"
    args["--device"] = "cuda"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--gpu"]
    args["--compute_stats"] = False
    args["--compute_deg"] = False
    args["--data_path"] = "/root/lsf_workspace/graph_data/cobi/graph_data_refined_spiro/"

    trainer = TrainManager(args)
    trainer.run()

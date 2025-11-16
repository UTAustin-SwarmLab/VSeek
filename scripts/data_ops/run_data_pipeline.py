from data.lvb import LongVideoBench
from data.lvbench import LVBench
from data.videomme import VideoMME
from data.mlvu import MLVU

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../src/vseek/config/retriever", config_name="config")
def main(cfg: DictConfig):
    if cfg.dataset.name == "lvb":
        dataset = LongVideoBench(cfg)
        dataset.save_it_as_vseek_data(desired_interval_in_sec=cfg.dataset.desired_interval_in_sec)
    elif cfg.dataset.name == "lvbench":
        dataset = LVBench(cfg)
        dataset.save_it_as_vseek_data(desired_interval_in_sec=cfg.dataset.desired_interval_in_sec)
    elif cfg.dataset.name == "videomme":
        dataset = VideoMME(cfg)
        dataset.save_it_as_vseek_data(desired_interval_in_sec=cfg.dataset.desired_interval_in_sec)
    elif cfg.dataset.name == "mlvu":
        dataset = MLVU(cfg)
        dataset.save_it_as_vseek_data(desired_interval_in_sec=cfg.dataset.desired_interval_in_sec)
    # dataset.fix_subtitle_embeddings()


if __name__ == "__main__":
    main()

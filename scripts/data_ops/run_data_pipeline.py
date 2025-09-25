from data.lvb import LongVideoBench
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../src/vseek/config/retriever", config_name="config")
def main(cfg: DictConfig):
    dataset = LongVideoBench(cfg)
    dataset.save_it_as_vseek_data(desired_interval_in_sec=cfg.dataset.desired_interval_in_sec)
    # dataset.fix_subtitle_embeddings()


if __name__ == "__main__":
    main()

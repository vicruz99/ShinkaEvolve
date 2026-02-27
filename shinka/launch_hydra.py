#!/usr/bin/env python3
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from shinka.core import EvolutionRunner, AsyncEvolutionRunner  # import both


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    print("Experiment configurations:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    job_cfg = hydra.utils.instantiate(cfg.job_config)
    db_cfg = hydra.utils.instantiate(cfg.db_config)
    evo_cfg = hydra.utils.instantiate(cfg.evo_config)

    use_async = cfg.get("async_runner", False)  # new config flag

    if use_async:
        runner = AsyncEvolutionRunner(
            evo_config=evo_cfg,
            job_config=job_cfg,
            db_config=db_cfg,
            verbose=cfg.verbose,
            max_evaluation_jobs=cfg.get("max_evaluation_jobs", None),
            max_proposal_jobs=cfg.get("max_proposal_jobs", 10),
        )
        asyncio.run(runner.run())
    else:
        runner = EvolutionRunner(
            evo_config=evo_cfg,
            job_config=job_cfg,
            db_config=db_cfg,
            verbose=cfg.verbose,
        )
        runner.run()


if __name__ == "__main__":
    main()
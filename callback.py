
from hydra.experimental.callback import Callback
from hydra.core.utils import JobReturn
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import wandb

class LogCallback(Callback):
    def __init__(self):
        pass
    def on_job_end(self, config: DictConfig, job_return: JobReturn):
        
        study = job_return._return_value['study']
        logging = job_return._return_value['logging']
        conf= dict(model_name = config.models.name,
                    ds_name = config.ds.name,
                    holdout = config.ds.holdout)
        
        if logging:
            with wandb.init(
                    config = conf,
                    project=config.project_name,
                    tags = [config.models.name]) as run:
                
                for step, trial in enumerate(study.trials):

                    run.log(trial.params, step = step)
                    run.log({"best_mae": trial.value})

                run.summary['best_mae'] = study.best_value
                run.summary = study.best_params
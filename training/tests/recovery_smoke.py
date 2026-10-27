import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import torch

from dermoai.common import read_json, write_json
from dermoai.experiments import Experiments
from dermoai.training import train


def main():
    project = Path(__file__).resolve().parents[1]
    source = sorted((project / 'runtime').glob('smoke-*/config.json'))[-1]
    config = read_json(source)
    root = project / 'runtime' / ('recovery-' + str(time.time_ns()))
    root.mkdir()
    config['runtime'] = str(root / 'runs')
    write_json(root / 'config.json', config)
    import dermoai.training as training
    original_infer = training.infer_rows
    attempts = 0

    def injected_oom(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise torch.cuda.OutOfMemoryError('Synthetic injected failure for rollback validation')
        return original_infer(*args, **kwargs)

    with patch.object(training, 'infer_rows', side_effect=injected_oom):
        status = train(root)
    assert status['state'] == 'completed' and status['oom_retries'] == 1 and status['batch_size'] == 1
    experiments = Experiments(config['runtime'])
    job = experiments.start({**config, 'epochs': 5, 'schedule_epochs': 5}, approved=True)
    experiments.stop(job['run_id'])
    deadline = time.time() + 120
    while time.time() < deadline:
        stopped = experiments.status(job['run_id'])
        if stopped['state'] not in ('running', 'starting'):
            break
        time.sleep(1)
    assert stopped['state'] == 'stopped', stopped
    assert Path(stopped['last_checkpoint']).exists()
    early_root = root / 'early-stop'
    early_root.mkdir()
    write_json(early_root / 'config.json', {**config, 'epochs': 3, 'early_stopping_patience': 1})
    with patch.object(training, 'infer_rows', return_value=(list(range(12)), list(range(12)))):
        early_status = train(early_root)
    assert early_status['epoch'] == 2 and early_status['stop_reason'] == 'validation_selection_score_early_stopping'
    assert early_status['best_epoch'] == 1
    result = {'passed': True, 'oom_test': 'Injected CUDA OOM; epoch rolled back and completed at batch size 1', 'stop_test': 'Worker stopped safely with checkpoint retained', 'early_stopping_test': 'Fixed mocked validation scores stop at epoch 2 and retain epoch 1 as best; scores are not real performance', 'synthetic_only': True}
    write_json(root / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

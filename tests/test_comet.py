import transformers
import comet_ml

def test_import():
    experiment = comet_ml.OfflineExperiment(
        offline_directory="/tmp",
        log_code=False,
        log_env_details=False,
    )

    assert experiment.alive is True
    experiment.end()
    

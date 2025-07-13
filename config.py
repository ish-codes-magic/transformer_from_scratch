from pathlib import Path

def get_config():
    return {
        "batch_size":1,
        "num_epochs": 20,
        "lr": 0.0001,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "fr",
        "model": "weights",
        "model_filename": "tmodel_",
        "preload": None,
        "tokeniser_file": "tokeniser_{0}.json",
        "experiment_name": "runs/tmodel"
    }
    
    
def get_weights_file_path(config, epoch: str):
    model_folder = config['model']
    model_basename = config['model_filename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path(model_folder) / model_filename)

    
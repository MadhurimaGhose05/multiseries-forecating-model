from models.chronos_model import Chronos2Forecaster

def train_or_prepare(artifact_dir="artifacts"):
    # Chronos-2 is pretrained: there is no task-specific gradient training here.
    # Loading once verifies weights/API and downloads the model cache.
    model = Chronos2Forecaster(device="cpu").load()
    model.save_calibration(f"{artifact_dir}/chronos_calibration.json")
    print("Chronos-2 pretrained CPU pipeline loaded; weights are cached by Hugging Face.")
    return model

if __name__ == "__main__": train_or_prepare()

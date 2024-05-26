import os
import shutil
from tensorflow.keras.models import load_model

def deploy_models(models_dir, deployment_dir):
    if not os.path.exists(deployment_dir):
        os.makedirs(deployment_dir)

    # Copy trained models to deployment directory
    for model_file in os.listdir(models_dir):
        if model_file.endswith('.h5'):
            shutil.copy2(os.path.join(models_dir, model_file), os.path.join(deployment_dir, model_file))

    # Optionally, include any necessary preprocessing or postprocessing scripts
    # shutil.copytree('preprocess/', os.path.join(deployment_dir, 'preprocess/'))

if __name__ == "__main__":
    models_dir = '../results/models/'
    deployment_dir = '../deployment/models/'

    deploy_models(models_dir, deployment_dir)


![logo](https://github.com/volvox292/mass2smiles/assets/63146629/7e5b37dc-534b-4780-b310-45f197283709)

Mass2SMILES is an open-source Python based deep learning approach for structure and functional group prediction from mass spectrometry data (MS/MS). Spectral data can be provided as MGF files (GNPS-syle) and model inference is most effciently performed via the provided docker container.


supplementary data with container and model at (you must have a vaild licence for NIST): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7883491.svg)](https://doi.org/10.5281/zenodo.7883491)

recent update containing dockerfiles to build two separate containers, adjust to your needs, this Mass2SMILES model container is using GPU, the cddd does not seem to work on newer cuda drivers, therefore it is 
 build using tensorflow cpu, but can be speed up by changing the number of cores: e.g. InferenceModel(cpu_threads=128). You need to point to your input and output dir, now the mass2smiles model is built into the container. Using this setup inference speed is highly improved.  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14778327.svg)](https://doi.org/10.5281/zenodo.14778327)

the pre-print is available at: https://doi.org/10.1101/2023.07.06.547963

```bash {bash, echo=T, eval=F}
# the container is available as tarball in supplementary or via docker pull delser292/mass2smiles:final
# unzip the docker.zip, the mass2smiles folder contains the model files and scripts to execute everything and it is important to specify the path to this folder when starting predictions.

# The predictions can be started through this command:

docker run -v c:/your_path/to_the_folder/mass2smiles/:/app  mass2smiles:transformer_v1 conda run -n tf python app/mass2smiles_transformer.py your_mgf_file.mgf /app
```

The model architecture:

![architecture](https://github.com/volvox292/mass2smiles/assets/63146629/3e4313d8-43b2-469d-bab6-c8670a00f62d)




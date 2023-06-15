![logo](https://github.com/volvox292/mass2smiles/assets/63146629/950c4462-c3ac-46ba-be40-08174e637e29)

Mass2smiles is an open-source Python based deep learning approach for structure and functional group prediction from mass spectrometry data (MS/MS). Spectral data can be provided as MGF files (GNPS-syle) and model inference is most effciently performed via the provided docker container.


supplementary data with container and model at : [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7883491.svg)](https://doi.org/10.5281/zenodo.7883491)

the pre-print will be out soon!

```bash {bash, echo=T, eval=F}
# the container is available as tarball in supplementary or via docker pull delser292/mass2smiles:final
# The predictions can be started through this command:

docker run -v c:/your_path/to_the_folder/mass2smiles/:/app  mass2smiles:transformer_v1 conda run -n tf python app/mass2smiles_transformer.py your_mgf_file.mgf /app
```

The model architecture:

![architecture](https://github.com/volvox292/mass2smiles/assets/63146629/603a5307-d04a-4e87-95cc-2571ec424f5f)



@ECHO OFF

SET _INDIR=.\cepstrums\tmp
SET _OUTDIR=.\models

python create_svm_model.py --indir "%_INDIR%" --outmodel "%_OUTDIR%\ntt_sensor_svm.model"


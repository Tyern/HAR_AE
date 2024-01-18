@ECHO OFF

SET _INDIR="D:/work/fujisawa/KeioSensor/CreateModelFromSpectrumAnnotations/cepstrums"
REM 001 005 010 011 013

SET _OUTFILE="./models/20180730/eval001_005.csv"
SET _MODEL="./models/20180730/model001_005.model"
SET _T=001 005
python create_model.py --indir %_INDIR% --outeval %_OUTFILE% --outmodel %_MODEL% --timestep 3 --timeshift 1 --car_ids %_T% --eval_car_ids 001 005 010 011 013
SET _OUTFILE="./models/20180730/eval001_010.csv"
SET _MODEL="./models/20180730/model001_010.model"
SET _T=001 010
python create_model.py --indir %_INDIR% --outeval %_OUTFILE% --outmodel %_MODEL% --timestep 3 --timeshift 1 --car_ids %_T% --eval_car_ids 001 005 010 011 013
SET _OUTFILE="./models/20180730/eval001_011.csv"
SET _MODEL="./models/20180730/model001_011.model"
SET _T=001 011
python create_model.py --indir %_INDIR% --outeval %_OUTFILE% --outmodel %_MODEL% --timestep 3 --timeshift 1 --car_ids %_T% --eval_car_ids 001 005 010 011 013
SET _OUTFILE="./models/20180730/eval001_013.csv"
SET _MODEL="./models/20180730/model001_013.model"
SET _T=001 013
python create_model.py --indir %_INDIR% --outeval %_OUTFILE% --outmodel %_MODEL% --timestep 3 --timeshift 1 --car_ids %_T% --eval_car_ids 001 005 010 011 013
SET _OUTFILE="./models/20180730/eval005_010.csv"
SET _MODEL="./models/20180730/model005_010.model"
SET _T=005 010
python create_model.py --indir %_INDIR% --outeval %_OUTFILE% --outmodel %_MODEL% --timestep 3 --timeshift 1 --car_ids %_T% --eval_car_ids 005 005 010 011 013
SET _OUTFILE="./models/20180730/eval005_011.csv"
SET _MODEL="./models/20180730/model005_011.model"
SET _T=005 011
python create_model.py --indir %_INDIR% --outeval %_OUTFILE% --outmodel %_MODEL% --timestep 3 --timeshift 1 --car_ids %_T% --eval_car_ids 005 005 010 011 013
SET _OUTFILE="./models/20180730/eval005_013.csv"
SET _MODEL="./models/20180730/model005_013.model"
SET _T=005 013
python create_model.py --indir %_INDIR% --outeval %_OUTFILE% --outmodel %_MODEL% --timestep 3 --timeshift 1 --car_ids %_T% --eval_car_ids 005 005 010 011 013
SET _OUTFILE="./models/20180730/eval010_011.csv"
SET _MODEL="./models/20180730/model010_011.model"
SET _T=010 011
python create_model.py --indir %_INDIR% --outeval %_OUTFILE% --outmodel %_MODEL% --timestep 3 --timeshift 1 --car_ids %_T% --eval_car_ids 010 005 010 011 013
SET _OUTFILE="./models/20180730/eval010_013.csv"
SET _MODEL="./models/20180730/model010_013.model"
SET _T=010 013
python create_model.py --indir %_INDIR% --outeval %_OUTFILE% --outmodel %_MODEL% --timestep 3 --timeshift 1 --car_ids %_T% --eval_car_ids 010 005 010 011 013
SET _OUTFILE="./models/20180730/eval011_013.csv"
SET _MODEL="./models/20180730/model011_013.model"
SET _T=011 013
python create_model.py --indir %_INDIR% --outeval %_OUTFILE% --outmodel %_MODEL% --timestep 3 --timeshift 1 --car_ids %_T% --eval_car_ids 011 005 011 011 013

SET _OUTFILE="./models/20180730/eval001.csv"
SET _MODEL="./models/20180730/model001.model"
SET _T=001
python create_model.py --indir %_INDIR% --outeval %_OUTFILE% --outmodel %_MODEL% --timestep 3 --timeshift 1 --car_ids %_T% --eval_car_ids 001 005 010 011 013
SET _OUTFILE="./models/20180730/eval005.csv"
SET _MODEL="./models/20180730/model005.model"
SET _T=005
python create_model.py --indir %_INDIR% --outeval %_OUTFILE% --outmodel %_MODEL% --timestep 3 --timeshift 1 --car_ids %_T% --eval_car_ids 001 005 010 011 013
SET _OUTFILE="./models/20180730/eval010.csv"
SET _MODEL="./models/20180730/model010.model"
SET _T=010
python create_model.py --indir %_INDIR% --outeval %_OUTFILE% --outmodel %_MODEL% --timestep 3 --timeshift 1 --car_ids %_T% --eval_car_ids 001 005 010 011 013
SET _OUTFILE="./models/20180730/eval011.csv"
SET _MODEL="./models/20180730/model011.model"
SET _T=011
python create_model.py --indir %_INDIR% --outeval %_OUTFILE% --outmodel %_MODEL% --timestep 3 --timeshift 1 --car_ids %_T% --eval_car_ids 001 005 010 011 013
SET _OUTFILE="./models/20180730/eval013.csv"
SET _MODEL="./models/20180730/model013.model"
SET _T=013
python create_model.py --indir %_INDIR% --outeval %_OUTFILE% --outmodel %_MODEL% --timestep 3 --timeshift 1 --car_ids %_T% --eval_car_ids 001 005 010 011 013

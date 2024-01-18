import pandas as pd
import numpy as np
import csv
import glob
import os

def predata(indir: str, classpoint: float, outdir: str):
    if  os.path.exists(outdir + str(classpoint)) == False:os.mkdir(outdir + str(classpoint))#フォルダ作成
    inputfile = glob.glob(indir + "*")
    for csvfile in inputfile:
        predictdata = pd.read_csv(csvfile)
        predictdata = predictdata.set_index("timestamp")

        #marginsampling
        max2max = []
        for index, row in predictdata.iterrows():
            Row = np.array(row)
            Row = np.sort(Row)
            max2max.append(Row[3] - Row[2])
        
        data = {"margin": max2max}
        f = pd.DataFrame(data)
        f["timestamp"] = predictdata.index
        f = f.set_index("timestamp")

        #class化
        maginclass = []
        for index, row in f.iterrows():
            i = float(row)
            if i <= 0.1:
                maginclass.append(0.1)
            elif i <= 0.2:
                maginclass.append(0.2)
            elif i <= 0.3:
                maginclass.append(0.3)
            elif i <= 0.4:
                maginclass.append(0.4)
            elif i <= 0.5:
                maginclass.append(0.5)
            elif i <= 0.6:
                maginclass.append(0.6)
            elif i <= 0.7:
                maginclass.append(0.7)
            elif i <= 0.8:
                maginclass.append(0.8)
            elif i <= 0.9:
                maginclass.append(0.9)
            elif i <= 1:
                maginclass.append(1)
        
        #savefile
        outputfile = csvfile.replace(indir,"").replace("predicted","")
        f["class"] = maginclass
        classlist = f[f["class"] == classpoint].index
        classlist = pd.DataFrame(classlist)
        classlist.to_csv(outdir + "%s/"%str(classpoint) + str(classpoint) + outputfile)

def create_vaguedata(class_file: str, car_id: list, outdir: str, classposition: float):
    for cid in car_id:
        _cid = "-" + cid
        file_list = glob.glob(class_file + "%s/" %str(classposition) + "*%s*" %_cid)
        for filename in file_list:
            #clf = class_file + cld#パス結合
            classcsv = pd.read_csv(filename)
            cepdata = filename.replace(class_file,"").replace("%s/"%str(classposition),"").replace(str(classposition),"").replace(_cid,"").replace(".csv","").replace("." + cid,cid)

            accel = pd.read_csv("./cepstrums/%s/%s.accel.csv" %(cid, cepdata))
            gyro = pd.read_csv("./cepstrums/%s/%s.gyro.csv" %(cid, cepdata))
            label = pd.read_csv("./cepstrums/%s/%s.label.csv" %(cid, cepdata))

            #accel[accel["timestamp"] == classcsv["timestamp"]]
            classcsv = classcsv.set_index("timestamp")
            vagueaccel = pd.DataFrame(columns=("timestamp","car_id","cepstrum_x00","cepstrum_x01","cepstrum_x02","cepstrum_x03","cepstrum_x04","cepstrum_x05","cepstrum_x06","cepstrum_x07","cepstrum_x08","cepstrum_x09","cepstrum_x10","cepstrum_x11","cepstrum_x12","cepstrum_x13","cepstrum_x14","cepstrum_x15","cepstrum_x16","cepstrum_x17","cepstrum_x18","cepstrum_x19","cepstrum_x20","cepstrum_x21","cepstrum_x22","cepstrum_x23","cepstrum_x24","cepstrum_x25","cepstrum_x26","cepstrum_x27","cepstrum_x28","cepstrum_x29","cepstrum_y00","cepstrum_y01","cepstrum_y02","cepstrum_y03","cepstrum_y04","cepstrum_y05","cepstrum_y06","cepstrum_y07","cepstrum_y08","cepstrum_y09","cepstrum_y10","cepstrum_y11","cepstrum_y12","cepstrum_y13","cepstrum_y14","cepstrum_y15","cepstrum_y16","cepstrum_y17","cepstrum_y18","cepstrum_y19","cepstrum_y20","cepstrum_y21","cepstrum_y22","cepstrum_y23","cepstrum_y24","cepstrum_y25","cepstrum_y26","cepstrum_y27","cepstrum_y28","cepstrum_y29","cepstrum_z00","cepstrum_z01","cepstrum_z02","cepstrum_z03","cepstrum_z04","cepstrum_z05","cepstrum_z06","cepstrum_z07","cepstrum_z08","cepstrum_z09","cepstrum_z10","cepstrum_z11","cepstrum_z12","cepstrum_z13","cepstrum_z14","cepstrum_z15","cepstrum_z16","cepstrum_z17","cepstrum_z18","cepstrum_z19","cepstrum_z20","cepstrum_z21","cepstrum_z22","cepstrum_z23","cepstrum_z24","cepstrum_z25","cepstrum_z26","cepstrum_z27","cepstrum_z28","cepstrum_z29"))
            vaguegyro = pd.DataFrame(columns=("timestamp","car_id","cepstrum_x00","cepstrum_x01","cepstrum_x02","cepstrum_x03","cepstrum_x04","cepstrum_x05","cepstrum_x06","cepstrum_x07","cepstrum_x08","cepstrum_x09","cepstrum_x10","cepstrum_x11","cepstrum_x12","cepstrum_x13","cepstrum_x14","cepstrum_x15","cepstrum_x16","cepstrum_x17","cepstrum_x18","cepstrum_x19","cepstrum_x20","cepstrum_x21","cepstrum_x22","cepstrum_x23","cepstrum_x24","cepstrum_x25","cepstrum_x26","cepstrum_x27","cepstrum_x28","cepstrum_x29","cepstrum_y00","cepstrum_y01","cepstrum_y02","cepstrum_y03","cepstrum_y04","cepstrum_y05","cepstrum_y06","cepstrum_y07","cepstrum_y08","cepstrum_y09","cepstrum_y10","cepstrum_y11","cepstrum_y12","cepstrum_y13","cepstrum_y14","cepstrum_y15","cepstrum_y16","cepstrum_y17","cepstrum_y18","cepstrum_y19","cepstrum_y20","cepstrum_y21","cepstrum_y22","cepstrum_y23","cepstrum_y24","cepstrum_y25","cepstrum_y26","cepstrum_y27","cepstrum_y28","cepstrum_y29","cepstrum_z00","cepstrum_z01","cepstrum_z02","cepstrum_z03","cepstrum_z04","cepstrum_z05","cepstrum_z06","cepstrum_z07","cepstrum_z08","cepstrum_z09","cepstrum_z10","cepstrum_z11","cepstrum_z12","cepstrum_z13","cepstrum_z14","cepstrum_z15","cepstrum_z16","cepstrum_z17","cepstrum_z18","cepstrum_z19","cepstrum_z20","cepstrum_z21","cepstrum_z22","cepstrum_z23","cepstrum_z24","cepstrum_z25","cepstrum_z26","cepstrum_z27","cepstrum_z28","cepstrum_z29"))
            vaguelabel = pd.DataFrame(columns=("timestamp","car_id","NO_LABEL","ROLL","RUN","DOOR"))

            for index, row in classcsv.iterrows():
                vagueaccel = vagueaccel.append(accel[accel["timestamp"] == index])
                vaguegyro = vaguegyro.append(gyro[gyro["timestamp"] == index])
                vaguelabel = vaguelabel.append(label[label["timestamp"] == index])

            vagueaccel = vagueaccel.set_index("timestamp")
            vaguegyro = vaguegyro.set_index("timestamp")
            vaguelabel = vaguelabel.set_index("timestamp")
            if os.path.exists(outdir + str(classposition)) == False:os.mkdir(outdir + str(classposition))#フォルダ作成
            if os.path.exists(outdir + str(classposition) + "/%s" %str(cid)) == False:os.mkdir(outdir + str(classposition) + "/%s" %str(cid))
            vagueaccel.to_csv(outdir + "%s/" %str(classposition) + "%s/"%str(cid) + "%s.vague.accel.csv"%cepdata)
            vaguegyro.to_csv(outdir + "%s/" %str(classposition) + "%s/"%str(cid) + "%s.vague.gyro.csv"%cepdata)
            vaguelabel.to_csv(outdir + "%s/" %str(classposition) + "%s/"%str(cid) + "%s.vague.label.csv"%cepdata)


def marginsampling(indir: str, car_ids: list, outdir: str):
    for carid in car_ids:
        if  os.path.exists(outdir + str(carid)) == False:os.mkdir(outdir + str(carid))#フォルダ作成
        inputfile = glob.glob(indir + "*%s*000.csv" %carid)
        for csvfile in inputfile:
            predictdata = pd.read_csv(csvfile)
            predictdata = predictdata.set_index("timestamp")

            #marginsampling
            max2max = []
            for index, row in predictdata.iterrows():
                Row = np.array(row)
                Row = np.sort(Row)
                max2max.append(Row[3] - Row[2])
            
            data = {"margin": max2max}
            f = pd.DataFrame(data)
            f["timestamp"] = predictdata.index
            f = f.set_index("timestamp")
            f.to_csv(outdir + carid + "/%s.csv" % csvfile.replace("predicted.cepstrums","").replace("./models/20180809/predict/",""))


def adddata(indir:str, car_ids:list, outdir:str):
    marginsampling(indir,car_ids,outdir)
    
    for carid in car_ids:
        ptns = glob.glob(indir + "%s/*.label.csv"%carid)
        for ptn in ptns:
            label = pd.read_csv(ptn)
            accel = pd.read_csv(ptn.replace("label","accel"))
            gyro = pd.read_csv(ptn.replace("label","gyro"))
#            gyro.head()

            #if os.path.exists(outdir + str(carid)) == False:os.mkdir(outdir + str(carid))
            day = ptn.replace("./cepstrums/%s/" %carid, "").replace(".label.csv","").replace(carid + "_","").replace("_000","")
            #if os.path.exists(outdir + "%s/%s"%(carid, day)) == False:os.mkdir(outdir + "%s/%s"%(carid, day))
            for i in range(int(len(label) / 100)):
                if len(label) - i * 200 > 0:
                    label2 = label.iloc[i*100:len(label) - i * 100]
                    label2 = label2.set_index("timestamp")
                    label2.to_csv(outdir + "%s/%s/%04d.label.csv" %(day,carid,int(i)))
                    accel2 = accel.iloc[i*100:len(accel) - i * 100]
                    accel2 = accel.set_index("timestamp")
                    accel2.to_csv(outdir + "%s/%s/%04d.accel.csv" %(day,carid,int(i)))
                    gyro2 = gyro.iloc[i*100:len(gyro) - i * 100]
                    gyro2 = gyro2.set_index("timestamp")
                    gyro2.to_csv(outdir + "%s/%s/%04d.gyro.csv" %(day,carid,int(i)))


def marginsampling_for_LSTM(inpredict:str,car_ids:list, outaddpredict:str):
    for carid in car_ids:
        
        max001 = pd.read_csv(outaddpredict + "%s"%carid)
        window = len(pd.read_csv("./models/20180817/data/001/38001_20170406_000.label.csv"))
        while window <= len(max001):
            for slid in range(len(max001)+1-window):
                if slid == 0:
                    minest = max001.iloc[slid:slid + window]
                elif mean(minest["margin"]) > mean(max001.iloc[slid:slid + window]["margin"]):
                    minest = max001.iloc[slid:slid + window]
            minest.to_csv("./models/20180817/data/%s.csv"%window)
            window += 200
            

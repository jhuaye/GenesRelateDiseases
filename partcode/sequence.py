import pandas as pd
import csv
list=['class_Brain.Alzheimer','class_Brain.Multiple.Sclerosis','class_Brain.Parkinson','class_Disease.Brain', 'class_Disease.Heart',
      'class_Disease.Immune' ,'class_Disease.Muscle','class_Disease.Neoplasm','class_Disease.Nutrition','class_Heart.Arteriosclerosis' ,
      'class_Heart.Coronary.Disease','class_Heart.Hypertension','class_Heart.Myocardial.Infarction','class_Immune.Hypersensitivity' ,
      'class_Muscle.Arthritis', 'class_Muscle.Osteoporosis', 'class_Neoplasm.Adenocarcinoma', 'class_Neoplasm.Breast', 'class_Neoplasm.Colorectal',
      'class_Neoplasm.Lung' ,'class_Neoplasm.Prostatic', 'class_Neoplasm.Stomach', 'class_Nutritional.Diabetes.Type1', 'class_Nutritional.Diabetes.Type2'
         ,'class_Nutritional.Obesity', 'class_Respiratory.Asthma' ,'class_Disease'
]
for shu in range(0,23):
    b=pd.read_csv('ready/'+str(shu)+".csv",sep=',',index_col=0)
    for disease in list:
        c=b.sort_values(by=disease,axis=0,inplace=False,ascending=False)
    # print(c)
        get=b.columns.get_loc(disease)
        a=c.iloc[:,b.columns.get_loc(disease):b.columns.get_loc(disease)+2]
    # print(a)
        lim=0


            # print(i,j[0])
        with open("test1/"+disease+".csv", "a+") as csvfile:

            writer = csv.writer(csvfile)
            writer.writerow(["genes", "Probability"])
            for i, j in a.iterrows():
                if j[1] == 0:
                    lim = lim + 1
                    if lim>30 and lim<61:

        # 先写入columns_name

        # 写入多行用writerows
                        writer.writerow([i,j[0]])
    # print(index)
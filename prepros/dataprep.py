import pandas as pd
import numpy as np
from script.helper import *
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

pd.set_option("Max_rows",None)

def feature_eng():
    df = pd.read_csv("datasets/water_potability.csv")

    #nan değerlerin düzeltilmesi
    imputer = KNNImputer(n_neighbors=10)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    #permissable_ph cat oluşturulması
    df.loc[(df["ph"]<6.5) | (df["ph"]>8.5),"permissable_ph"]="bad"
    df.loc[(df["ph"]>=6.5)&(df["ph"]<=8.5),"permissable_ph"]="good"

    #hardness cat oluşturulma 4 kategori düşünüldü

    df["hardness_cat"] = pd.qcut(df["Hardness"],5,labels=[5,4,3,2,1])

    #solids cat oluşturma

    df["Solids"] = df["Solids"]/(67)

    df.loc[(df["Solids"]>0) & (df["Solids"]<250),"tds"]="good"
    df.loc[(df["Solids"]>=250) & (df["Solids"]<500),"tds"]="des_limit"
    df.loc[(df["Solids"]>=500) & (df["Solids"]<=750),"tds"]="not_so_des_lim"
    df.loc[(df["Solids"]>=750) ,"tds"]="maximum_limit"

    #chloramines

    df.loc[(df["Chloramines"]>0)&(df["Chloramines"]<2),"Chloramines_limit"]= "Best"
    df.loc[(df["Chloramines"]>=2)&(df["Chloramines"]<=4),"Chloramines_limit"] = "top_limit"
    df.loc[(df["Chloramines"]>4)&(df["Chloramines"]<=6),"Chloramines_limit"] = "not_so_drinkable"
    df.loc[(df["Chloramines"]>6),"Chloramines_limit"]="deadly"

    #sulfate

    df.loc[(df["Sulfate"]<250),"Sulfate_level"]="Good"
    df.loc[(df["Sulfate"]>=250)&(df["Sulfate"]<350),"Sulfate_level"]="Reasonable"
    df.loc[(df["Sulfate"]>=350),"Sulfate_level"]="Risky"


    #conductivity

    df.loc[(df["Conductivity"]<=400),"Conductivity_cat"]="good"
    df.loc[(df["Conductivity"]>400),"Conductivity_cat"]="bad"

    #Trihalomethanes

    df.loc[df["Trihalomethanes"]<=80,"Tri_cat"]="good"
    df.loc[df["Trihalomethanes"]>80,"Tri_cat"]="bad"

    #Turbidity

    df.loc[(df["Turbidity"]<5),"Tur_cat"]="Good"
    df.loc[(df["Turbidity"]>=5,"Tur_cat")]="bad"

    return  df


def dataprep(df):
    #aşırı outlier var ise onların düzeltilmesi
    cat_cols, num_cols, cat_but_car=grab_col_names(df)

    for col in num_cols:
        replace_with_thresholds(df,col,0.05,0.95)

#rare_analyser(df,"Potability",cat_cols)
#rare bakıldığında az rare olduğundan ve dataframe küçük olduğundan rare yapılmadı

#label encoder

    binary_cols = [col for col in df.columns if df[col].dtype not in [int,float] and df[col].nunique()==2]

    for col in binary_cols:
        label_encoder(df,col)

#OHE

    cat_cols.remove("Potability")

    df = pd.get_dummies(df,columns=cat_cols,drop_first=True)

    return df


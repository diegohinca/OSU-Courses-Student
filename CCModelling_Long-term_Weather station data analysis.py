import requests as rq
import pandas as pd
from os import path
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import time
import statsmodels.api as sm

st=[74671013806, 72327013897]

Ft_Campbell=st[0]
Nashville=st[1]
Funcion=0
# Funcion="download"

if Funcion=="download":
    num=2021-1901+1
    count=0
    for i in range(num):
        Year=1901+i
        r=rq.get("https://www.ncei.noaa.gov/data/local-climatological-data/access/%s"%(Year))
        if station in r.text:
            r1 = rq.get("https://www.ncei.noaa.gov/data/local-climatological-data/access/%s/%s.csv" % (Year, station))
            print(Year,"Yes")
            count+=1
            with open('%s-%s.csv'%(Year,station), 'wb') as f:
                f.write(r1.content)
        else:
            print(Year,"No")
    print("Total Years=",count)
# Funcion="createDF"
if Funcion=="createDF":
    DF =pd.read_csv("00_titles.csv")
    DF.drop(DF.columns.difference(["STATION","DATE","HourlyDryBulbTemperature"]), 1, inplace=True)
    for i in range(79):
        for stations in st:
            Yeari = i + 1943
            Year = str(Yeari)

            if path.exists("%s-%s.csv" % (Year, stations)):
                print("Adding_%s_%s"%(Year,stations))
                DFi = pd.read_csv("%s-%s.csv" % (Year, stations))
                DFi.drop(DFi.columns.difference(["STATION", "DATE", "HourlyDryBulbTemperature"]), 1, inplace=True)
                DF = DF.append(DFi)
# Funcion="start"
if Funcion=="start":
    DF = pd.read_csv("DataframeFULL03-12-1629.csv",dtype={"DATE":str,"STATION":str})
    DF.drop(DF.columns.difference(["STATION", "DATE", "HourlyDryBulbTemperature"]), 1, inplace=True)
    DF.drop_duplicates(inplace=True)
    DF["Temp"]=pd.to_numeric(DF["HourlyDryBulbTemperature"].str.strip("s,*"))
    DF["Temp"].replace(r'^\s*$', np.NaN, inplace=True)
    DF["DATE"]=pd.to_datetime(DF["DATE"],format="%Y-%m-%dT%H")
    DF["Year"] = pd.DatetimeIndex(DF["DATE"]).year
    DF["Month"] = pd.DatetimeIndex(DF["DATE"]).month
    DF["Day"] = pd.DatetimeIndex(DF["DATE"]).day
    DF["Hour"] = pd.DatetimeIndex(DF["DATE"]).hour
    DF["Minute"] = pd.DatetimeIndex(DF["DATE"]).minute
    DF["Second"] = pd.DatetimeIndex(DF["DATE"]).second

    DF=DF.dropna(axis="rows")
    # Count= pd.pivot_table(DFN, index=["Year", "Month", "Day", "Hour"], columns=["STATION"], values=["Temp"], aggfunc='count')
    # Count = Count[(Count["Temp"] > 1)]
    # Count = pd.pivot_table(Count, index=["Year"], columns=["STATION"], values=["Temp"], aggfunc='count')
###

    now = datetime.now()
    cdate = now.strftime("%m-%d-%H%M")
    DF.to_csv("2_DF_Start_%s.csv" % (cdate))

# Funcion="complete the dataset"
if Funcion=="complete the dataset":
    ####Create scheleton of dates

    YearsNASH=[1951,
               1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,
               1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,2000,2001,2004,
               2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,
               2019,
               2020]
    YearsFTCAMP=[1944,1951,1952,1953,1954,1955,1956,1957,1958,1959,1960,1961,1962,1963,
               1964,1965,1966,1967,1968,1969,1970,
               1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,
               1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1999,
               2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,
               2019,2020]
    ###NASHVILLE
    NASH_NP=np.empty((376944,4))
    FTC_NP=np.empty((553392,4))

    YNASHRANGE=2020-1951+1
    YFTCRANGE=2020-1944+1

    i=0
    for y in YearsNASH:
        if y % 4==0:
            bis=1
        else:
            bis=0
        for m in range(12):
            m+=1
            if m==1 or m==3 or m==5 or m==7 or m==8 or m==10 or m==12:
                days=31
            elif m==2:
                days=28+bis
            else:
                days=30
            for d in range(days):
                d+=1
                for h in range (24):
                    NASH_NP[i][0] = y
                    NASH_NP[i][1] = m
                    NASH_NP[i][2] = d
                    NASH_NP[i][3] = h

                    i += 1
    print("lengthNASH",i)

    i = 0
    for years in YearsFTCAMP:
        if y % 4 == 0:
            bis = 1
        else:
            bis = 0
        for m in range(12):
            m += 1
            if m == 1 or m == 3 or m == 5 or m == 7 or m == 8 or m == 10 or m == 12:
                days = 31
            elif m == 2:
                days = 28 + bis
            else:
                days = 30
            for d in range(days):
                d += 1
                for h in range(24):
                    FTC_NP[i][0] = y
                    FTC_NP[i][1] = m
                    FTC_NP[i][2] = d
                    FTC_NP[i][3] = h
                    i += 1
    print("lengthFTC",i)

    NASH_PD= pd.DataFrame(NASH_NP, columns=["Year","Month","Day","Hour"],dtype=int)
    FTC_PD= pd.DataFrame(FTC_NP, columns=["Year","Month","Day","Hour"],dtype=int)
    print(NASH_PD.head())

    NASH_PD["DATE"] = pd.to_datetime(NASH_PD["Year"].astype(str) + '-' +NASH_PD["Month"].astype(str)+ '-' +
                             NASH_PD["Day"].astype(str) + '-'+NASH_PD["Hour"].astype(str),format="%Y-%m-%d-%H")
    NASH_PD=NASH_PD.set_index(["Year","Month","Day","Hour"])
    NASH_PD["Temp"]=0

    FTC_PD["DATE"] = pd.to_datetime(FTC_PD["Year"].astype(str) + '-' +
                              FTC_PD["Month"].astype(str)+ '-' +
                             FTC_PD["Day"].astype(str) + '-' +
                             FTC_PD["Hour"].astype(str),
                              format="%Y-%m-%d-%H")
    FTC_PD["Temp"]=0
    FTC_PD=FTC_PD.set_index(["Year","Month","Day","Hour"])
    print("NASHVILLE",NASH_PD.head())
    print("Ft Campbell",FTC_PD.head())


####Loading the data
    NASH = pd.read_csv("MID_Nashville.csv", index_col=0)
    FTCAMP = pd.read_csv("MID_FTCambell.csv", index_col=0)


    NASH =NASH[
          (NASH["Year"]==1951)
          |((1973<= NASH["Year"]) & (NASH["Year"] <=1995))
          |((2000 <= NASH["Year"]) & (NASH["Year"] <= 2001))
          |((2004 <= NASH["Year"]) & (NASH["Year"] <= 2020))
          ]
    # NASH["Year_ix"]=NASH["Year"]-1951+1
    print("HEAD",NASH.head())
    NASH=NASH.set_index(["Year","Month","Day","Hour"])
    print("HEAD",NASH.head())
    NASH.drop(NASH.columns.difference(['DATE', 'Temp', 'Year', 'Month','Day', 'Hour']), 1,inplace=True)
    NASH_FINAL=pd.concat([NASH_PD,NASH], sort=False)
    NASH_FINAL.reset_index(inplace=True)
    NASH_FINAL["Temp"] = NASH_FINAL["Temp"].interpolate(method="polynomial", order=3)
    NASH_FINAL=NASH_FINAL.set_index(["Date"])


    FTCAMP=FTCAMP[(FTCAMP["Year"]==1944)
          |((1951<= FTCAMP["Year"]) & (FTCAMP["Year"] <=1970))
          |((1973 <= FTCAMP["Year"]) & (FTCAMP["Year"] <= 1997))
          |(1999 == FTCAMP["Year"])
          |((2005 <= FTCAMP["Year"]) & (FTCAMP["Year"] <= 2020))
          ]
    # FTCAMP["Year_ix"]=FTCAMP["Year"]-1944+1

    FTCAMP = FTCAMP.set_index(["Year", "Month", "Day", "Hour"])
    FTCAMP.drop(FTCAMP.columns.difference(['DATE', 'Temp', 'Year', 'Month', 'Day', 'Hour']), 1, inplace=True)


    FTCAMP_FINAL = pd.concat([FTC_PD, FTCAMP], sort=False)
    FTCAMP_FINAL.reset_index(inplace=True)
    FTCAMP_FINAL["Temp"]=FTCAMP_FINAL["Temp"].interpolate(method="polynomial", order=3)
    FTCAMP_FINAL=FTCAMP_FINAL.set_index(["Date"])

    now = datetime.now()
    cdate = now.strftime("%m-%d-%H%M")
    NASH_FINAL.to_csv("31_Nashville_%s.csv" % (cdate))
    FTCAMP_FINAL.to_csv("31_FTCambell_%s.csv" % (cdate))

# Funcion="prepare2"
if Funcion=="prepare2":
###################
    NASH_FINAL = pd.read_csv("31_Nashville_GOODYEARS2.csv", index_col=0)
    NASH_FINAL["Temp"].replace(0, np.NaN, inplace=True)
    NASH_FINAL["Temp"] = NASH_FINAL["Temp"].interpolate(method="polynomial", order=3)
    NASH_FINAL = pd.pivot_table(NASH_FINAL, index=["Year","Month","Day","Hour"], values=["Temp"], aggfunc="mean")
    NASH_FINAL.reset_index(inplace=True)
    NASH_FINAL["DATE"] = pd.to_datetime(NASH_FINAL["Year"].astype(str) + '-' +
                              NASH_FINAL["Month"].astype(str)+ '-' +
                             NASH_FINAL["Day"].astype(str) + '-' +
                             NASH_FINAL["Hour"].astype(str),
                              format="%Y-%m-%d-%H")
    NASH_FINAL=NASH_FINAL.set_index(["DATE"])
    NASH_FINAL["Temp"].replace(0, np.NaN, inplace=True)



    Nashville_T= NASH_FINAL["Temp"]
    Nashville_D= NASH_FINAL.index
###########
    FTCAMP_FINAL = pd.read_csv("31_FTCambell_GOODYEARS2.csv", index_col=0)
    FTCAMP_FINAL["Temp"].replace(0, np.NaN, inplace=True)
    FTCAMP_FINAL["Temp"] = FTCAMP_FINAL["Temp"].interpolate(method="polynomial", order=3)
    FTCAMP_FINAL = pd.pivot_table(FTCAMP_FINAL, index=["Year", "Month", "Day", "Hour"], values=["Temp"], aggfunc="mean")
    FTCAMP_FINAL.reset_index(inplace=True)
    FTCAMP_FINAL["DATE"] = pd.to_datetime(FTCAMP_FINAL["Year"].astype(str) + '-' +
                                        FTCAMP_FINAL["Month"].astype(str) + '-' +
                                        FTCAMP_FINAL["Day"].astype(str) + '-' +
                                        FTCAMP_FINAL["Hour"].astype(str),
                                        format="%Y-%m-%d-%H")
    FTCAMP_FINAL = FTCAMP_FINAL.set_index(["DATE"])
    FTCAMP_FINAL["Temp"].replace(0, np.NaN, inplace=True)

    Ft_Campbell_T = FTCAMP_FINAL["Temp"]
    Ft_Campbell_D = FTCAMP_FINAL.index
    now = datetime.now()
    cdate = now.strftime("%m-%d-%H%M")
    NASH_FINAL.to_csv("MID_Nashville_%s.csv" % (cdate))
    FTCAMP_FINAL.to_csv("MID_FTCambell_%s.csv" % (cdate))
# #############################
#
# Funcion="Pivot_y_m"
if Funcion=="Pivot_y_m":
    NASH_FINAL = pd.read_csv("END_Nashville.csv", index_col=0)
    FTCAMP_FINAL = pd.read_csv("END_FTCambell.csv", index_col=0)
    NASH_FINAL = pd.pivot_table(NASH_FINAL, index=["Year"],columns=["Month"], values=["Temp"], aggfunc="mean")
    FTCAMP_FINAL = pd.pivot_table(FTCAMP_FINAL, index=["Year"],columns=["Month"], values=["Temp"], aggfunc="mean")
    NASH_FINAL.columns = NASH_FINAL.columns.droplevel(0)
    FTCAMP_FINAL.columns = FTCAMP_FINAL.columns.droplevel(0)
    print(NASH_FINAL.head(100))
    print(FTCAMP_FINAL.head(100))
    now = datetime.now()
    cdate = now.strftime("%m-%d-%H%M")
    NASH_FINAL.to_csv("MID_Nashville_%s.csv" % (cdate))
    FTCAMP_FINAL.to_csv("MID_FTCambell_%s.csv" % (cdate))
#
# Funcion="analyse"
if Funcion=="analyse":
    NASH_FINAL = pd.read_csv("MID_FTCambell_READY.csv", index_col=0)
    NASH_cols=NASH_FINAL.columns.to_numpy()
    NASH_X = NASH_FINAL.index.to_numpy()



    coef=np.array([[[],[],[],[],[]]])
    X=NASH_X
    for mes in range(12):
        NASH_Y = NASH_FINAL[NASH_cols[mes]]
        Y=NASH_Y
        model=sm.OLS(Y,X)
        X_mes=model.fit()
        results=X_mes.params
        r2=X_mes.rsquared
        radj=X_mes.rsquared_adj
        t=X_mes.tvalues
        p=X_mes.pvalues
        coef_i =np.append(mes+1,results)
        coef_i=np.append(coef_i,r2)
        coef_i=np.append(coef_i,radj)
        coef_i=np.append(coef_i,t)
        coef_i=np.append(coef_i,p)
        coef=np.append(coef,coef_i)
        print(mes)
        print(X_mes.summary())
        print(coef_i)
    coef = coef.reshape((mes+1, 6))
    print(coef)

    with open("REGRESSIONS_FTC.csv",'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Slope",  "R2", "Radj", "t","P-values"])
        for rows in range(len(coef[:, 1])):
            writer.writerow(coef[rows, :])

# Funcion = "visualise"
if Funcion == "visualise":
    DATA = pd.read_csv("MID_FTCambell_READY.csv", index_col=0)
    MODEL_PARAMS = pd.read_csv("REGRESSIONS_F.csv", index_col=0)
    average_slope=MODEL_PARAMS["Slope"].mean()
    print(average_slope)
    YEARS= DATA.index.to_numpy()

    average_T0=58.5145140691303         #(FT_Camp=58.5145140691303  ,    NASH=  58.91769)
    YEARNUM=YEARS-min(YEARS)+1
    pred_year=np.array([])
    for years in YEARNUM:
        pred_year_i=average_T0+average_slope*years
        pred_year=np.append(pred_year,pred_year_i)
    DATA.columns=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    print(pred_year)
    LINEAR = pd.DataFrame(pred_year, columns=['Linear Reg (based on average month slope)'])
    LINEAR["YEARS"]=YEARS
    LINEAR=LINEAR.set_index("YEARS")
    print(LINEAR)

    font=20
    fontsm=12
    fn="Gulim"
    #

    plt.rc('xtick', labelsize=fontsm)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsm)
    ax1=DATA.plot(style="o-",alpha=0.7,linewidth=0.2)#,title="Nashville Month average and Linear regression")
    LINEAR.plot(ax=ax1, style="--", color="k", linewidth=0.6)
    plt.gcf().suptitle("Monthly average evolution and Linear regression- Ft. Campbell",fontsize=font, fontname=fn)
    plt.ylabel("Temperature-" r'$[°F]$',fontsize=font, fontname=fn)
    plt.xlabel('Year',fontsize=font, fontname=fn)
    plt.legend(loc='lower left', mode="expand",ncol=13 )
    plt.show()

# Funcion = "visualise2"
if Funcion == "visualise2":

    DATA = pd.read_csv("Slopes.csv", index_col=0)

    font = 20
    fontsm = 12
    fn = "Gulim"
    #

    plt.rc('xtick', labelsize=fontsm)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsm)
    ax1 = DATA.plot(style="s", alpha=0.7, linewidth=0.5)  # ,title="Nashville Month average and Linear regression")

    plt.gcf().suptitle("Temperature change rate by month", fontsize=font, fontname=fn)
    plt.ylabel("Temperature-" r'$[°C/100y]$', fontsize=fontsm, fontname=fn)
    plt.xlabel('Month', fontsize=fontsm, fontname=fn)
    plt.grid(True)
    plt.legend(loc='lower center')
    plt.show()



    # plt.title('Nashville Complete datasets (filtered years only)',fontsize=font, fontname=fn)
    # plt.plot(Nashville_D,Nashville_T, color="darkorange", linewidth=0.3, label="Nashville")
    # plt.ylabel('°F',fontsize=font, fontname=fn)
    # plt.xlabel('Year',fontsize=fontsm, fontname=fn)
    # plt.legend()
    # plt.show()

    # ax2 = FTCAMP_FINAL.plot(style="o",alpha=0.5)
    # plt.show()

# plt.scatter(x, y, color = "blue", alpha=0.5, marker = "o", s = 30)
#
#     # calculate prediction vector
#     y_pred = w[0] + w[1]*x
#
#     # plot the regression line
#     plt.plot(x, y_pred, color = "r" )
#
#     plt.xlabel('x')
#     plt.ylabel('y')
#
#     plt.show()



    # # font = {'family': 'serif',
    # #         'color':  'darkred',
    # #         'weight': 'normal',
    # #         'size': 16,
    # #         }
    #

    #
    # plt.title('Font Campbell Complete datasets (filtered years only)', fontsize=font, fontname=fn)
    # plt.plot(Ft_Campbell_D, Ft_Campbell_T, color="b", linewidth=0.3, label="Ft_Campbell")
    # plt.ylabel('°F', fontsize=font, fontname=fn)
    # plt.xlabel('Year', fontsize=fontsm, fontname=fn)
    # plt.legend()
    # plt.show()

    #


# Fig, axs = plt.subplots(2)
#     font=16
#     axs[0].set_title('Filtered years Complete datasets',fontsize=font, fontname="Times New Roman")
#     axs[0].plot(Nashville_D,Nashville_T, color="darkorange", linewidth=0.3, label="Nashville")
#     axs[0].set_ylabel('°F',fontsize=font)
#
#     # axs[0].set_title('Ft_Campbell')
#     axs[1].plot(Ft_Campbell_D,Ft_Campbell_T, color="c", linewidth=0.3, label="Ft_Campbell")
#     axs[1].set_ylabel('°F', fontsize=font)
#     plt.legend()
#     plt.legend()
#     plt.show()


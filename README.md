# CLTV-Prediction-with-BGNBD-GG-Model
CLTV= Expected Number of Transaction * Expected Average Profit There are two statistical distributions to calculate CLTV and these are BG/NBD model and GG Submodel. Both models make predictions by reducing the mass to the person. The last version of formula is “CLTV= BG/NBD Model * GG Submodel”. In addition, this model helps to make predictions on any time period basis.
from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

creds = {'user': '',
         'passwd': '',
         'host': '',
         'port': ,
         'db': ''}

connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'
conn = create_engine(connstr.format(**creds))
retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)
df=retail_mysql_df.copy()
df.head()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#datacleaning & preparing
df.describe()
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df["Quantity"] > 0) & (df['Price'] > 0)]
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df['TotalPrice'] = df['Price'] * df["Quantity"]
df["InvoiceDate"].max()
today_date = dt.datetime(2011,12,11)
df= df[df['Country']=='United Kingdom']
df.columns
# Lifetime Veri Yapısının Hazırlanması
# recency: Son satın alma üzerinden geçen zaman. Haftalık. (cltv_df'de analiz gününe göre, burada kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç
cltv_df = df.groupby('CustomerID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
#monetary ortalama
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
#sifirdan buyuk olanlari siliyoruz
cltv_df = cltv_df[cltv_df["monetary"] > 0]
cltv_df.head()
# frequency'nin 1'den büyük olması gerekmektedir
cltv_df = cltv_df[cltv_df["frequency"] > 1]
#BGNBD için recency ve T'nin haftalık cinsten ifade edilmesi
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7
print("#######gorev2#######")
#2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.

#BG-NBD Modeli
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])


#GammaGamma Modeli
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

#6 aylık Tahmin Modelimiz
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,#ay
                                   freq="W",
                                   discount_rate=0.01)

cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)
cltv_final = cltv_df.merge(cltv, on="CustomerID", how="left")
cltv_final.sort_values(by="clv", ascending=False)[10:30]
print("#######gorev2#######")
#Farklı Zaman Periyotlarından oluşan CLTV Analizi
#▪ 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
#▪ 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.
#▪ Fark var mı? Varsa sizce neden olabilir?

cltv1 = ggf.customer_lifetime_value(bgf,
                                    cltv_df['frequency'],
                                    cltv_df['recency'],
                                    cltv_df['T'],
                                    cltv_df['monetary'],
                                    time=1,  # months
                                    freq="W",  # T haftalık
                                    discount_rate=0.01)
rfm_cltv1_final = cltv_df.merge(cltv1, on="CustomerID", how="left")
rfm_cltv1_final.sort_values(by="clv", ascending=False).head(10)

cltv12 = ggf.customer_lifetime_value(bgf,
                                     cltv_df['frequency'],
                                     cltv_df['recency'],
                                     cltv_df['T'],
                                     cltv_df['monetary'],
                                     time=12,  # months
                                     freq="W",  # T haftalık
                                     discount_rate=0.01)

rfm_cltv12_final = cltv_df.merge(cltv12, on="CustomerID", how="left")
rfm_cltv12_final.sort_values("clv", ascending=False).head(10)

print("#######gorev3#######")
#Segmentasyon ve Aksiyon Önerileri
#▪ 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba
#(segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
#▪ 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon
#önerilerinde bulununuz

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_final = cltv_df.merge(cltv, on="CustomerID", how="left")
cltv_final.head()


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])


cltv_final["cltv_segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()

cltv_final[cltv_final["cltv_segment"]=="A"].shape


print("#######gorev4#######")
cltv_final=cltv_final.reset_index()
cltv_final["CustomerID"] = cltv_final["CustomerID"].astype("int64")
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'
conn = create_engine(connstr.format(**creds))
cltv_final.to_sql(name='BITA_AZARI', con=conn, if_exists='replace', index=False)
pd.read_sql_query("show tables", conn)

pd.read_sql_query("select * from BITA_AZARI limit 15", conn)

pd.read_sql_query("show tables", conn)



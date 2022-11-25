import numpy as np
import pandas as pd
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import tarfile
import gzip
import re
import os
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

init_notebook_mode(connected=True)

#stat_num = 1000 # Number of stations to plot for testing
year_num = 20 # Number of past years to consider
extremes_num = 10 # Number of hottest and coldest places to display

yearfiles = os.listdir("D:/Big data/archive/gsod_all_years")
yearfiles.sort()
yearfiles = yearfiles[-year_num:]
#need to check
years = [int(re.findall('\d+',yearfile)[0]) for yearfile in yearfiles]

station_loc = pd.read_csv('D:/Big data/archive/isd-history.csv')
station_loc = station_loc.replace([0.0, -999.0, -999.9],np.nan)
station_loc = station_loc[pd.notnull(station_loc['LAT']) & pd.notnull(station_loc['LON'])]
#need to check
station_loc = station_loc[[int(re.findall('^\d{4}', str(end_year))[0])==max(years) for end_year in station_loc['END']]]
station_loc = station_loc[[int(re.findall('^\d{4}', str(beg_year))[0])<=min(years) for beg_year in station_loc['BEGIN']]]
#print station_loc

station_loc['LBL'] = station_loc[['STATION NAME','STATE','CTRY']].apply(lambda x: x.str.cat(sep=', '), axis=1)
station_loc['ELEV_LBL'] = station_loc['ELEV(M)'].apply(lambda x: 'Elevation: '+str(x)+' m' if ~np.isnan(x) else np.nan)
station_loc['LBL'] = station_loc[['LBL','ELEV_LBL']].apply(lambda x: x.str.cat(sep='<br>'), axis=1)
station_loc = station_loc.drop(['STATION NAME','STATE','ELEV_LBL','ICAO','BEGIN','END'], axis=1)
#station_loc = station_loc.sample(stat_num)

df = pd.DataFrame([])
df_day = pd.DataFrame([])

def preprocess_station_file_content(content):
    headers=content.pop(0)
    headers=[headers[ind] for ind in [0,1,2,3,4,8,11,12,13]]
    for d in range(len(content)):
        content[d]=[content[d][ind] for ind in [0,1,2,3,5,13,17,18,19]]
    content=pd.DataFrame(content, columns=headers)
    content.rename(columns={'STN---': 'USAF'}, inplace=True)
    content['MAX'] = content['MAX'].apply(lambda x: re.sub("\*$","",x))
    content['MIN'] = content['MIN'].apply(lambda x: re.sub("\*$","",x))
    #content['PRCP'] = content['PRCP'].apply(lambda x: re.sub("\+$","",x))
    #content['PRCP'] = content['PRCP'].apply(lambda x: x.replace("G",""))
    #content['PRCP'] = content['PRCP'].apply(lambda x: x.replace(x[len(x) - 1:], ""))
    content['PRCP'] = content['PRCP'].apply(lambda x: re.sub(x,x[:-1],x))
    content[['WBAN','TEMP','DEWP','WDSP','MAX','MIN','PRCP']] = content[['WBAN','TEMP','DEWP','WDSP','MAX','MIN','PRCP']].apply(pd.to_numeric)
    content['YEARMODA']=pd.to_datetime(content['YEARMODA'], format='%Y%m%d', errors='ignore')
    content['YEAR']=pd.DatetimeIndex(content['YEARMODA']).year
    content['MONTH']=pd.DatetimeIndex(content['YEARMODA']).month
    content['DAY']=pd.DatetimeIndex(content['YEARMODA']).day
    return content

yearfile = yearfiles[-1]
print(yearfile)
i=0
tar = tarfile.open("D:/Big data/archive/gsod_all_years/"+yearfile, "r")
print(len(tar.getmembers()[1:]))
#for member in np.random.choice(tar.getmembers()[1:], size=stat_num, replace=False):
for member in tar.getmembers()[1:]:
    name_parts = re.sub("\.op\.gz$","",re.sub("^\./","",member.name)).split("-")
    usaf = name_parts[0]
    wban = int(name_parts[1])
    if station_loc[(station_loc['USAF']==usaf) & (station_loc['WBAN']==wban)].shape[0]!=0:
        i=i+1
        #if i%(stat_num//10) == 0: print(i)
        f=tar.extractfile(member)
        f=gzip.open(f, 'rb')
        content=[re.sub(" +", ",", line.decode("utf-8")).split(",") for line in f.readlines()]
        content=preprocess_station_file_content(content)
        #print (content)
        df_day = df_day.append(content[content['YEARMODA']==content['YEARMODA'].max()])
        content = content.groupby(['USAF','WBAN','YEAR','MONTH']).agg('median').reset_index()
        df = df.append(content)
tar.close()

import findspark
findspark.init()

from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# Load and parse the data file, converting it to a DataFrame.
data = df
features = data.iloc[:,3]
indexedFeatures = data.iloc[:,8]

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestRegressor(featuresCol="indexedFeatures")

# Chain indexer and forest in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, rf])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

rfModel = model.stages[1]
print(rfModel)  # summary only
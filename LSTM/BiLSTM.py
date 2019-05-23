import numpy as np
import pandas as pd
from keras.models import Sequential 
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error as mse

input_file = "train.csv"
df = pd.read_csv(input_file)

def remove_duplicates(df):
	ColoumnArr = np.array(df.ix[:, 'Timestamp'])
	i=0
	ArrLen = len(ColoumnArr)
	index_duplicate= []
	# identify duplicates by index
	while(i<ArrLen-1):
		if ColoumnArr[i]==ColoumnArr[i+1]:
			index_duplicate.append(i+1)
		i+=1
	# remove duplicates
	df=df.drop(index_duplicate)
	return df


def avg_over_time(df, indexCol=0):
	avg={}
	colLen= df.shape[0]
	for x in range(colLen):
		time = str(df[x][indexCol])[11:]
		if time not in avg:
			avg[time]= [[0.0, 0.0, 0.0, 0.0, 0.0], 0]
		for colNo in range(1,6):
			avg[time][0][colNo-1]+= float(df[x][colNo])
		avg[time][1]+=1
	for key, val in avg.items():
		avg[key]= [x*1.0/val[1] for x in val[0]]
	return avg


def replace_noise(df, indexCol=0):
	avg= avg_over_time(df, indexCol)
	colLen= df.shape[0]
	for x in range(colLen):
		time = str(df[x][indexCol])[11:]
		for col in range(1,6):
			if df[x][col]==0:
				try:
					df[x][col]= avg[time][col-1]
				except KeyError:
					print(x)
					print(col)
	return df

df= df.drop(['Label'], axis=1)
df= df.replace(np.nan, 0)
df= remove_duplicates(df).values
df= replace_noise(df)
print(df[0])

#data preperation 
seq_length= 100
DataX= []
DataY= []

for x in range(len(df)-seq_length):
	SeqX= df[x: x+seq_length, 1:6]
	SeqY= df[x+seq_length, 5]
	DataX.append(SeqX)
	DataY.append(SeqY)

DataX=np.array(DataX)
DataY=np.array(DataY)

# transforming to (samples, seq length, features)
DataX= np.reshape(DataX, (DataX.shape[0], DataX.shape[1], DataX.shape[2]))
DataY= np.reshape(DataY, (DataY.shape[0], 1))
TrainDataX= DataX[:int(0.7*len(DataX))]
TrainDataY= DataY[:int(0.7*len(DataY))]
TestDataX= DataX[int(0.7*len(DataX)):]
TestDataY= DataY[int(0.7*len(DataY)):]

#developing the model
model = Sequential()
model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=(TrainDataX.shape[1],TrainDataX.shape[2])))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.2))
model.add(Dense(TrainDataY.shape[1]))
filename="best_weight.hdf5"
model.load_weights(filename)
model.compile(loss='mean_squared_error', optimizer='adam')
#checkpoint = ModelCheckpoint(filepath, monitor= 'loss' , verbose=1, save_best_only=True,
#mode= min )
#callbacks_list = [checkpoint]
# fit the model
#model.fit(TrainDataX, TrainDataY, nb_epoch=50, batch_size=64, callbacks= callbacks_list)
print("loaded weights")
y = model.predict(TestDataX)
print("predicted")
print(mse(TestDataY, y))

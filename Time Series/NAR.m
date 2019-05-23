clear all
%Importing data from dataset
data_input=xlsread('datasets/Wallmart_sample.xlsx','A2:A175001');
data_output=xlsread('datasets/Wallmart_sample.xlsx','B2:B175001');


%Batched Mean Normalization- Batch_size=100
% For input data
batch_size=100;
batch_input100=[];
for i=0:1749
    temp=data_input(1+i*batch_size:batch_size*(i+1),1);
    batch_input100=[batch_input100; mean(temp)]
end

%Batched Mean Normalization- Batch_size=100
% For output data
batch_output100=[];
for i=0:1749
    temp=data_input(1+i*batch_size:batch_size*(i+1),1);
    batch_output100=[batch_output100; mean(temp)]
end


%Taking some samples for testing
samples=data_input(15300:15400,1); 
actual=data_output(15300:15400);

%Trasnposing input,output vector (ANN only accepts column wise)
batch_input100=batch_input100';
batch_output100=batch_output100';


%Creating ANN model for organic data
myANN=newff(minmax(batch_input100),[2,1],{'tansig','purelin'});
[myANN, tr]=train(myANN,batch_input100,batch_output100);

%Testing for 100 samples
samples=samples';
predicted=myANN(samples);
mse=perform(myANN,predicted,actual);
predicted=predicted';

x=1:101;
y=[predicted actual]

figure; 
for i = 1:101 
    plot(x(1:i),y(1:i,:));
    pause(0.05) 
end


clc
clear all
close all
fpath="E:\Btech Final Yr Project\Datasets"

train_imds=imageDatastore(fpath,'IncludeSubfolders',true,'LabelSource','foldernames');
targetSize=[227, 227];

for i = 1:numel(train_imds.Files)
    img=imread(train_imds.Files{i});
    img=imresize(img, targetSize);
    imwrite(img,train_imds.Files{i});
end

data=fullfile(fpath,'train');
data2=fullfile(fpath,'test');

testdata=imageDatastore(data2,"IncludeSubfolders",true,'LabelSource','foldernames');
traindata=imageDatastore(data,'IncludeSubfolders',true,'LabelSource','foldernames');
count=traindata.countEachLabel;

net=alexnet;
layers=[imageInputLayer([227,227,3])
    net(2:end-3)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer()
    ];

opt=trainingOptions('sgdm','MaxEpochs',20,"InitialLearnRate",0.1);
training=trainNetwork(traindata,layers,opt)

save training
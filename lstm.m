Xtrain=mts.train;
Ytrain=categorical(mts.trainlabels);%将数值数组转化为类别数组
Xtest=mts.test;
Ytest=categorical(mts.testlabels);
%% 原始数据特征分析
figure(1)
plot(Xtrain{1}')
%xlabel("时间步" )
xlabel({'时间步';'(3) 算法2'})
ylabel("属性值")
legend("特征 " + string(1:41))%%%%特征维度

%% 构建LSTM网络
inputSize = 62;%%%%特征的维度
numHiddenUnits = 50;%LSTM网络隐藏神经元数目
numClasses = 2;%%%%label标签，类别数
%% lstm隐含层构建
layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
%% 参数初始化
maxEpochs = 50;%最大训练周期数
miniBatchSize = 128;%批量化
learning_rate = 0.001;%学习率
options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',learning_rate, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');
%% 训练
[net,info]=trainNetwork(Xtrain,Ytrain,layers, options);
%% 预测
YPred = classify(net,Xtest, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');
%% 精确度检验
acc = sum(YPred == Ytest)./numel(Ytest);
%% 结果可视化
s= string(YPred);
Ypred = double(s);
figure(1)
plot(Ypred,'*');
grid on;
hold on;
plot(mts.testlabels,'d');
Text_E=num2str(acc*100);
legend('预测类别','真实类别');
xlabel('测试样本个数');
ylabel('类别标签');
title(['准确率为',Text_E,'%'])
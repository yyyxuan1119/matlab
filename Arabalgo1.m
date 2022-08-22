clear all;
close all;
clc;
%%%%% 数据
load ArabicDigits;%%%%导入原始数据
D=eye(13,6600);%%% 
for  k=1:1:6600%%%样本
A=mts.train{1,1};
B=mts.train{1,k};
m=size(A,2);
n=size(B,2);
if(m<=n)
      B(:,m+1:n)=[];  
else
      A(:,n+1:m)=[];          
end
E{k}=A-B; 
D(:,k)=sqrt(sum(E{k}.^2,2));
end
C=sort(D,2);
C(:,1)=[];
K=2;
MIN=C(:,1:1:K);
MAX=C(:,m-K:1:m-1);
%MAX=C(:,k-K:1:k-1); %%CMU数据集使用
AMIN=sum(MIN,2)/2;
AMAX=sum(MAX,2)/2;
I=AMAX./AMIN;
[I1 I2]=sort(I,1,'descend');
i=1;
a=0;
b=sum(I);
k=13;%%%特征
for i=1:1:k
    a=a+I(i)/b;
    fprintf('符合的值为%f\n',I1(i))
    fprintf('对应的属性为%d\n',I2(i))
    if a<=0.8
        continue
    end
    break
end
%%训练集删除不显著的特征
for num=1:1:6600%%%%
    mts.train{num}(3,:)=[];
    mts.train{num}(3,:)=[];%%%%这里需要判读
end                          
%%测试集
for num=1:1:2200%%%%%
    mts.test{num}(3,:)=[];
    mts.test{num}(3,:)=[];

end 
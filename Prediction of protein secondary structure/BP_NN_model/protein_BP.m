clc;
clear all;
load datax;


%特征归一化
[input,xmin,xmax] = premnmx(trainvectorx');


%构造输出矩阵
output=trainvectory;

% 隐藏层和输出层使用Sigmoid传输函数

%使用 'traingdx' 自适应调整学习速率

% learn属性 'learngdm' 附加动量因子的梯度下降学习函数
net = newff( minmax(input) , [3000 400], {'tansig' 'tansig'} , 'traingdx' , 'learngdm') ;

%设置训练参数
net.trainparam.show = 1000;%每间隔步显示一次训练结果

net.trainparam.epochs = 15000 ;%允许最大训练步数500步

net.trainparam.goal = 0.01 ;%训练目标最小误差0.01

net.trainParam.lr = 0.05 ;%学习速率0.05

%开始训练
net = train( net, input , output' ) ;

%读取测试数据


%测试数据归一化
[input,amin,xmax] = premnmx( testvectorx')  ;
testInput = tramnmx( testvectorx' , amin,xmax) ;
Y = sim( net , testInput ) ;
%统计识别正确率
[s1 , s2] = size( Y ) ;
[d1 , d2] = size(testvectory');
testvectory=testvectory';
hitNum = 0 ;
point=1;jk=1;
for i = 1 : s2
 while point<400  
 [m , Index] = max( Y(point :point+3 , i ) ) ;
 result(jk)=Index;
 jk=jk+1;
 point=point+4;

 end
if point>400
     point=1
 end
end

point=1;jk=1;
for i = 1 : d2
 while point<400  
 [m , Index2] = max(testvectory(point :point+3 ,i)) ;
 Y_test(jk)=Index2;
 jk=jk+1;
 point=point+4;

 end
if point>400
     point=1
 end
end

for i=1:length(Y_test)
    if Y_test(i)==result(i);
        hitNum=hitNum + 1 ;
    end
end

sprintf('识别率是 %.3f%%',100 * hitNum / (d1*d2/4) )

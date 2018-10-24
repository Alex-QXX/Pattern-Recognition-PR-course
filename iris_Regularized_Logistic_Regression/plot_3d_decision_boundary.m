% 加载参数
load('./data/W1.mat');
load('./data/d1.mat');

% 加载原始数据 前两类
load('./data/data_x0');
load('./data/data_y0');
load('./data/data_z0');

load('./data/data_x1');
load('./data/data_y2');
load('./data/data_z3');
% 画出01类别的三维决策边界，是一个3维隐函数
solve_z=solve('W1(1)*x+W1(2)*y+W1(3)*z+W1(4)*x^2+W1(5)*y^2+W1(6)*z^2+W1(7)*x*y+W1(8)*y*z+W1(9)*x*z+b1','z');

% 画隐函数图
ezmesh(solve_z(1))
hold on
scatter3(x0,y0,z0,'*','r')
hold on 
scatter3(x1,y1,z1,'.','g')
view([30,45])

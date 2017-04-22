%% KELM plot
close all
clear all

dataset = 'MNIST';                  % for graph titles and file load



% --------- LOAD FILES ----------------------------------------------------
kelmrbffile = strcat(dataset,'rbf3');               % kelm-data
kelmpoly2file = strcat(dataset,'poly');
kelmpoly1file = strcat(dataset,'poly1');
elmfile = strcat(dataset,'_ELM_with_penalty');      % elm-data

% ------------ KELM RBF ---------------------------------------------------
load(kelmrbffile);
rbfparam = kernelparam;
% For MNIST, toggle comments below ----------
%acc = mean(acc) ;                      % commented for MNIST
%time = mean(classTime + trainTime);    % commented for MNIST
time = classTime + trainTime;           % activated for MNIST

rbfline = 'k*-';
figure(1)
hold on
plot(datasize,acc,rbfline);
figure(2)
hold on
plot(datasize,time,rbfline);
figure(3)
hold on
plot(acc,time,rbfline);


% -------------- KELM POLY 2 -----------------------------------------------
load(kelmpoly2file);
poly2param = kernelparam;
% For MNIST, toggle comments below ----------
%acc = mean(acc) ;                      % commented for MNIST
%time = mean(classTime + trainTime);    % commented for MNIST
time = classTime + trainTime;           % activated for MNIST

poly2line = 'k+:';
figure(1)
hold on
plot(datasize,acc,poly2line);
figure(2)
hold on
plot(datasize,time,poly2line);
figure(3)
hold on
plot(acc,time,poly2line);

% -------------- KELM POLY 1 -----------------------------------------------
load(kelmpoly1file);
poly1param = kernelparam;
% For MNIST, toggle comments below ----------
%acc = mean(acc) ;                      % commented for MNIST
%time = mean(classTime + trainTime);    % commented for MNIST
time = classTime + trainTime;           % activated for MNIST

poly1line = 'kd-.';
figure(1)
hold on
plot(datasize,acc,poly1line);
figure(2)
hold on
plot(datasize,time,poly1line);
figure(3)
hold on
plot(acc,time,poly1line);


% -------------- ELM plot -------------------------------------------------

load(elmfile);
% For MNIST, toggle comments below ----------
%acc = mean(acc) ;                      % commented for MNIST
%time = mean(classTime + trainTime);    % commented for MNIST
time = classTime + trainTime;           % activated for MNIST
%-------------------------

elmline = 'kx--';
figure(1)
hold on
plot(datasize,acc,elmline);
figure(2)
hold on
plot(datasize,time,elmline);
figure(3)
hold on
plot(acc,time,elmline);



% ----- Labels, titles etc ---------------------------------------
try
    cd /Users/Viktor/Dropbox/KTH/År 3/Period 4/Kex/Saved Data
catch
    
end

figure(1)
xlabel('Fraction of total database used for training');
if strcmp(dataset,'MNIST')
    xlabel('Number of feature vectors used for training');
end
ylabel('Accuracy');
title(dataset);
legend('KELM RBF','KELM Polynomial','KELM Linear','ELM','location','northwest')
savefig(strcat(dataset,'_figure_',num2str(1)))
saveas(figure(1),strcat(dataset,'_figure_',num2str(1),'.png'))

figure(2)
xlabel('Fraction of total database used for training');
if strcmp(dataset,'MNIST')
    xlabel('Number of feature vectors used for training');
end
ylabel('Training and classification time [s]');
title(dataset);
legend('KELM RBF','KELM Polynomial','KELM Linear','ELM','location','northwest')
savefig(strcat(dataset,'_figure_',num2str(2)))
saveas(figure(2),strcat(dataset,'_figure_',num2str(2),'.png'))

figure(3)
ylabel('Training and classification time [s]');
xlabel('Accuracy');
title(dataset);
legend('KELM RBF','KELM Polynomial','KELM Linear','ELM','location','northwest')
savefig(strcat(dataset,'_figure_',num2str(3)))
saveas(figure(3),strcat(dataset,'_figure_',num2str(3),'.png'))

movegui(1,'northwest')
movegui(2,'north')
movegui(3,'northeast')




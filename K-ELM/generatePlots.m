%% KELM plot
close all
clear all

dataset = 'MNIST';                  % for graph titles

kelmfile = strcat(dataset,'rbf3');             % kelm-data
elmfile = strcat(dataset,'_ELM_with_penalty'); % elm-data

load(kelmfile);

% For MNIST, toggle comments below ----------
%acc = mean(acc) ;                      % commented for MNIST
%time = mean(classTime + trainTime);    % commented for MNIST
time = classTime + trainTime;           % activated for MNIST
%-------------------------

figure(1)
plot(datasize,acc);

figure(2)
plot(datasize,time);

figure(3)
plot(acc,time);

% ELM plot

load(elmfile);

% For MNIST, toggle comments below ----------
%acc = mean(acc) ;                      % commented for MNIST
%time = mean(classTime + trainTime);    % commented for MNIST
time = classTime + trainTime;           % activated for MNIST
%-------------------------

figure(1)
hold on
plot(datasize,acc);
xlabel('fraction of total database used for training');
ylabel('accuracy');
title(strcat('Results on',' ',dataset));
legend(strcat('KELM',kernel,num2str(kernelparam)),'ELM','location','best')

figure(2)
hold on
plot(datasize,time);
xlabel('fraction of total database used for training');
ylabel('time [s]');
title(strcat('Results on ',dataset));
legend(strcat('KELM',kernel,num2str(kernelparam)),'ELM','location','best')

figure(3)
hold on
plot(acc,time);
ylabel('time[s]');
xlabel('accuracy');
title(strcat('Results on ',dataset));
legend(strcat('KELM',kernel,num2str(kernelparam)),'ELM','location','best')

movegui(1,'northwest')
movegui(2,'north')
movegui(3,'northeast')




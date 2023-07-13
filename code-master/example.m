T=2;                       % real time [0,2]

del=0.01;                  % integration interval
truetime=0:del:T;
Tindex=length(truetime);

Tdel=0.2;                  % sample interval- In the paper [F. Dondelinger et al, AISTATS 2013], it is mentioned that the sample rate is 0.25. But in the experiment we can see that there are 11 data points, which means that the sample rate is 0.2
samptime=0:Tdel:T;         % sample times
TTT=length(samptime);      % number of sampled points

itrue=round(samptime./del+ones(1,TTT));

xx(1,:) = [5 3]
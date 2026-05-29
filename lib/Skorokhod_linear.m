% -------------------------------------------------------------------------
% THIRD-PARTY CODE — NOT covered by this repository's MIT license.
%
% Skorokhod reflection map, part of the multi-level Monte Carlo simulator of:
%   J. Blanchet, X. Chen, N. Si, P. W. Glynn, "Efficient steady-state
%   simulation of high-dimensional stochastic networks," Stochastic Systems
%   11 (2021) 174-192.
% Obtained from the authors and redistributed here with their permission.
% Copyright (c) the original authors. All rights reserved.
% -------------------------------------------------------------------------
function[Y]=Skorokhod_linear(dim,X,R)
%X,Y are column vectors
Y=X;
eps=1e-8;
while sum(Y<-eps) >0
    base=find(Y<eps);
    R_b=R(base,base);
    if size(base)>0
    L_b=-R_b\X(base);
    Y=X+R(:,base)*L_b;
    end
end
end
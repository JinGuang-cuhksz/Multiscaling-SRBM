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
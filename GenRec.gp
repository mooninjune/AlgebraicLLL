/*******************************************************************************
           AN ALGORITHM FOR FINDING A SHORT GENERATOR IN Q(\ZETA_512)
                   IMPLEMENTED BY ALEXANDRE GÉLIN - LIP6/UPMC
*******************************************************************************/
\\ This code has been taken from (https://alexgelin.fr/index_en.html)


\\build a random element in nf with coefficients in [-mcoeff..mcoeff]
nfrandom(mcoeff,nf)={
pol=nf.pol;
d=poldegree(pol)-1;
polrand=random(2*mcoeff*t^d)-mcoeff*sum(i=0,d,t^i);
return(polrand)
};

\\construction of the log-unit lattice
LogUnit()={
LU=List([]);
M=matrix(n4m1,n4);
for(i=1,n4m1, lu=Mod((1-t^(2*i+1))/(1-t),PolC);
   listput(LU,lift(lu));
   conjugates=conjvec(lu);
   for(j=1,n4,M[i,j]=log(abs(conjugates[2*j]))));
MD=mattranspose((M*mattranspose(M))^-1)*M;
};

\\short generator recovery [CDPR]
ShortGenRec(gen)={
Y=matrix(1,n4);
conjugates=conjvec(Mod(gen,PolC));
for(j=1,n4,Y[1,j]=log(abs(conjugates[2*j])));
X=Y*mattranspose(MD);
for(i=1,n4m1,X[1,i]=round(X[1,i]));
u=prod(i=1,n4m1,LU[i]^X[1,i]);
return(lift(Mod(gen/u,PolC)))
};

\\recover the secret key (up to sign)
SVRec(gen)={
i=0;
while(polcoeff(gen,i)%2==0,i+=1);
return(lift(Mod(gen*t^(n2-i),PolC)))
};

\\find g s.t. g*OK = a*OK+b*OK
\\param n: degree of pow-of-2 cyclo field

\\dont forget to set n
\\construction of the field
n2=n/2; n4=n/4; n4m1=n4-1;
PolC=polcyclo(n,t);
dim=poldegree(PolC);
NF=nfinit(PolC);

allocatemem(2^10*11*10^6)
\\for the reduction to the totally real subfield
\r ./GentrySzydlo.gp

AB = idealhnf(NF,g0,g1);
[GSmat,GSgen]=GentrySzydlo(AB);
BKZin=mattranspose(GSmat[1..n4,]);

/*
det=matdet(BKZin);
\\External call to fplll for the BKZ reduction
write("BKZin",BKZin);
str=concat("bash scriptBKZ 24 ",Str(n4));
externstr(str);
\r BKZout
gen=redmat[1][1]+sum(i=1,n4m1,redmat[1][i+1]*(t^i-t^(n2-i)));
\\BKZ24 suffices for most instances but if not, use BKZ30
if(idealnorm(NF,gen)!=det^2, str=concat("bash scriptBKZ 30 ",Str(n4)); externstr(str); read("BKZout"); gen=redmat[1][1]+sum(i=1,n4m1,redmat[1][i+1]*(t^i-t^(n2-i))));
\\Recover the private-key (up to sign)
Gen=lift(gen/Mod(1+GSgen,PolC));
\\LogUnit();
\\SGen=ShortGenRec(Gen);
\\res=SVRec(SGen);

res = Gen

norm_err = idealnorm(NF,gen) / det^2;

return(res)
*/

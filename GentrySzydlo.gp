/*******************************************************************************
           AN ALGORITHM FOR FINDING A SHORT GENERATOR IN Q(\ZETA_512)
                   IMPLEMENTED BY ALEXANDRE GÉLIN - LIP6/UPMC
*******************************************************************************/

\\ This is code written by Joseph de Vilmarest during an
\\ internship under the supervision of Pierre-Alain Fouque.
\\ We make use of it for going through our attack.
\\ Source: (https://alexgelin.fr/index_en.html) 



Mch=(matrix(dim,dim,i,j,polcoeff(NF.zk[j],i-1,t)))^-1;
zkofpol(U)=Mch*matrix(dim,1,i,j,polcoeff(U,i-1,t));

findep()={
seuil=1.1^n;
e=n;
p=1;
while(p<seuil,e+=n;p=1;fordiv(e,k,if((k%n==0)&&isprime(k+1),p*=(k+1))));
return([e,p])
};

ep=findep();
e=ep[1];
p=ep[2];
factorp=factor(p);

inversemodp(pol)={
rep=1/Mod(Mod(pol,PolC),factorp[1,1]);
for(i=2,matsize(factorp)[1],rep=chinese(rep,1/Mod(Mod(pol,PolC),factorp[i,1])));
return(rep)
};

findq()={
q=2;
while((gcd(q,2*n)!=1)||(if(n%2,2*n,n)%gcd((q^znorder(Mod(q,n))-1)*q,e)!=0),q=nextprime(q+1));
return(q)
};

findk(q,zn)={
seuil=1.2^(dim^2/zn);
return(floor(log(seuil)/log(q))+1)
};

q=findq();
zn=znorder(Mod(q,n));
k=findk(q,zn);

b=bezout(e,(q^zn-1)*q^(k-1));
g=b[3];
w=b[1]%((q^zn-1)*q^(k-1));
Pfactor=lift((factorpadic(PolC,q,k))[1,1]);

accesscol(M,j)=sum(i=1,dim,M[i,j]*NF.zk[i]);

polconj(U)=sum(i=0,poldegree(U,t),polcoeff(U,i,t)*t^(n-i));

polnorm(U)=Mod(U*polconj(U),PolC);

gram(ideal,r)={
l=length(ideal);
base=vector(l,i,1.*accesscol(ideal,i));
inv=1./r;
M=matrix(l,l);
for(i=1,l,for(j=1,i,mult=Mod(base[i]*polconj(base[j]),PolC);coeff=trace(inv*(mult+Mod(polconj(lift(mult)),PolC)))/2;M[i,j]=coeff;M[j,i]=coeff));
return(M)
};

commondenom(m)={
d=1;
for(i=1,length(m[,1]),for(j=1,length(m[1,]),d=lcm(d,denominator(m[i,j]))));
return(d)
};

commondenomvect(vect)={
d=1;
for(i=1,length(vect),d=lcm(d,denominator(vect[i])));
return(d)
};

idealmultiply(ideal1,ideal2)={
id1_lll = ideal1*qflll(ideal1);
id2_lll = ideal2*qflll(ideal2);
deter=abs(matdet(id1_lll)*matdet(id2_lll));
l1=length(ideal1);
l2=length(ideal2);
l=max(l1,l2);
famgene=matrix(l,2*l);
debut=0;
while(1,famgene=concat(famgene,matrix(l,2*l-matsize(famgene)[2]));
for(j=debut+1,2*l,polj=(NF.zk[1+random(dim)]*(accesscol(ideal1,1+random(l1))+accesscol(ideal1,1+random(l1))+accesscol(ideal1,1+random(l1)))*(accesscol(ideal2,1+random(l2))+accesscol(ideal2,1+random(l2))+accesscol(ideal2,1+random(l2))))%PolC;poljzk=zkofpol(polj);for(i=1,dim,famgene[i,j]=poljzk[i,1]));
c=commondenom(famgene);
if(c==1,famgene=mathnfmod(famgene,deter),famgene=mathnf(c*famgene)/c);
debut=matsize(famgene)[2];
famgene_lll = famgene*qflll(famgene);
if(debut==l&&deter==abs(matdet(famgene_lll)),famgene=famgene_lll;return(famgene));
)
};

idealmulelt(ideal,elt)={
size=matsize(ideal);
rep=matrix(size[1],size[2]);
for(j=1,size[2],polj=(accesscol(ideal,j)*elt)%PolC;poljzk=zkofpol(polj);for(i=1,size[1],rep[i,j]=poljzk[i,1]));
return(rep)
};

reduction(ideal,r)={
G=gram(ideal,r);
B=ideal*qflllgram(G);
for(i=1,dim,y=accesscol(B,i);if(gcd(polresultant(y,PolC),p)==1,mult=mathnf(idealmulelt(idealinv(NF,mathnf(B)),y));return([mult,y])));
};

powering(ideal,r,e)={
if(e==0,return([idealhnf(NF,1),polnorm(1),Mod(Mod(1,PolC),p),Mod(Mod(1,Pfactor),q^k)]));
if(e==1,return([ideal,r,Mod(Mod(1,PolC),p),Mod(Mod(1,Pfactor),q^k)]));
if(e%2==0,
if(e%4==0,Kus=powering(ideal,r,e/4);Cx1=reduction(idealmultiply(Kus[1],Kus[1]),Kus[2]^2);Cx2=reduction(idealmultiply(Cx1[1],Cx1[1]),polnorm(Cx1[2])^2/(Kus[2]^4));return([Cx2[1],Kus[2]^4*polnorm(Cx2[2])/(polnorm(Cx1[2])^2),Kus[3]^4*Cx1[2]^2*inversemodp(Cx2[2]),Kus[4]^4*Cx1[2]^2/Cx2[2]]));
Kus=powering(ideal,r,e/2-1);
Cx1=reduction(idealmultiply(Kus[1],ideal),r*Kus[2]);
Cx2=reduction(idealmultiply(Cx1[1],Cx1[1]),
polnorm(Cx1[2])^2/(Kus[2]*r)^2);
return([Cx2[1],(Kus[2]*r)^2*polnorm(Cx2[2])/polnorm(Cx1[2])^2,Kus[3]^2*Cx1[2]^2*inversemodp(Cx2[2]),Kus[4]^2*Cx1[2]^2/Cx2[2]]));
if(e%2==1,Kus=powering(ideal,r,(e+1)/2);Cx1=reduction(idealmultiply(Kus[1],Kus[1]),Kus[2]^2);Cx2=reduction(idealmultiply(Cx1[1],ideal),r*polnorm(Cx1[2])/Kus[2]^2);return([Cx2[1],Kus[2]^2*polnorm(Cx2[2])/(r*polnorm(Cx1[2])),Kus[3]^2*Cx1[2]*inversemodp(Cx2[2]),Kus[4]^2*Cx1[2]/Cx2[2]]))
};

henselstep(r,i,vg)={
dPr=g*(r*Mod(Mod(1,Pfactor),q))^(g-1);
if(dPr==0,return(r));
dPrinv=1/dPr;
m=lift(lift((Mod(Mod(r,Pfactor),q^(i+1)))^g-vg))/q^i;
return(r-lift(lift(m*dPrinv))*q^i)
};

findroot(vg)={
racine=centerlift(lift(polrootsff(x^g-vg*Mod(1,q),q,Pfactor)[1]));
for(i=2,k,racine=henselstep(racine,i-1,vg));
return(racine)
};

lllfinal(racine)={
m=matrix(dim+1,dim+1,i,j,0);
for(i=1,zn,m[i,i]=q^k);
for(i=zn+1,dim,m[i,i]=-1);
m[dim+1,dim+1]=1;
for(j=zn+1,dim,polj=lift(lift(Mod(Mod(t^(j-1),Pfactor),q^k)));for(i=1,zn,m[i,j]=polcoeff(polj,i-1,t)));
for(i=1,zn,m[i,dim+1]=polcoeff(racine,i-1,t));
m=m*qflll(m);
rep=sum(i=1,dim,m[i,1]*t^(i-1));
return(rep)
};

reductionbis(idealrationnel)={
B=idealrationnel*qflll(idealrationnel);
cd=commondenom(B);

idrat_lll = idealrationnel * qflll( idealrationnel ) ;
\\deter=matdet(idealrationnel);  this might crash
deter=matdet(idrat_lll);
B=cd*B;
for(i=1,dim,y=accesscol(B,i);if(gcd(polresultant(y,PolC)/deter,p)==1&&gcd(polnorm(y),p)==1,mult=mathnf(idealmulelt(idealinv(NF,mathnf(B)),y));return([mult,y/cd])));
};

gentryszydlo(ideal,r)={
red=reductionbis(ideal);
idealr=red[1]*qflll(red[1]);
pow=powering(idealr,polnorm(red[2])/r,e);
vg=(pow[4]*centerlift(lift(inversemodp(lift(lift(pow[3]))))))^w;
racine=findroot(vg);
generateur=lllfinal(racine);
return(lift(Mod(red[2],PolC)/generateur))
};

idealconj(ideal)={
rep=matrix(dim,dim,i,j,0);
for(j=1,dim,poljzk=zkofpol(polconj(accesscol(ideal,j))%PolC);for(i=1,dim,rep[i,j]=poljzk[i,1]));
return(rep)
};

inter(M,N)={
K=qflll(concat(M,N),4)[1];
K=K*qflll(K);
rep=matrix(0,0);
for(j=1,matsize(K)[2],c=commondenomvect(K[,j]);rep=concat(rep,M*matrix(matsize(M)[2],1,i,k,c*K[i,j])));
return(rep)
};

decomp(base,M)={
rep=matrix(matsize(base)[2],matsize(M)[2]);
for(j=1,matsize(M)[2],cl=matker(concat(base,matrix(matsize(M)[1],1,a,b,M[a,j])));for(i=1,matsize(base)[2],rep[i,j]=cl[i,1]/cl[matsize(base)[2]+1,1]));
return(rep)
};

GentrySzydlo(ideal)={
redmain=reductionbis(ideal);
idealc=idealconj(ideal);
idealquotient=idealmultiply(idealc,redmain[1]);
gs=gentryszydlo(idealquotient,polnorm(redmain[2]));
genquotient=lift(Mod(gs,PolC)/redmain[2]);
Oplus=matrix(dim,dim,i,j,polcoeff((t^(j-1)+polconj(t^(j-1)))%PolC,i-1));
Oplus=mathnf(Oplus);
Oplus=inter(idealhnf(NF,2),Oplus)/2;
idealproduit=Mch^-1*idealmulelt(ideal,genquotient+1);
idealproduit=idealproduit*qflll(idealproduit);
intersection=inter(Oplus,idealproduit);
interdecomp=decomp(Oplus,intersection);
if(matdet(interdecomp)^2!=abs(matdet(idealproduit)),genquotient=genquotient*t;idealproduit=Mch^-1*idealmulelt(ideal,genquotient+1);idealproduit=idealproduit*qflll(idealproduit);intersection=inter(Oplus,idealproduit);interdecomp=decomp(Oplus,intersection));
return([intersection,genquotient])
};

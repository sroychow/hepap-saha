from iminuit import Minuit
import numpy as np
import math
from math import*
from matplotlib import pyplot as plt
import pandas as pd
import statistics
#=====================================================================================================================================
#required functions:
def f1(t): #fit func: a0_fit*exp(-b*x) + a1_fit*exp{-(x-mu_fit)**2/a_fit**2}, fit values obtained from fitting data with fit func.
    a0_fit = 7.072927906619347 
    b_fit = -1.1814512107411528 
    a1_fit = 82.63120365913976
    mu_fit = 125.3
    a_fit = 2
    return t**(a0_fit + b_fit*math.log(t)) + (a1_fit*math.e**(-(t - mu_fit)**2/a_fit**2))
def f2(t): #fit func: a0_fit*exp(-b*x), fit values obtained from fitting data with fit func.
    a0_fit = 7.0716143377141725 
    b_fit = -1.181045308790491
    return t**(a0_fit + b_fit*math.log(t))
#=======================================================================================================================================
#functions to create and store toy datasets:    
def num1(): #create toy data by introducing Poisson fluctuation in the bkgd fit model
    data = np.loadtxt('cms_mass_events_2012_bkg_w.txt')
    x = (data[:,0])
    r = len(x)
    y=[]
    for i in range(r):
        y.append(f2(x[i]))
    n = np.random.poisson(lam=(y), size=(34)) 
    print("num1",n)
    print("num1",x)
    np.savetxt('toy_data_bkg_local_f.txt',n)
    #print(n)
    return n
def num2(): #store the toy data
    n=[]
    dat = np.loadtxt('toy_data_bkg_local_f.txt')
    for i in dat:
        n.append(i)
    #print("num2",n)  
    return n
def num3(): #create toy data by introducing Poisson fluctuation in the data fit model
    data = np.loadtxt('cms_mass_events_2012_bkg_w.txt')
    x = (data[:,0])
    r = len(x)
    y=[]
    for i in range(r):
        y.append(f1(x[i]))
    n = np.random.poisson(lam=(y), size=(34))
    np.savetxt('toy_data_sig_local_f.txt',n)
    #print(n)
    return n
def num4(): #store the toy data
    n=[]
    dat = np.loadtxt('toy_data_sig_local_f.txt')
    for i in dat:
        n.append(i)
    #print(n)  
    return n         
#=====================================================================================================================================    
#neg-log-likelihood functions:
def fact(n):
    f=1
    for i in range(1,n+1):
        f = f*i
    return f    
    
def nll1(a0,b): #neg-log-likelihood func for toy data generated from bkgd only fit with and exp
    data = np.loadtxt('cms_mass_events_2012_bkg_w.txt')
    x = (data[:,0])
    r = len(x)
    n = num2() 
    f=0       
    #i>14 to i<=24 sig-reg
    for i in range(r): 
        f_i = x[i]**(a0 + b*math.log(x[i]))
        n_i = math.floor(n[i])
        if(f_i<=0):
           print("f_i",f_i)
        f = f + (f_i - n[i]*math.log(f_i) + n_i*math.log(n_i) - n_i)
    return f

def nll2(a0,b,a1): #neg-log-likelihood func for toy data generated from bkgd only fit with and exp+gauss
    data = np.loadtxt('cms_mass_events_2012_bkg_w.txt')
    x = (data[:,0])
    r = len(x)
    n = num2()    
    f=0     
    mu = 125.3
    a = 2
    #i>14 to i<=24 sig-reg
    for i in range(r): 
           n_i = math.floor(n[i])
           f_i = x[i]**(a0 + b*math.log(x[i])) + (a1*math.e**(-(x[i] - mu)**2/a**2))
           if(f_i<=0):
              print("f_i",f_i)
           f = f + (f_i - n[i]*math.log(f_i) + n_i*math.log(n_i) - n_i)
    return f

def nll5(a0,b): #neg-log-likelihood func for toy data generated from bkgd+sig fit with and exp
    data = np.loadtxt('cms_mass_events_2012_bkg_w.txt')
    x = (data[:,0])
    r = len(x)
    r1 = len(x)
    n = num4() 
    f=0       
    #i>14 to i<=24 sig-reg
    for i in range(r): 
        f_i = x[i]**(a0 + b*math.log(x[i]))
        n_i = math.floor(n[i])
        if(f_i<=0):
          print("f_i",f_i)
        f = f + (f_i - n[i]*math.log(f_i) + n_i*math.log(n_i) - n_i)
    return f

def nll6(a0,b,a1): #neg-log-likelihood func for toy data generated from bkgd+sig fit with and exp+gauss
    data = np.loadtxt('cms_mass_events_2012_bkg_w.txt')
    x = (data[:,0])
    r = len(x)
    n = num4()    
    f=0     
    mu = 125.3
    a = 2
    #i>14 to i<=24 sig-reg
    for i in range(r):
           n_i = math.floor(n[i])
           f_i = x[i]**(a0 + b*math.log(x[i])) + (a1*math.e**(-(x[i] - mu)**2/a**2))
           if(f_i<=0):
              print("f_i",f_i)
           f = f + (f_i - n[i]*math.log(f_i) + n_i*math.log(n_i) - n_i)
    return f

def nll3(n,a0,b): #calculates neg-log-likelihood and chisq for exp fit
    lchi=[]
    chisq=0
    data = np.loadtxt('cms_mass_events_2012_bkg_w.txt')
    x = (data[:,0])
    r = len(x) 
    f=0       
    #i>14 to i<=24 sig-reg
    #print(n)
    for i in range(r): 
           n_i = math.floor(n[i])
           f_i = x[i]**(a0 + b*math.log(x[i]))
           chisq_i = (f_i - n[i])**2/(f_i)
           chisq += chisq_i
           f = f + (f_i - n[i]*math.log(f_i) + n_i*math.log(n_i) - n_i)
    lchi.append(f)
    lchi.append(chisq)       
    return lchi

def nll4(n,a0,b,a1): #calculates neg-log-likelihood and chisq for exp+gauss fit
    lchi=[]
    chisq=0
    data = np.loadtxt('cms_mass_events_2012_bkg_w.txt')
    x = (data[:,0])
    r = len(x) 
    f=0     
    mu = 125.3
    a = 2
    #i>14 to i<=24 sig-reg
    for i in range(r): 
           n_i = math.floor(n[i])
           f_i = x[i]**(a0 + b*math.log(x[i])) + (a1*math.e**(-(x[i] - mu)**2/a**2))
           f = f + (f_i - n[i]*math.log(f_i) + n_i*math.log(n_i) - n_i)
           chisq_i = (f_i - n[i])**2/(f_i)
           chisq += chisq_i
    lchi.append(f)
    lchi.append(chisq)       
    return lchi
#===============================================================(main code)====================================================================    
llrb=[]
chi1=[]
chi2=[]
fit_a0_exp=[]
fit_b_exp=[]
neg_like_exp=[]
fit_a0_gxp=[]
fit_b_gxp=[]
fit_a1=[]
neg_like_gxp=[]
nll_diff=[]
a1_neg=[]
like_ratio=[]
k_exp=0
ka_exp=0
k_gxp=0
ka_gxp=0
neg_a1=0
pos_a1=0
neg_nll=0
k=0
for i in range(10000000):
    gxp_valid = False
    gxp_acc = False
    #print('=====================================================')
    n=num3()  #put num1() for bkgd, num3() for full data
    m=Minuit(nll5,a0=2,b=0) #use nll1 for bkgd, nll5 for full data
    #m.limits["a0"] = (0,100000)
    #m.limits["b"] = (0,1)
    m.migrad()
    m.hesse()
    print('exp-valid-acc',m.valid,m.accurate)
    if(m.valid):
      k_exp=k_exp+1    
    if(m.accurate):
       ka_exp=ka_exp+1
    a0_fit_exp = m.values["a0"]
    b_fit_exp = m.values["b"]
    print("exp-fit",a0_fit_exp,b_fit_exp)
    
    
    m=Minuit(nll6,a0=2,b=0,a1=70)  #use nll2 for bkgd, nll6 for full data
    #m.limits["a0"] = (0,100000)
    m.limits["a1"] = (0,1000)
    #m.limits["b"] = (0,1)
    m.migrad()
    m.hesse()
    print('gxp-valid-acc',m.valid,m.accurate)
    if(m.valid):
      k_gxp=k_gxp+1   
      gxp_valid = True 
    if(m.accurate):
       ka_gxp=ka_gxp+1
       gxp_acc = True
    a0_fit_gxp = m.values["a0"]
    b_fit_gxp = m.values["b"]
    a1_fit_gxp = m.values["a1"]
    print("exp-gauss-fit",a0_fit_gxp,b_fit_gxp,a1_fit_gxp)  
    
    if (gxp_valid*gxp_acc):
       k=k+1
       
       lchi1 = nll3(n,a0_fit_exp,b_fit_exp)
       nl1 = lchi1[0]
       chi1.append(lchi1[1]) 
       fit_a0_exp.append(a0_fit_exp)
       fit_b_exp.append(b_fit_exp)
       neg_like_exp.append(nl1)
       
       lchi2 = nll4(n,a0_fit_gxp,b_fit_gxp,a1_fit_gxp)
       nl2 = lchi2[0]
       chi2.append(lchi2[1]) 
       fit_a0_gxp.append(a0_fit_gxp)
       fit_b_gxp.append(b_fit_gxp)
       fit_a1.append(a1_fit_gxp)
       neg_like_gxp.append(nl2)
       nll_diff.append(2*(nl1-nl2))
       
       llrb.append(2*(nl1-nl2))
       
       if ((a1_fit_gxp)<0.001):
          neg_a1 = neg_a1+1
       if ((nl1-nl2)<0):
          neg_nll = neg_nll+1
          print(nl1-nl2,math.e**(nl2-nl1),a1_fit_gxp)
          like_ratio.append(math.e**(nl2-nl1))
          a1_neg.append(a1_fit_gxp)
       
       #print("iteration ", i, "neg log-like exp ", nl1, "  neg log-like exp+gauss ", nl2, "  neg log ratio, exp/exp+gauss ", llrb[k-1])
       
    if (k>=1000):
       break   
   
print('nll-diff-neg-pos',neg_a1,neg_nll)
data = {'a0_exp': fit_a0_exp,'b_exp': fit_b_exp,'nll_exp': neg_like_exp, 'a0_gxp': fit_a0_gxp, 'b_gxp': fit_b_gxp, 'a1_gxp': fit_a1,
        'nll_gxp': neg_like_gxp, 'nll_diff': nll_diff}
dat = {'likelihood ratio': like_ratio, 'a1_gxp': a1_neg}        
df=pd.DataFrame(data)
df1=pd.DataFrame(dat)
#print(df) 
print('significance',sqrt(statistics.mean(nll_diff)))
#print("bkgd: valid acc exp ",k_exp,ka_exp, " bkgd: valid acc gxp ", k_gxp,ka_gxp)
df.to_csv('fit_val_cms_2012_calib_f2.csv') 
df1.to_csv('neg-nll-like-a1_f2.csv')   
np.savetxt("nll-ratio-10K-stir-full-cms_2012_calib_f2.txt",llrb)
np.savetxt("chisq-exp-10K-stir-full-cms_2012_calib_f2.txt",chi1)
np.savetxt("chisq-gxp-10K-stir-full-cms_2012_calib_f2.txt",chi2)

'''print("===============================================full===================================================")    
llrsb=[]
kf_exp=0
kaf_exp=0
kf_gxp=0
kaf_gxp=0
for i in range(10000):
    n=num3()
    m=Minuit(nll5,a0=10000,b=0.03)
    m.limits["a0"] = (0,5000000)
    m.limits["b"] = (0,1)
    m.migrad()
    m.hesse()
    if(Minuit.valid):
      kf_exp=kf_exp+1    
    if(Minuit.accurate):
       kaf_exp=kaf_exp+1
    a0_fit = m.values["a0"]
    b_fit = m.values["b"]
    llrs1 = nll3(n,a0_fit,b_fit)

    m=Minuit(nll6,a0=10000,b=0.03,a1=10)
    m.limits["a0"] = (0,50000000)
    m.limits["a1"] = (0,10000)
    m.limits["b"] = (0,1)
    m.migrad()
    m.hesse()
    if(Minuit.valid):
      kf_gxp=kf_gxp+1    
    if(Minuit.accurate):
       kaf_gxp=kaf_gxp+1
    a0_fit = m.values["a0"]
    b_fit = m.values["b"]
    a1_fit = m.values["a1"]
    llrs2 = nll4(n,a0_fit,b_fit,a1_fit)
    
    llrsb.append(llrs1/llrs2)
    print("iteration ", i, "neg log-like exp ", llrs1, "  neg log-like exp+gauss ", llrs2, "  neg log ratio, exp/exp+gauss ",llrsb[i])  
np.savetxt("nll-ratio-full-data.txt",llrsb)  
#print("full: valid acc exp ",kf_exp,kaf_exp, " full: valid acc gxp ", kf_gxp,kaf_gxp) ''' 
#======================================================================================================================================                

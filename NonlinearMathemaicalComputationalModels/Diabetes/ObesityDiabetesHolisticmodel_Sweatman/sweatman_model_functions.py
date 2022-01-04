import numpy as np

def f(x):
    n = 3.24
    IS = 0.5 * ( (n +1 )/(n+ np.power(x,2)) ) + 0.5
    return (IS)

def fASi(IF,varval,varval0,pfac):
    ASi_val = IF*np.power((f(varval/(2*varval0))),pfac) + (1-IF)
    return ASi_val

def fXLIPIDSi(IF,varval,varval0,pfac): #(lipid content)
    fval = IF*np.power((f(varval/(varval0))),pfac) + (1-IF)
    return fval

def g(x):
    g=((7/3)/((7/3)+np.power(x,2)))+0.3
    return g

def yCHO(CHOin,CHOin0):
     return(CHOin/CHOin0)

def gSi(CHOin,CHOin0):
    yCHO_val=yCHO(CHOin,CHOin0)
    return( g(yCHO_val) )

def fSL(IF,VLDLTG,VLDLTG0,F,F0,p5,p6):
    fval = IF*np.power((f(VLDLTG/(VLDLTG0))),p5)*np.power((f(F/(F0))),p6) + (1-IF)
    return fval

def fHGPI( hsi_val,I,I0 ):
    return( (7/4)*( (4/3)*np.power(I0,2)/( ((4/3)*np.power(I0,2)) + np.power(hsi_val*I,2)   ) ) )  
                   
def fHGPGg(gg,gg0):
    return( ( np.power(gg,2)/(np.power(gg,2)+np.power(gg0,2)) ) + 0.5 )
                   
def fLep_plus(L,L0,fSL_val,SL): 
    return( np.power(SL*fSL_val*L,2)/( np.power(SL*fSL_val*L,2) + np.power(L0,2) ))

def fLep_minus(L,L0,fSL_val,SL): 
    return( np.power(L0,2)/( np.power(SL*fSL_val*L,2) + np.power(L0,2) ))

def fHGPL(flep_minus): 
    return( flep_minus + 0.5)
                   
def HGP(HGP0,fHGPI_val,fHGPGg_val,fHGPL_val):
    return(HGP0*fHGPI_val*fHGPGg_val*fHGPL_val)

def fGlu_eff(EG0,fsL_val,SL,L,L0,G):
    return( EG0 * ((np.power(fsL_val*SL*L,2)/(np.power(fsL_val*SL*L,2) +(3/7)* np.power(L0,2)) + 0.3)*G))

def fmuscle_glu_intake(PSI,PSI_val,fLep_plus_val,I,G):
    return(0.7 * PSI * PSI_val *(fLep_plus_val+0.5)*I*G)

def fadipose_glu_intake(ASI,ASI_val,fLep_minus_val,SI0,F,F0,I,G):
    return(0.3 * ASI * ASI_val *(fLep_minus_val+0.5)*SI0*(F/F0)*I*G)

def ffast(CHOin,CHOin0,aCHO):
    xCHO = 1 + (CHOin-CHOin0)/(aCHO*CHOin0)
    return (2/(1 + np.power(xCHO,2)))
def ffed(vfast):
    return(2-vfast)


def fVLDLTGI(HSI,fHSI,I,I0):
    return(0.33*(2*np.power(I0,2)/( np.power(HSI*fHSI*I,2) + np.power(I0,2) ))+0.67)

def fNEFApI(fASI_val,ASI,I,I0):
    fNEFApI_val = (np.power(I0,2) / ( np.power(fASI_val*ASI*I,2) + np.power(I0,2))) + 0.5
    return(fNEFApI_val)

def fsigma(sigma_r,flep_minus_val,G,alpha_gi):
    sigma_val = sigma_r * (flep_minus_val+0.5)*np.power(G,2)/( np.power(alpha_gi,2) + np.power(G,2) )
    return(sigma_val)

def dG_dt(vfast,vHGP , KGP ,U0 , vGlu_eff , vmuscle_glu_intake , vadipose_glu_intake):
    dG_dt=vfast*(vHGP + KGP -U0 - vGlu_eff - vmuscle_glu_intake - vadipose_glu_intake)
    return(dG_dt)

def dI_dt(ffast,beta, sigma,LBM,Vplbm,lambdaI, I,Iin):
    return(ffast*( ( beta* sigma/(LBM*Vplbm) )-lambdaI*I)+Iin)

def dGgp_dt(vfast,sigma_Gg, flep_minus_val, ACSI,fACSI_val, Ien, I0, lambda_Gg,Gg,Ggin):
    production =vfast*2* sigma_Gg *(np.power(I0,2)/( np.power(ACSI*fACSI_val*Ien,2) + np.power(I0,2) ))*(0.5+flep_minus_val)
    decay = vfast*lambda_Gg*Gg
    dGgp_Val = production - decay+Ggin
    return(dGgp_Val)
def dNEFAp_dt(vfast,rLI, flep_plus_val, fNEFApI_val, F, F0, kNEFApfast, NEFApfast):
    production = vfast*rLI*(0.5+flep_plus_val)*fNEFApI_val*(1.6*(np.power(F0,2)/( np.power(F,2) + np.power(F0,2) ))+0.2)
    decay = vfast*kNEFApfast*NEFApfast
    dNEFAp_Val = production - decay
    return(dNEFAp_Val)

def fVLDLTGI(HSI,vHSI,I,I0):
    return(0.33*(2*np.power(I0,2)/( np.power(HSI*vHSI*I,2) + np.power(I0,2) ))+0.67)
def dVLDLTGpfast_dt(vfast,rVLDLTGpfast,VLDLTGI_val,vlep_plus_val,HLipid,HLipid0,kVLDLTGpfast,kCHO,VLDLTGpfast):
    return(vfast*(rVLDLTGpfast*VLDLTGI_val*(1.5-vlep_plus_val)*(HLipid/HLipid0)) - (kVLDLTGpfast*kCHO*VLDLTGpfast))

def fLI(KL,fASI_val,ASI,I,I0):
    vLI = 1.0 - (KL/2)+(KL*np.power(fASI_val*ASI*I,2) / ( np.power(fASI_val*ASI*I,2) + np.power(I0,2))) 
    return(vLI)

def dL_dt(ffast,rL,vLI,F,F0,L,hL,leptin_in):
    return(ffast*(rL*(F/F0)*vLI-(np.log(2)*L/hL)) + leptin_in)

def fADNL(rADNL,flep_minus_val,ycho_val,ASI,fASI_val,I):
    adnl_val = rADNL* (flep_minus_val+0.5)*(2*np.power(ycho_val,2)/(1+np.power(ycho_val,2)))*(ASI*fASI_val*I)
    return (adnl_val)

def ffatplus(TGin,XCMTG,ffast,kVLDLTGpfast,VLDLTGpfast,ffed,kVLDLTGpfed,adnl_val,kCHO):
    dVLDLTGpfast = 0.5*ffast*kVLDLTGpfast*kCHO*VLDLTGpfast
    dVLDLTGpfed = 0.5*ffed*kVLDLTGpfed*kCHO*1.2*VLDLTGpfast
    fatplus_val = 0.45*TGin + XCMTG +dVLDLTGpfast +dVLDLTGpfed + ffed*adnl_val 
    return(fatplus_val)
def ffatminus(rLI,ffast,flep_plus_val,fASI_val,ASI,I,I0,F,F0,rLox,ATGox,ffed):
    fNEFApI_val = (np.power(I0,2) / ( np.power(fASI_val*ASI*I,2) + np.power(I0,2))) + 0.5
    lipolysis_val = 0.9 * rLI * ffast*(flep_plus_val+0.5)*fNEFApI_val*(( 1.6*np.power(F,2)/(np.power(F,2) + np.power(F0,2)) ) + 0.2)
    TGoxi_val = rLox * ATGox * (flep_plus_val+0.5)*(F/F0)*((ffed*0.92/2.15)+(ffast*1.23/2.15))
    fatminu_val = lipolysis_val + TGoxi_val
    return(fatminu_val)

def dF_dt(c,vfatplus,vfatminus,LBM):
    return(c*(vfatplus-vfatminus)*LBM)
#!/usr/bin/python3

'''Functions used by meancontour and simultip

Used as a standalone script, draw some charts
'''

'''
© Bernard Billoud / Morphogenesis of Macro-Algae team (CNRS/Sorbonne Université)

Bernard.Billoud@sb-roscoff.fr

This software is a computer program whose purpose is to provide
useful functions to meancontour.py and simultip.py. Used as a
standalone script, it draws some charts.

This software is governed by the CeCILL-C license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/or redistribute the software under the terms of the CeCILL-C
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

The fact that you are presently reading this means that you have had
knowledge of the CeCILL-C license and that you accept its terms.
'''

import os
import sys
import time
import numpy as np
import argparse
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
import datetime
from scipy import interpolate

rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex = True)
params = {'text.latex.preamble': [
    r'\usepackage{siunitx}', 
    r'\sisetup{detect-family = true}',
    r'\usepackage{textcomp}',
    r'\usepackage{textgreek}',
    r'\usepackage{sansmath}',
    r'\usepackage{subscript}',
    r'\usepackage{amsmath}']}
plt.rcParams.update(params)   

def midpoint(pxA, pxB) :
    """Midpoint between two points, passed as tuples.

    .. note:: Use lowest number of dimensions.
    """
    mp = []
    for i in range(min(len(pxA), len(pxB))):
        mp.append((pxA[i]+pxB[i])/2)
    return tuple(mp)

def eucdist(pxA, pxB) :
    """Euclidian distance between two points, passed as tuples.

    .. note:: Use lowest number of dimensions.
    """
    d2 = 0.0
    for i in range(min(len(pxA), len(pxB))):
        d2 += (pxA[i]-pxB[i])**2
    return np.sqrt(d2)

def angle(pxA, pxO, pxB) :
    """Angle pXA-pxO-pxB, each defined as a 2-d tuple.

    Uses the dot-product OA·OB = OA×OB×cos(AOB)
    and the sign of the vector product OA⋀OB

    .. note:: Works only for points in a plane.
    """
    vecOA = (pxA[0]-pxO[0], pxA[1]-pxO[1])
    vecOB = (pxB[0]-pxO[0], pxB[1]-pxO[1])
    dotpr = vecOA[0]*vecOB[0]+vecOA[1]*vecOB[1]
    p = eucdist(pxO, pxA)*eucdist(pxO, pxB)
    if p == 0:
        print('WARNING: no angle for', pxA, pxO, pxB)
        a = 0
    else:
        cosi = dotpr/p
        if cosi > 1.0 :  # This should never happen!
            cosi = 1.0
        if cosi < -1.0 : # This should never happen!
            cosi = -1.0
        a = np.arccos(cosi) # gives the positive angle
        if vecOA[0]*vecOB[1]-vecOA[1]*vecOB[0] < 0 :
            a = -a
    return a

def curvature(pxA, pxO, pxB) :
    """Curvature at point pxO, based on neighbor points pxA and pxB."""
    return 2*np.sin(angle(pxA, pxO, pxB))/eucdist(pxA, pxB)

def s2hhmmss(ts):
    """Format time in seconds into a HHhMMmSSs string."""
    ts = int(ts)
    th = int(ts/3600)
    ts = ts%3600
    tm = int(ts/60)
    ts = ts%60
    hhmmss='{0:02d}h{1:02d}m{2:02d}s'.format(th,tm,ts)
    return hhmmss

def modelen(Ks,Kt,cwt,tur,Phi_ext,Sy):
    """Normal strain rate εn

    Computed according to the viscoplastic model
    under the assumptions of thin wall, tranverse isotropy
    and orthogonal growth
    """
    Ss = tur/(2*cwt*Kt)
    St = Ss*(2-Ks/Kt)
    nu = (1-Ks/Kt)/2
    Se2 = nu*(St-Ss)**2+(1-nu)*(St**2+Ss**2)
    if Se2 >= 0:
        Se = np.sqrt(Se2)
    else:
        Se = 0
    if Se>Sy:
        K = np.sqrt(2*((nu**2-nu+1)*(Ss**2+St**2)+(nu**2-4*nu+1)*Ss*St))
        eps_n = Phi_ext*(Se-Sy)*(nu-1)*(Ss+St)/K
    else:
        eps_n = 0
    # print('T =',tur,'κs =',Ks,'κθ =',Kt,'δ =',cwt,'σs =',Ss,'σθ =',St,
    #       'ν =',nu,'σe² =',Se2,'σe =',Se,'σy =',Sy,'εn =',eps_n)
    return eps_n

def modelvn(Ks,Kt,cwt,tur,Phi_ext,Sy):
    """Normal velocity Vn

    Computed according to the viscoplastic model
    under the assumptions of thin wall, tranverse isotropy
    and orthogonal growth
    """
    Ss = tur/(2*cwt*Kt)
    St = Ss*(2-Ks/Kt)
    nu = (1-Ks/Kt)/2
    Se2 = nu*(St-Ss)**2+(1-nu)*(St**2+Ss**2)
    if Se2 >= 0:
        Se = np.sqrt(Se2)
    else:
        Se = 0
    if Se>Sy:
        K = np.sqrt(2*((nu**2-nu+1)*(Ss**2+St**2)+(nu**2-4*nu+1)*Ss*St))
        eps_t = Phi_ext*(Se-Sy)*(St-nu*Ss)/K
        vn = eps_t/Kt
    else:
        vn = 0
    # print('T =',tur,'κs =',Ks,'κθ =',Kt,'δ =',cwt,'σs =',Ss,'σθ =',St,
    #       'ν =',nu,'σe² =',Se2,'σe =',Se,'σy =',Sy,'Vn =',vn)
    return vn

def findR0(tckspl, start, splstp):
    """Find t so that interpolated R is close to 0."""
    tlo, thi = start-splstp, start+splstp
    tmi = start
    [cRl, cRh] = interpolate.splev([tlo, thi], tckspl)[0]
    ntry = 0
    while cRh<0 and ntry<1000:
        thi += splstp
        cRh = interpolate.splev(thi, tckspl)[0]
        ntry += 1
    if ntry<1000:
        ntry = 0
        while cRl>0 and ntry<1000:
            tlo -= splstp
            cRl = interpolate.splev(tlo, tckspl)[0]
            ntry += 1
    if ntry<1000:
        tmi = ((tlo*cRh)-(thi*cRl))/(cRh-cRl)
        cRm = interpolate.splev(tmi, tckspl)[0]
        ntry = 0
        while abs(cRm)>1E-9 and ntry<1000:
            if cRm<0:
                tlo = tmi
                cRl = cRm
            if cRm>0:
                thi = tmi
                cRh = cRm
            tmi = ((tlo*cRh)-(thi*cRl))/(cRh-cRl)
            cRm = interpolate.splev(tmi, tckspl)[0]
            ntry += 1
    return tmi

def read_data_file(paradic,instr):
    """Read input file and return data as a dictionary.
    
    File is expected to contain (optional) parameters
        followed by (mandatory) data for a contour
    Data: expected fields are (in any order)
        Set R X s Phi Kappa_s Kappa_t Nu
        Other fields are ignored
    """

    datdic = {}   # Dictionary of data (will be returned)
    
    point = False # are we reading data for a point?
    
    for ln in instr : # read initial input stream line by line
    
        # parse input line
        ln = ln[:-1] # remove '\n'
        fields = ln.split(sep = '\t')
        if fields[0] == 'End':
            point = False
        else:
            if not point:
                names = fields
                if names[0] == 'Set': # this is the header line
                    for n in names:
                        datdic[n] = []
                    point = True # next lines are data for one point
        
                else: # this is a parameter (value or function or file)
                    paradic[names[0]] = []
                    for n in names[1:]:
                        try: # try to convert into a float
                            fv = float(n)
                            paradic[names[0]].append(fv)
                        except ValueError: # or keep as / convert to string
                            paradic[names[0]].append(str(n))
            else:
                data = fields
                i = 0
                for n in names:
                    try: # try to convert into a float
                        fv = float(data[i])
                        datdic[n].append(fv)
                    except ValueError: # or keep as / convert to string
                        datdic[n].append(str(data[i]))
                    i += 1
    return datdic

def constant(para,s):
    return para[0]

def linear(para,s):
    [minv, maxv, smin, smax] = para
    if s <= smin:
        v = minv
    elif s >= smax:
        v = maxv
    else:
        v = minv+(maxv-minv)*(s-smin)/(smax-smin)
    return v

def sigmoid(para,s):
    [minv, maxv, half, scal] = para
    return maxv - (maxv - minv) / (1 + np.exp((abs(s)-half)/scal))

def gaussian(para,s):
    [minv, maxv, half] = para
    return maxv - (maxv - minv) * np.exp(-s * s / (half * half / np.log(2)))

def lockhart(para,s):
    [minv, maxv, half] = para
    return maxv - (maxv - minv) * (1 / (1 + (s / half)**2))

def pearson(para,s):
    [minv, maxv, half, m] = para
    return maxv-(maxv-minv)*((1+(s/(half/np.sqrt((1/2)**(-1/m)-1)))**2)**(-m))

def file(para,s):
    [slst,vlst] = para[0:2]
    abss = abs(s)
    if abss < slst[0]:
        val = vlst[0]
    else:
        npt=len(slst)
        i = 1
        while i < npt and slst[i] < abss:
            i += 1
        if i < npt:
            val = vlst[i-1] + (vlst[i] - vlst[i-1]) * (abss - slst[i-1]) / (slst[i] - slst[i-1])
        else:
            val = vlst[-1]
    return val

def para_func(fp, s):
    """Return value of function fp at s."""
    func = fp[0]
    return globals()[func.lower()](fp[1:],s)

def wall_thick(fp, s):
    '''
    Cell wall thickness at meridional abscissa s
    '''
    return para_func(fp,s)

def sigma_y(fp, s):
    '''
    Stress σy at meridional abscissa s
    '''
    return para_func(fp,s)

def phi_ext(fp, s):
    '''
    Extensibility Φ at meridional abscissa s
    '''
    return para_func(fp,s)

def tipadvance(Ks0,Kt0,turgor,thickpara,phipara,sypara):
    """Compute ΔX for tip point (s=0)."""
    Ks = Ks0[0]
    Kt = Kt0[0]
    cwt = wall_thick(thickpara,0)
    Ss = turgor/(2*cwt*Kt)
    St = Ss*(2-Ks/Kt)
    nu = (1-Ks/Kt)/2
    Se = np.sqrt(nu*(St-Ss)**2+(1-nu)*(St**2+Ss**2))
    K = np.sqrt(2*((nu**2-nu+1)*(Ss**2+St**2)+(nu**2-4*nu+1)*Ss*St))
    Phi_ext=phi_ext(phipara,0)
    Sy=sigma_y(sypara,0)
    if Se>Sy:
        eps_t = Phi_ext*(Se-Sy)*(St-nu*Ss)/K
        vn = eps_t/Kt
    else:
        vn = 0
    Deltax = vn # actually *np.cos(φ) but at tip, φ=0

    return Deltax

def set_data_lists(s0,X0,R0,Ph0,Ks0,Kt0,turgor,thickpara,phipara,sypara,expgro,sbs):
    """Prepare data lists suitable for charts."""
    sl  = [] # meridional curvilinear abscissa s (µm)
    xl  = [] # axial abscissa x (µm)
    rl  = [] # radial abscissa r (µm)
    Phl = [] # Angle between normal and axis φ (rad)
    Ksl = [] # Meridional curvature κs (µm⁻¹)
    Ktl = [] # Circumferential curvature κθ (µm⁻¹)
    Lal = [] # Curvature anisotropy λ (no unit)
    Wtl = [] # Cell Wall thickness δ (µm)
    Ssl = [] # Meridional stress σs (MPa)
    Stl = [] # Circumferential stress σθ (MPa)
    Gal = [] # stress anisotropy γ (no unit)
    Nul = [] # Flow coupling ν (no unit)
    Sel = [] # Global stress σe (MPa)
    Knl = [] # K = √(2((ν²-ν+1)(σs²+σθ²)+(ν²-4ν+1)σs σθ)) (MPa)
    Fil = [] # Extensibility Φ (min⁻¹·MPa⁻¹)
    Etl = [] # Circumferential strain rate εθ (µm·min⁻¹·µm⁻¹)
    Esl = [] # Meridional strain rate εs (µm·min⁻¹·µm⁻¹)
    Enl = [] # Normal strain rate εn (µm·min⁻¹·µm⁻¹)
    Wsl = [] # Cell wall thining compensation (µm·µm⁻¹)
    Lel = [] # strain rate anisotropy λ (no unit)
    Vnl = [] # Vn (µm·min⁻¹)
    Vxl = [] # Vx (µm·min⁻¹)
    Vel = [] # Expected Vn at each point, computed using tip (µm·min⁻¹)
    Val = [] # Expected Vn at each point, computed using dome (µm·min⁻¹)
    Eel = [] # Expected strain rate ε* computed as K κθ Vn /(σθ - ν σs) (min⁻¹)
    Syl = [] # Yield theshold σy (MPa)
    for i in range(len(s0)):
        s = s0[i]
        sl.append(s)
        x = max(X0)-X0[i]
        xl.append(x)
        r = R0[i]
        rl.append(r)
        Phi = Ph0[i]
        Phl.append(Phi)
        Ks = Ks0[i]
        Ksl.append(Ks)
        Kt = Kt0[i]
        Ktl.append(Kt)
        La = (Kt-Ks)/(Kt+Ks)
        Lal.append(La)
        cwt = wall_thick(thickpara,s)
        Wtl.append(cwt)
        Ss = turgor/(2*cwt*Kt)
        Ssl.append(Ss)
        St = Ss*(2-Ks/Kt)
        Stl.append(St)
        Ga = (St-Ss)/(St+Ss)
        Gal.append(Ga)
        nu = (1-Ks/Kt)/2
        Nul.append(nu)
        Se2 = nu*(St-Ss)**2+(1-nu)*(St**2+Ss**2)
        if Se2 >= 0:
            Se = np.sqrt(Se2)
        else:
            print('WARNING: s =',s,'σs =',Ss,'σθ =',St,
                       'ν =',nu,'σe² =',Se2,'-> σe = 0')
            Se = 0
        Sel.append(Se)
        K = np.sqrt(2*((nu**2-nu+1)*(Ss**2+St**2)+(nu**2-4*nu+1)*Ss*St))
        Knl.append(K)
        Phi_ext=phi_ext(phipara,s)
        Fil.append(Phi_ext)
        Sy=sigma_y(sypara,s)
        Syl.append(Sy)
        if Se>Sy:
            fip = Phi_ext*(Se-Sy)
            eps_t = fip*(St-nu*Ss)/K
            eps_s = fip*(Ss-nu*St)/K
            eps_n = fip*(nu-1)*(Ss+St)/K
            if eps_t==0 and eps_s==0:
                Le = 1
            else:
                Le = (eps_t - eps_s)/(eps_t + eps_s)
            vn = eps_t/Kt
        else:
            fip = 0
            eps_t = 0
            eps_s = 0
            eps_n = 0
            Le = None
            vn = 0
        cws = -eps_n * cwt
        Esl.append(eps_s)
        Etl.append(eps_t)
        Enl.append(eps_n)
        Wsl.append(cws)
        Lel.append(Le)
        vx = vn*np.cos(Ph0[i])
        Vnl.append(vn)
        Vxl.append(vx)
        if i>0:
            sl.insert(0, -s)
            xl.insert(0, x)
            rl.insert(0, -r)
            Phl.insert(0, -Phi)
            Ksl.insert(0, Ks)
            Ktl.insert(0, Kt)
            Lal.insert(0, La)
            Lel.insert(0, Le)
            Wtl.insert(0, cwt)
            Ssl.insert(0, Ss)
            Stl.insert(0, St)
            Gal.insert(0, Ga)
            Nul.insert(0, nu)
            Sel.insert(0, Se)
            Knl.insert(0, K)
            Fil.insert(0, Phi_ext)
            Syl.insert(0, Sy)
            Esl.insert(0, eps_s)
            Etl.insert(0, eps_t)
            Enl.insert(0, eps_n)
            Wsl.insert(0, cws)
            Vnl.insert(0, vn)
            Vxl.insert(0, vx)

    # compute ∆X for tip point
    npt0 = len(s0)
    tip = int((len(sl)-1)/2)
    deltaX = np.mean(Vxl[tip-1:tip+2])*sbs/60 # Vxl in µm·min⁻¹; ∆X in µm·step⁻¹
    dR = [0]
    dX = [xl[tip]]
    i = 1
    while i < npt0:
        dR.append(R0[i])
        dR.insert(0,R0[i])
        dX.append(X0[i]+deltaX)
        dX.insert(0,X0[i]+deltaX)
        i += 1
    nptd = len(dR)
    tckspl, uc = interpolate.splprep([dR, dX], k = 3, s = 0)

    # interpolate
    splstp = 1/(2*npt0)    # approx. spline step
    t0 = findR0(tckspl, 0.5, splstp)
    splstp = (1-t0)/(npt0) # correct spline step
    trange = np.append(np.arange(t0, 1, splstp), [1])   # build t range
    splpts = interpolate.splev(trange, tckspl)          # interpolate
    npts = len(splpts[0])

    # list resampled and rescaled points
    sR = []
    sX = []
    for i in range(npts):
        sR.append(splpts[0][i])
        sX.append(splpts[1][i])
    sR[0]=0

    # compute Va
    Val=[deltaX*60/sbs] # adjusted Vn at each point
    ci = 1
    while ci < npt0:
        fwdpt = (R0[ci],X0[ci]+1.0)
        ctrpt = (R0[ci],X0[ci])
        ctrph = Ph0[ci]
        if -np.pi/2+1E-6<ctrph<np.pi/2-1E-6:
            sj = 1
            Va = 0
            while sj<npts and angle(fwdpt,ctrpt,(sR[sj],sX[sj])) > ctrph:
                sj += 1
            if 0<sj<npts:
                alo=angle(fwdpt,ctrpt,(sR[sj],sX[sj]))
                ahi=angle(fwdpt,ctrpt,(sR[sj-1],sX[sj-1]))
                tlo=trange[sj]
                thi=trange[sj-1]
                while abs(tlo-thi) > 1E-12:
                    tmi = (tlo+thi)/2
                    splpt = interpolate.splev(tmi, tckspl)
                    ami = angle(fwdpt,ctrpt,(splpt[0],splpt[1]))
                    if ami > ctrph:
                        thi = tmi
                        ahi = ami
                    else:
                        tlo = tmi
                        alo = ami
                tmi = (tlo+thi)/2
                splpt = interpolate.splev(tmi, tckspl)
                ami = angle(fwdpt,ctrpt,(splpt[0],splpt[1]))
                Va = eucdist(ctrpt,(splpt[0],splpt[1]))
        else:
            Va = 0
        Val.append(Va*60/sbs)
        Val.insert(0,Va*60/sbs)
        ci+=1

    xxl=[]

    if expgro>0:
            
        # compute ∆X for tip point
        npt0=len(s0)
        tip=int((len(sl)-1)/2)
        deltaX=expgro/3600*sbs # expgro in µm·h⁻¹; ∆X in µm·step⁻¹
        dR = [0]
        dX = [xl[tip]]
        i = 1
        while i < npt0:
            dR.append(R0[i])
            dR.insert(0,R0[i])
            dX.append(X0[i]+deltaX)
            dX.insert(0,X0[i]+deltaX)
            i += 1
        nptd = len(dR)
        tckspl, uc = interpolate.splprep([dR, dX], k = 3, s = 0)
    
        # interpolate
        splstp = 1/(2*npt0)    # approx. spline step
        t0 = findR0(tckspl, 0.5, splstp)
        splstp = (1-t0)/(npt0) # correct spline step
        trange = np.append(np.arange(t0, 1, splstp), [1])   # build t range
        splpts = interpolate.splev(trange, tckspl)          # interpolate
        npts = len(splpts[0])
    
        # list resampled and rescaled points
        sR = []
        sX = []
        for i in range(npts):
            sR.append(splpts[0][i])
            sX.append(splpts[1][i])
        sR[0]=0
        
        # compute expected Vn at each point
        Vel=[deltaX*60/sbs] # expected Vn at each point (µm·min⁻¹)
        ci = 1
        while ci < npt0:
            fwdpt = (R0[ci],X0[ci]+1.0)
            ctrpt = (R0[ci],X0[ci])
            ctrph = Ph0[ci]
            if -np.pi/2+1E-6<ctrph<np.pi/2-1E-6:
                sj = 1
                while sj<npts and angle(fwdpt,ctrpt,(sR[sj],sX[sj])) > ctrph:
                    sj += 1
                if 0<sj<npts:
                    alo=angle(fwdpt,ctrpt,(sR[sj],sX[sj]))
                    ahi=angle(fwdpt,ctrpt,(sR[sj-1],sX[sj-1]))
                    tlo=trange[sj]
                    thi=trange[sj-1]
                    while abs(tlo-thi) > 1E-12:
                        tmi = (tlo+thi)/2
                        splpt = interpolate.splev(tmi, tckspl)
                        ami = angle(fwdpt,ctrpt,(splpt[0],splpt[1]))
                        if ami > ctrph:
                            thi = tmi
                            ahi = ami
                        else:
                            tlo = tmi
                            alo = ami
                    tmi = (tlo+thi)/2
                    splpt = interpolate.splev(tmi, tckspl)
                    ami = angle(fwdpt,ctrpt,(splpt[0],splpt[1]))
                    Ve = eucdist(ctrpt,(splpt[0],splpt[1]))
            else:
                Ve = 0
            Vel.append(Ve*60/sbs)
            Vel.insert(0,Ve*60/sbs)
            ci+=1

    for i in range(len(sl)):
        Eel.append((Knl[i]*Ktl[i]*Val[i])/(Stl[i]-Nul[i]*Ssl[i]))

    return sl,xl,rl,Phl,Ksl,Ktl,Lal,Wtl,Ssl,Stl,Gal,Nul,Sel,Knl,Etl,Esl,Enl,Wsl,Lel,Vnl,Vxl,Val,Vel,Fil,Syl,Eel

def draw_charts(s0,X0,R0,Ph0,Ks0,Kt0,turgor,thick,Phi,Sy,eg,sbs):
    sl,xl,rl,Phl,Ksl,Ktl,Lal,Wtl,Ssl,Stl,Gal,Nul,Sel,Knl,Etl,Esl,Enl,Wsl,Lel,Vnl,Vxl,Val,Vel,Fil,Syl,Eel = set_data_lists(s0,X0,R0,Ph0,Ks0,Kt0,turgor,thick,Phi,Sy,eg,sbs)

    fout = open('Inistat_out.tab','w')
    fout.write('s\tx\tr\tphi\tKs\tKt\tcwt\tSs\tSt\tSe\tNu\tK\tPhi\tSy\tEs\tEt\tEn\tVn\tVe\tVa\tLambda\tGamma\tWs\tPDS\n')
    for i in range(len(sl)):
        fout.write(str(sl[i])+'\t'+str(xl[i])+'\t'+str(rl[i])+'\t'+str(Phl[i])+'\t'+str(Ksl[i])+'\t'+str(Ktl[i])+'\t'+str(Wtl[i])+'\t'+str(Ssl[i])+'\t'+str(Stl[i])+'\t'+str(Sel[i])+'\t'+str(Nul[i])+'\t'+str(Knl[i])+'\t'+str(Fil[i])+'\t'+str(Syl[i])+'\t'+str(Esl[i])+'\t'+str(Etl[i])+'\t'+str(Enl[i])+'\t'+str(Vnl[i])+'\t'+str(Vel[i])+'\t'+str(Val[i])+'\t'+str(Lal[i])+'\t'+str(Gal[i])+'\t'+str(Wsl[i])+'\t'+str(Eel[i])+'\n')
    fout.close()

    # margins for horizontal plot
    hmarb = 0.185
    hmarl = 0.15
    hmart = 0.88
    hmarr = 0.98

    # margins for vertical / horizontal plot
    vmarb = 0.12
    vmarl = 0.21
    vmart = 0.88
    vmarr = 0.95

    # margins for square plot
    smarb = 0.13
    smarl = 0.20
    smart = 0.90
    smarr = 0.96

    # font size
    tfont = 56 # title font
    lfont = 52 # label font
    mfont = 48 # medium (tick label, legend)

    smin=min(sl)
    smax=max(sl)
    xmax=max(xl)+0.1
    cwtmin=-max(Wtl)*0.05
    cwtmax=max(Wtl)*1.05
    sigmin=min(Ssl+Stl+Sel+Syl+[0])
    sigmax=max(Ssl+Stl+Sel+Syl)
    sigrng=sigmax-sigmin
    sigmax+=sigrng/20
    sigmin-=sigrng/20
    kmin=min(Ksl+Ktl)
    kmax=max(Ksl+Ktl)
    krng=kmax-kmin
    kmin-=krng/20
    kmax+=krng/20

    CellWallColor = 'crimson'
    KsColor = 'seagreen'
    KtColor = 'limegreen'
    ExtColor = 'peru'
    SyColor = 'darkorange'
    SeColor = 'darkslateblue'
    EpsColor = 'magenta'
    VeloColor = 'purple'
    ExpColor = 'pink'

    eR=[]
    eX=[]
    mR=[]
    mX=[]
    tR=[]
    tX=[]
    j=int((len(Val)-1)/2)
    for i in range(len(s0)):
        eR.append(R0[i]-Val[j]*np.sin(Ph0[i])/2)
        eX.append(X0[i]+Val[j]*np.cos(Ph0[i])/2)
        mR.append(R0[i]-Vnl[j]*np.sin(Ph0[i])/2)
        mX.append(X0[i]+Vnl[j]*np.cos(Ph0[i])/2)
        tR.append(R0[i])
        tX.append(X0[i]+Vnl[int((len(Val)-1)/2)]/2)
        j+=1
    gromax=max(X0+eX+tX+mX)

    LoV = False
    if max(Vnl)<1:
        Vnl=[ v*1000 for v in Vnl]
        Vxl=[ v*1000 for v in Vxl]
        Val=[ v*1000 for v in Val]
        Vel=[ v*1000 for v in Vel]
        LoV = True
    vmin=min(Vnl+Val+Vel)
    vmax=max(Vnl+Val+Vel)
    vrng=vmax-vmin
    vmin-=vrng/20
    vmax+=vrng/20

    LoP = False
    if max(Fil)<0.01:
        Fil=[ p*1000 for p in Fil]
        LoP = True

    LoE = False
    if max(Esl)<0.1:
        Esl=[ e*1000 for e in Esl]
        Etl=[ e*1000 for e in Etl]
        Enl=[ e*1000 for e in Enl]
        LoE = True

    LoEel = False
    if max(Eel) < 0.1:
        Eel = [ p*1000 for p in Eel ]
        LoEel = True

    imid = int((len(sl)-1)/2)

    pngout = 'H_Inistat_cwt_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'$\mathrm{ Cell\ Wall\ Thickness}$', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Wtl, marker = 'o', s = 80, color = CellWallColor)
    plt.xlim(smin,smax)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.ylim(cwtmin,cwtmax)
    plt.ylabel(r'$\delta\ (\si{\micro\metre})$', size = lfont)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_cwt_x.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'$\mathrm{ Cell\ Wall\ Thickness}$', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(xl, Wtl, marker = 'o', s = 80, color = CellWallColor)
    plt.xlim(-0.5,xmax)
    plt.xlabel(r'$x\ (\si{\micro\metre})$', size = lfont)
    plt.ylim(cwtmin,cwtmax)
    plt.ylabel(r'$\delta\ (\si{\micro\metre})$', size = lfont)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_cwt_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_ticks(ticks = np.arange(0,max(Wtl[imid:])+0.1,0.1))
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title(r'$\mathrm{ Cell\ Wall\ Thickness}$', size = tfont, y=1.05)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Wtl[imid:], sl[imid:], marker = 'o', s = 80, color = CellWallColor)
    plt.xlim(cwtmin,cwtmax)
    plt.xlabel(r'$\delta\ (\si{\micro\metre})$', size = lfont, labelpad = 15)
    plt.ylim(smax,0)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont, rotation=-90, labelpad = 70)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_cwt_x.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_ticks(ticks = np.arange(0,max(Wtl[imid:])+0.1,0.1))
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title(r'$\mathrm{ Cell\ Wall\ Thickness}$', size = tfont, y=1.05)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Wtl, xl, marker = 'o', s = 80, color = CellWallColor)
    plt.xlim(cwtmin,cwtmax)
    plt.xlabel(r'$\delta\ (\si{\micro\metre})$', size = lfont, labelpad = 15)
    plt.ylim(xmax,0)
    plt.ylabel(r'$x\ (\si{\micro\metre})$', size = lfont, rotation=-90, labelpad = 70)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Kappas_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title('Meridional Curvature', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Ksl, marker = 'o', s = 80, color = KsColor)
    plt.xlim(smin,smax)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.ylim(kmin,kmax)
    plt.ylabel(r'$\kappa_s\ (\si{\micro\metre^{-1}})$', size = lfont)
    t = ax.yaxis.get_offset_text()
    t.set_size(6)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Kappas_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title('Meridional\nCurvature', size = tfont, y=1.03)
    plt.scatter(Ksl[imid:], sl[imid:], marker = 'o', s = 80, color = KsColor)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.xlim(kmin-(kmax-kmin)*0.05, kmax+(kmax-kmin)*0.05)
    plt.ylim(smax,0)
    plt.xlabel(r'$\kappa_s\ (\si{\micro\metre^{-1}})$', size = lfont, labelpad=15)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont ,rotation=-90, labelpad = 70)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Kappat_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title('Circumferential Curvature', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Ktl, marker = 'o', s = 80, color = KtColor)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(smin,smax)
    plt.ylabel(r'$\kappa_\theta\ (\si{\micro\metre^{-1}})$', size = lfont)
    plt.ylim(kmin,kmax)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Kappat_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title('Circumferential\nCurvature', size = tfont, y=1.03)
    plt.scatter(Ktl[imid:], sl[imid:], marker = 'o', s = 80, color = KtColor)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.xlim(kmin-(kmax-kmin)*0.05, kmax+(kmax-kmin)*0.05)
    plt.ylim(smax,0)
    plt.xlabel(r'$\kappa_\theta\ (\si{\micro\metre^{-1}})$', size = lfont, labelpad=15)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont ,rotation=-90, labelpad = 70)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Kappast_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title('Curvatures', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Ksl, marker = 'o', s = 80, color = KsColor, label=r'$\kappa_s$')
    plt.scatter(sl, Ktl, marker = 'o', s = 80, color = KtColor, label=r'$\kappa_\theta$')
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(smin,smax)
    plt.ylabel(r'$\kappa_i\ (\si{\micro\metre^{-1}})$', size = lfont)
    plt.ylim(kmin,kmax)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    leg=plt.legend(loc='center right', fontsize = mfont)
    leg.draw_frame(False)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Kappast_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title('Curvatures', size = tfont, y=1.03)
    plt.scatter(Ksl[imid:], sl[imid:], marker = 'o', s = 80, color = KsColor, label=r'$\kappa_s$')
    plt.scatter(Ktl[imid:], sl[imid:], marker = 'o', s = 80, color = KtColor, label=r'$\kappa_\theta$')
    plt.tick_params(axis='both', labelsize = mfont)
    plt.xlim(kmin-(kmax-kmin)*0.05, kmax+(kmax-kmin)*0.05)
    plt.ylim(smax,0)
    plt.xlabel(r'$\kappa_\theta\ (\si{\micro\metre^{-1}})$', size = lfont, labelpad=15)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont ,rotation=-90, labelpad = 70)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    leg=plt.legend(loc='lower center', fontsize = mfont)
    leg.draw_frame(False)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Kappast_Phi.png'
    fig = plt.figure(figsize = (12, 12))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = smarb, left = smarl-0.035, top = smart, right = smarr-0.035)
    plt.title('Curvatures', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Phl, Ksl, marker = 'o', s = 80, color = KsColor, label=r'$\kappa_s$')
    plt.scatter(Phl, Ktl, marker = 'o', s = 80, color = KtColor, label=r'$\kappa_\theta$')
    plt.xlabel(r'$\varphi\ (\mathrm{rad})$', size = lfont)
    plt.xlim(-np.pi/2,np.pi/2)
    plt.xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2],[r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$+\pi/4$', r'$+\pi/2$'])
    plt.ylabel(r'$\kappa_i\ (\si{\micro\metre^{-1}})$', size = lfont)
    plt.ylim(kmin,kmax)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axvline(x = -np.pi/2, color = 'gray', linewidth = 2, zorder = 0)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axvline(x = np.pi/2, color = 'gray', linewidth = 2, zorder = 0)
    leg=plt.legend(loc='lower center', fontsize = mfont)
    leg.draw_frame(False)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Sigmae_Phi.png'
    fig = plt.figure(figsize = (12, 12))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = smarb, left = smarl-0.035, top = smart, right = smarr-0.035)
    plt.title(r'Global Stress $vs$ N. Angle', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Phl, Sel, marker = 'o', s = 80, color = SeColor, label=r'$\sigma_e$')
    plt.plot(Phl,Syl, color = SyColor, linewidth = 6, zorder = 0, label=r'$\sigma_y$')
    plt.xlabel(r'$\varphi\ (\mathrm{rad})$', size = lfont)
    plt.xlim(-np.pi/2,np.pi/2)
    plt.xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2],[r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$+\pi/4$', r'$+\pi/2$'])
    plt.ylabel(r'$\sigma_e\ (\si{\mega\pascal})$', size = lfont)
    plt.ylim(sigmin,sigmax)
    plt.axvline(x = -np.pi/2, color = 'gray', linewidth = 2, zorder = 0)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axvline(x = np.pi/2, color = 'gray', linewidth = 2, zorder = 0)
    leg=plt.legend(fontsize=36)
    leg.draw_frame(False)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Ks_Kt.png'
    fig = plt.figure(figsize = (12, 12))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = smarb, left = smarl, top = smart, right = smarr)
    plt.title(r'$\kappa_\theta = f(\kappa_s)$', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Ksl, Ktl, marker = 'o', s = 80, color = 'green')
    plt.scatter(Ksl[int((len(Ksl)-1)/2)], Ktl[int((len(Ktl)-1)/2)], marker = 'o', s = 96, color = 'forestgreen')
    plt.xlabel(r'$\kappa_s\ (\si{\micro\metre^{-1}})$', size = lfont)
    plt.xlim(kmin,kmax)
    plt.ylabel(r'$\kappa_\theta\ (\si{\micro\metre^{-1}})$', size = lfont)
    plt.ylim(kmin,kmax)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Phi_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Extensibility', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Fil, marker = 'o', s = 80, color = ExtColor)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(smin,smax)
    plt.ylim(-max(Fil+[0.1])*0.05,max(Fil+[0.1])*1.05)
    if LoP:
        plt.ylabel(r'$\Phi\ (\times 10^{-3}\ \si{\minute^{-1}\cdot \mega\pascal^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$\Phi\ (\si{\minute^{-1}\cdot \mega\pascal^{-1}})$', size = lfont)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Phi_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title('Extensibility', size = tfont, y=1.03)
    plt.scatter(Fil[imid:], sl[imid:], marker = 'o', s = 80, color = ExtColor)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.xlim(-0.01,max(Fil+[0.1])*1.05)
    if LoP:
        plt.xlabel(r'$\Phi\ (\times 10^{-3}\ \si{\minute^{-1}\cdot \mega\pascal^{-1}})$', size = lfont/1.1, labelpad=15)
    else:
        plt.xlabel(r'$\Phi\ (\si{\minute^{-1}\cdot \mega\pascal^{-1}})$', size = lfont, labelpad=15)
    plt.ylim(smax,0)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont ,rotation=-90, labelpad = 70)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Phi_x.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Extensibility', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(xl, Fil, marker = 'o', s = 80, color = ExtColor)
    plt.xlabel(r'$x\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(-0.5,xmax)
    plt.ylim(-max(Fil+[0.1])*0.05,max(Fil+[0.1])*1.05)
    if LoP:
        plt.ylabel(r'$\Phi\ (\times 10^{-3}\ \si{\minute^{-1}\cdot \mega\pascal^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$\Phi\ (\si{\minute^{-1}\cdot \mega\pascal^{-1}})$', size = lfont)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Epsilons_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Meridional Stress', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Esl, marker = 'o', s = 80, color = EpsColor)
    plt.xlim(smin,smax)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.ylim(-max(Esl+Etl)*0.05,max(Esl+Etl)*1.05)
    if LoE:
        plt.ylabel(r'$\dot{\varepsilon}_s\ (\si{\nano\meter\cdot\micro\meter^{-1}\cdot\minute^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$\dot{\varepsilon}_s\ (\si{\micro\meter\cdot\micro\meter^{-1}\cdot\minute^{-1}})$', size = lfont)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Epsilons_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title('Meridional\nStrain rate', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Esl[imid:], sl[imid:], marker = 'o', s = 80, color = EpsColor)
    plt.xlim(-max(Esl+Etl)*0.05,max(Esl+Etl)*1.05)
    if LoE:
        plt.xlabel(r'$\dot{\varepsilon}_s\ (\si{\nano\meter\cdot\micro\meter^{-1}\cdot\minute^{-1}})$', size = lfont, labelpad = 15)
    else:
        plt.xlabel(r'$\dot{\varepsilon}_s\ (\si{\micro\meter\cdot\micro\meter^{-1}\cdot\minute^{-1}})$', size = lfont, labelpad = 15)
    plt.ylim(smax,0)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont ,rotation=-90, labelpad = 70)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Epsilont_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Circumferential Stress', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Etl, marker = 'o', s = 80, color = EpsColor)
    plt.xlim(smin,smax)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.ylim(-max(Esl+Etl)*0.05,max(Esl+Etl)*1.05)
    if LoE:
        plt.ylabel(r'$\dot{\varepsilon}_\theta\ (\si{\nano\meter\cdot\micro\meter^{-1}\cdot\minute^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$\dot{\varepsilon}_\theta\ (\si{\micro\meter\cdot\micro\meter^{-1}\cdot\minute^{-1}})$', size = lfont)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Epsilont_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title('Circumferential\nStrain Rate', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Etl[imid:], sl[imid:], marker = 'o', s = 80, color = EpsColor)
    plt.xlim(-max(Esl+Etl)*0.05,max(Esl+Etl)*1.05)
    if LoE:
        plt.xlabel(r'$\dot{\varepsilon}_\theta\ (\si{\nano\meter\cdot\micro\meter^{-1}\cdot\minute^{-1}})$', size = lfont, labelpad = 15)
    else:
        plt.xlabel(r'$\dot{\varepsilon}_\theta\ (\si{\micro\meter\cdot\micro\meter^{-1}\cdot\minute^{-1}})$', size = lfont, labelpad = 15)
    plt.ylim(smax,0)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont ,rotation=-90, labelpad = 70)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Epsilonn_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Normal Strain Rate', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Enl, marker = 'o', s = 80, color = EpsColor)
    plt.xlim(smin,smax)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.ylim(min(Enl)-(max(Enl)-min(Enl))*0.05,max(Enl)+(max(Enl)-min(Enl))*0.05)
    if LoE:
        plt.ylabel(r'$\dot{\varepsilon}_n\ (\si{\nano\meter\cdot\micro\meter^{-1}\cdot\minute^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$\dot{\varepsilon}_n\ (\si{\micro\meter\cdot\micro\meter^{-1}\cdot\minute^{-1}})$', size = lfont)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Epsilonn_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title('Normal\nStrain Rate', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Enl[imid:], sl[imid:], marker = 'o', s = 80, color = EpsColor)
    plt.xlim(min(Enl)-(max(Enl)-min(Enl))*0.05,max(Enl)+(max(Enl)-min(Enl))*0.05)
    if LoE:
        plt.xlabel(r'$\dot{\varepsilon}_n\ (\si{\nano\meter\cdot\micro\meter^{-1}\cdot\minute^{-1}})$', size = lfont, labelpad = 15)
        plt.xticks(np.arange(-30,10,10))
    else:
        plt.xlabel(r'$\dot{\varepsilon}_n\ (\si{\micro\meter\cdot\micro\meter^{-1}\cdot\minute^{-1}})$', size = lfont, labelpad = 15)
    plt.ylim(smax,0)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont ,rotation=-90, labelpad = 70)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Epsilonn_cwt.png'
    fig = plt.figure(figsize = (12, 12))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = smarb, left = smarl, top = smart, right = smarr)
    plt.title(r'Normal Strain Rate', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Wtl, Enl, marker = 'o', s = 80, color = EpsColor)
    plt.scatter(Wtl[int((len(Wtl)-1)/2)], Enl[int((len(Enl)-1)/2)], marker = 'o', s = 96, color = 'mediumorchid')
    plt.xlim(cwtmin,cwtmax)
    plt.xticks(np.arange(0,cwtmax,0.1))
    plt.xlabel(r'$\delta\ (\si{\micro\metre})$', size = lfont)
    plt.ylim(min(Enl)-(max(Enl)-min(Enl))*0.05,max(Enl)+(max(Enl)-min(Enl))*0.05)
    if LoE:
        plt.yticks(np.arange(-30,10,10))
        plt.ylabel(r'$\dot{\varepsilon}_n\ (\si{\nano\meter\cdot\micro\meter^{-1}\cdot\minute^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$\dot{\varepsilon}_n\ (\si{\micro\meter\cdot\micro\meter^{-1}\cdot\minute^{-1}})$', size = lfont)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Lambda_Gamma.png'
    fig = plt.figure(figsize = (12, 12))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = smarb, left = smarl, top = smart, right = smarr)
    plt.title(r'$\lambda = f(\gamma)$', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Gal, Lal, marker = 'o', s = 80, color = 'gold')
    plt.scatter(Gal[int((len(Gal)-1)/2)], Lal[int((len(Lal)-1)/2)], marker = 'o', s = 96, color = 'goldenrod')
    plt.plot([-0.05,0.4],[-0.15,1.2], color = 'gray', linewidth = 2, zorder = 0)
    plt.plot([-0.05,0.4],[-0.05,0.4], color = 'gray', linewidth = 2, zorder = 0)
    plt.ylabel(r'$\lambda = (\dot{\varepsilon}_\theta-\dot{\varepsilon}_s)/(\dot{\varepsilon}_\theta+\dot{\varepsilon}_s)$', size = lfont)
    plt.ylim(-0.1,1.1)
    plt.xlabel(r'$\gamma = (\sigma_\theta-\sigma_s)/(\sigma_\theta+\sigma_s)$', size = lfont)
    plt.xlim(-0.05,0.35)
    plt.xticks([0.0,0.1,0.2,0.3],['0.0','0.1','0.2','0.3'])
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axvline(x = 1/3, color = 'gray', linewidth = 2, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 1, color = 'gray', linewidth = 2, zorder = 0)
    plt.fill_between([-0.4,0.4],[-0.4,0.4],[-1000,1000],color='lightgray',zorder=0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Gamma_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Stress Anisotropy', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Gal, marker = 'o', s = 80, color = 'gold')
    plt.xlim(smin,smax)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.ylim(min(Gal)-(max(Gal)-min(Gal))*0.05,max(Gal)+(max(Gal)-min(Gal))*0.05)
    plt.yticks(np.arange(0,0.4,0.1))
    plt.ylabel(r'$\gamma = (\sigma_\theta-\sigma_s)/(\sigma_\theta+\sigma_s)$', size = lfont)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 1/3, color = 'gray', linewidth = 2, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Lambda_s.png'
    sa = np.array(sl).astype(np.double)
    Lea = np.array(Lel).astype(np.double)
    LeOK = np.isfinite(Lea)
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Strain Rate Anisotropy', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Lal, marker = 'o', s = 80, color = 'gold')
    plt.scatter(sa[LeOK], Lea[LeOK], marker = 'o', s = 80, color = 'darkgoldenrod')
    plt.xlim(smin,smax)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.ylim(min(Lal)-(max(Lal)-min(Lal))*0.05,max(Lal)+(max(Lal)-min(Lal))*0.05)
    plt.ylabel(r'$\lambda = (\dot{\varepsilon}_\theta-\dot{\varepsilon}_s)/(\dot{\varepsilon}_\theta+\dot{\varepsilon}_s)$', size = lfont)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 1.0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Nu_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Flow coupling', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Nul, marker = 'o', s = 80, color = 'firebrick')
    plt.xlim(smin,smax)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.ylim(min(Nul)-(max(Nul)-min(Nul))*0.05,max(Nul)+(max(Nul)-min(Nul))*0.05)
    plt.ylabel(r'$\nu$', size = lfont)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0.5, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_H_K_s.png'
    plt.figure(figsize = (16, 9))
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title(r'$K = f(s)$', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Knl, marker = 'o', s = 80, color = 'orangered')
    plt.ylabel(r'$K\ (\si{\mega\pascal})$', size = lfont)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(smin,smax)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Sigmas_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Meridional Stress', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Ssl, marker = 'o', s = 80, color = SeColor)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(smin,smax)
    plt.ylabel(r'$\sigma_s\ (\si{\mega\pascal})$', size = lfont)
    plt.ylim(sigmin,sigmax)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Sigmas_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title('Meridional\nStress', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Ssl, sl, marker = 'o', s = 80, color = SeColor)
    plt.xlim(sigmin,sigmax)
    plt.xlabel(r'$\sigma_s\ (\si{\mega\pascal})$', size = lfont, labelpad = 15)
    plt.ylim(smax,0)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont ,rotation=-90, labelpad = 70)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Sigmat_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Circumferential Stress', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Stl, marker = 'o', s = 80, color = SeColor)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(smin,smax)
    plt.ylabel(r'$\sigma_\theta\ (\si{\mega\pascal})$', size = lfont)
    plt.ylim(sigmin,sigmax)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Sigmat_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title('Circumferential\nStress', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Stl, sl, marker = 'o', s = 80, color = SeColor)
    plt.xlim(sigmin,sigmax)
    plt.xlabel(r'$\sigma_\theta\ (\si{\mega\pascal})$', size = lfont, labelpad = 15)
    plt.ylim(smax,0)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont ,rotation=-90, labelpad = 70)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Sigmas_x.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Meridional Stress', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(xl, Ssl, marker = 'o', s = 80, color = SeColor)
    plt.xlabel(r'$x\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(-0.5,xmax)
    plt.ylabel(r'$\sigma_s\ (\si{\mega\pascal})$', size = lfont)
    plt.ylim(sigmin,sigmax)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Sigmat_x.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Circumferential Stress', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(xl, Stl, marker = 'o', s = 80, color = SeColor)
    plt.xlabel(r'$x\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(-0.5,xmax)
    plt.ylabel(r'$\sigma_\theta\ (\si{\mega\pascal})$', size = lfont)
    plt.ylim(sigmin,sigmax)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Sigmae_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Global stress', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Sel, marker = 'o', s = 80, color = SeColor, zorder = 2, label=r'$\sigma_e$')
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(smin,smax)
    plt.ylabel(r'$\sigma_e\ (\si{\mega\pascal})$', size = lfont)
    plt.ylim(sigmin,sigmax)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    leg.draw_frame(False)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Sigmaye_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Global stress', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Sel, marker = 'o', s = 80, color = SeColor, zorder = 2, label=r'$\sigma_e$')
    plt.plot(sl,Syl, color = SyColor, linewidth = 6, zorder = 1, label=r'$\sigma_y$')
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(smin,smax)
    plt.ylabel(r'$\sigma_e\ (\si{\mega\pascal})$', size = lfont)
    plt.ylim(sigmin,sigmax)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    leg=plt.legend(loc='center right', fontsize = mfont)
    leg.draw_frame(False)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Sigmaye_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title('Global Stress and\nYield threshold', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Sel, sl, marker = 'o', s = 80, color = SeColor, label=r'$\sigma_e$')
    plt.plot(Syl, sl, color = SyColor, linewidth = 6, zorder = 0, label=r'$\sigma_y$')
    plt.xlim(sigmin,sigmax)
    plt.xlabel(r'$\sigma\ (\si{\mega\pascal})$', size = lfont)
    plt.ylim(smax,0)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont ,rotation=-90, labelpad = 70)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    leg=plt.legend(loc='lower right', fontsize = mfont)
    leg.draw_frame(False)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Sigmay_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Yield Threshold', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Syl, marker = 'o', s = 80, color = SyColor, label=r'$\sigma_y$')
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(smin,smax)
    plt.ylabel(r'$\sigma_y\ (\si{\mega\pascal})$', size = lfont)
    plt.ylim(sigmin,sigmax)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Sigmae_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title('Global Stress', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Sel, sl, marker = 'o', s = 80, color = SeColor)
    plt.xlim(sigmin,sigmax)
    plt.xlabel(r'$\sigma_e\ (\si{\mega\pascal})$', size = lfont, labelpad = 15)
    plt.ylim(smax,0)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont ,rotation=-90, labelpad = 70)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Sigmae_x.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title('Global Stress '+r'$vs\ x', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(xl, Sel, marker = 'o', s = 80, color = SeColor, zorder = 2, label=r'$\sigma_e$')
    plt.plot(xl,Syl, color = SyColor, linewidth = 6, zorder = 1, label=r'$\sigma_y$')
    plt.xlabel(r'$x\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(-0.5,xmax)
    plt.ylabel(r'$\sigma_e\ (\si{\mega\pascal})$', size = lfont)
    plt.ylim(sigmin,sigmax)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    leg=plt.legend(loc='center right', fontsize = mfont)
    leg.draw_frame(False)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Sigmae_K.png'
    plt.figure(figsize = (12, 12))
    plt.subplots_adjust(bottom = smarb, left = smarl, top = smart, right = smarr)
    plt.title(r'$\sigma_e = f(K)$', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Knl, Sel, marker = 'o', s = 80, color = 'midnightblue', label=r'$\sigma_e$')
    plt.scatter(Knl[int((len(Knl)-1)/2)], Sel[int((len(Sel)-1)/2)], marker = 'o', s = 96, color = 'blue')
    plt.plot(Knl,Syl, color = 'coral', linewidth = 3, zorder = 0, label=r'$\sigma_y$')
    plt.xlabel(r'$K\ (\si{\mega\pascal})$', size = lfont)
    plt.ylabel(r'$\sigma_e\ (\si{\mega\pascal})$', size = lfont)
    plt.ylim(sigmin,sigmax)
    leg=plt.legend(loc='upper left', fontsize = mfont)
    leg.draw_frame(False)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Sigmae_Ks.png'
    plt.figure(figsize = (12, 12))
    plt.subplots_adjust(bottom = smarb, left = smarl, top = smart, right = smarr)
    plt.title(r'$\sigma_e = f(\kappa_s)$', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Ksl, Sel, marker = 'o', s = 80, color = 'chocolate', label=r'$\sigma_e$')
    plt.scatter(Ksl[int((len(Ksl)-1)/2)], Sel[int((len(Sel)-1)/2)], marker = 'o', s = 96, color = 'saddlebrown')
    plt.plot(Ksl,Syl, color = 'darkorange', linewidth = 3, zorder = 0, label=r'$\sigma_y$')
    plt.xlabel(r'$\kappa_s\ (\si{\micro\metre}^{-1})$', size = lfont)
    plt.xlim(kmin,kmax)
    plt.ylabel(r'$\sigma_e\ (\si{\mega\pascal})$', size = lfont)
    plt.ylim(sigmin,sigmax)
    leg=plt.legend(loc='upper left', fontsize = mfont)
    leg.draw_frame(False)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Nu_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title('Flow Coupling', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Nul, sl, marker = 'o', s = 80, color = 'firebrick')
    plt.xlim(min(Nul)-(max(Nul)-min(Nul))*0.05,max(Nul)+(max(Nul)-min(Nul))*0.05)
    plt.xlabel(r'$\nu$', size = lfont, labelpad = 15)
    plt.ylim(smax,0)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont ,rotation=-90, labelpad = 70)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Vn_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Normal Velocity', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Vnl, marker = 'o', s = 80, color = VeloColor)
    plt.xlim(smin,smax)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.ylim(vmin,vmax)
    if LoV:
        plt.ylabel(r'$V_n\ (\si{\nano\metre\cdot \minute^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$V_n\ (\si{\micro\metre\cdot \minute^{-1}})$', size = lfont)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Vn_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title('Normal Velocity', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Vnl, sl, marker = 'o', s = 80, color = VeloColor)
    plt.xlim(vmin,vmax)
    if LoV:
        plt.xlabel(r'$V_n\ (\si{\nano\metre\cdot \minute^{-1}})$', size = lfont, labelpad = 15)
    else:
        plt.xlabel(r'$V_n\ (\si{\micro\metre\cdot \minute^{-1}})$', size = lfont, labelpad = 15)
    plt.ylim(smax,0)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont ,rotation=-90, labelpad = 70)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    if eg>0:
        pngout = 'H_Inistat_Ven_s.png'
        fig = plt.figure(figsize = (16, 9))
        ax = fig.add_subplot(111)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(4)
        ax.xaxis.set_tick_params(width=4, length=10)
        ax.yaxis.set_tick_params(width=4, length=10)
        plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
        plt.title(r'Normal Velocity', size = tfont, y=1.03)
        plt.tick_params(axis='both', labelsize = mfont)
        plt.plot(sl, Val, zorder = 1, linewidth = 6, color = ExpColor, label = 'expected')
        plt.scatter(sl, Vnl, zorder = 2, marker = 'o', s = 80, color = VeloColor, label = 'model')
        plt.xlim(smin,smax)
        plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
        plt.ylim(vmin,vmax)
        if LoV:
            plt.ylabel(r'$V_n\ (\si{\nano\metre\cdot \minute^{-1}})$', size = lfont)
        else:
            plt.ylabel(r'$V_n\ (\si{\micro\metre\cdot \minute^{-1}})$', size = lfont)
        plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
        plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
        leg=plt.legend(loc='upper right', fontsize = mfont)
        leg.draw_frame(False)
        plt.savefig(pngout)
        plt.close()

    pngout = 'H_Inistat_Vn_x.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Normal Velocity', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(xl, Vnl, marker = 'o', s = 80, color = VeloColor)
    plt.xlabel(r'$x\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(-0.5,xmax)
    plt.ylim(vmin,vmax)
    if LoV:
        plt.ylabel(r'$V_n\ (\si{\nano\metre\cdot \minute^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$V_n\ (\si{\micro\metre\cdot \minute^{-1}})$', size = lfont)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Vx_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Axial Velocity', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Vxl, marker = 'o', s = 80, color = VeloColor)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(smin,smax)
    if LoV:
        plt.ylabel(r'$V_x\ (\si{\nano\metre\cdot \minute^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$V_x\ (\si{\micro\metre\cdot \minute^{-1}})$', size = lfont)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Vn_K.png'
    plt.figure(figsize = (12, 12))
    plt.subplots_adjust(bottom = smarb, left = smarl, top = smart, right = smarr)
    plt.title(r'$V_n = f(K)$', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Knl, Vnl, marker = 'o', s = 80, color = 'purple')
    plt.scatter(Knl[int((len(Knl)-1)/2)], Vnl[int((len(Vnl)-1)/2)], marker = 'o', s = 96, color = 'blue')
    plt.xlabel(r'$K\ (\si{\mega\pascal})$', size = lfont)
    if LoV:
        plt.ylabel(r'$V_n\ (\si{\nano\metre\cdot \minute^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$V_n\ (\si{\micro\metre\cdot \minute^{-1}})$', size = lfont)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Vn_Ks.png'
    plt.figure(figsize = (12, 12))
    plt.subplots_adjust(bottom = smarb, left = smarl, top = smart, right = smarr)
    plt.title(r'$V_n = f(\kappa_s)$', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Ksl, Vnl, marker = 'o', s = 80, color = 'darkorchid')
    plt.scatter(Ksl[int((len(Ksl)-1)/2)], Vnl[int((len(Vnl)-1)/2)], marker = 'o', s = 96, color = 'blue')
    plt.xlabel(r'$\kappa_s\ (\si{\micro\metre}^{-1})$', size = lfont)
    plt.xlim(kmin,kmax)
    if LoV:
        plt.ylabel(r'$V_n\ (\si{\nano\metre\cdot \minute^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$V_n\ (\si{\micro\metre\cdot \minute^{-1}})$', size = lfont)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_H_Vn_s.png'
    plt.figure(figsize = (16, 9))
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title(r'$V_n = f(s)$', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Vnl, marker = 'o', s = 80, color = 'purple')
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(smin,smax)
    if LoV:
        plt.ylabel(r'$V_n\ (\si{\nano\metre\cdot \minute^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$V_n\ (\si{\micro\metre\cdot \minute^{-1}})$', size = lfont)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Vn_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title('Normal Velocity', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Vnl, sl, marker = 'o', s = 80, color = 'purple')
    plt.xlim(-0.05*max(Vnl),max(Vnl)+0.05*max(Vnl))
    if LoV:
        plt.xlabel(r'$V_n\ (\si{\nano\metre\cdot \minute^{-1}})$', size = lfont, labelpad = 15)
    else:
        plt.xlabel(r'$V_n\ (\si{\micro\metre\cdot \minute^{-1}})$', size = lfont, labelpad = 15)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=-90)
    plt.ylim(smax,0)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont ,rotation=-90, labelpad = 70)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Van_s.png'
    plt.figure(figsize = (16, 9))
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title(r'$V_n = f(s)$', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.plot(sl, Val, zorder = 1, linewidth = 4, color = 'pink', label = 'adjusted')
    plt.scatter(sl, Vnl, zorder = 2, marker = 'o', s = 80, color = 'purple', label = 'model')
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(smin,smax)
    if LoV:
        plt.ylabel(r'$V_n\ (\si{\nano\metre\cdot \minute^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$V_n\ (\si{\micro\metre\cdot \minute^{-1}})$', size = lfont)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    leg=plt.legend(loc='upper right', fontsize = mfont)
    leg.draw_frame(False)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Vn_x.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Normal Velocity', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(xl, Vnl, marker = 'o', s = 80, color = VeloColor)
    plt.xlabel(r'$x\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(-0.5,xmax)
    plt.ylim(vmin,vmax)
    if LoV:
        plt.ylabel(r'$V_n\ (\si{\nano\metre\cdot \minute^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$V_n\ (\si{\micro\metre\cdot \minute^{-1}})$', size = lfont)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_Vx_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Axial Velocity', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Vxl, marker = 'o', s = 80, color = VeloColor)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(smin,smax)
    if LoV:
        plt.ylabel(r'$V_x\ (\si{\nano\metre\cdot \minute^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$V_x\ (\si{\micro\metre\cdot \minute^{-1}})$', size = lfont)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'H_Inistat_ExpStrRat_s.png'
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = hmarb, left = hmarl, top = hmart, right = hmarr)
    plt.title(r'Expected Strain Rate'+r'$vs\ s', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(sl, Eel, marker = 'o', s = 80, color = VeloColor)
    plt.xlabel(r'$s\ (\si{\micro\metre})$', size = lfont)
    plt.xlim(smin,smax)
    if LoEel:
        plt.ylabel(r'$\dot\varepsilon^*\ (\times 10^{-3}\ \si{\minute^{-1}})$', size = 52, labelpad = 15)
    else:
        plt.ylabel(r'$\dot\varepsilon^*\ (\si{\minute^{-1}})$', size = 52, labelpad = 15)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_ExpStrRat_s.png'
    fig = plt.figure(figsize = (9, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = vmarb, left = vmarl, top = vmart, right = vmarr)
    plt.title('Expected Strain Rate', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Eel, sl, marker = 'o', s = 80, color = 'purple')
    if LoEel:
        plt.xlabel(r'$\dot\varepsilon^*\ (\times 10^{-3}\ \si{\minute^{-1}})$', size = lfont, labelpad = 15)
    else:
        plt.xlabel(r'$\dot\varepsilon^*\ (\si{\minute^{-1}})$', size = lfont, labelpad = 15)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=-90)
    plt.ylim(smax,0)
    plt.ylabel(r'$s\ (\si{\micro\metre})$', size = lfont ,rotation=-90, labelpad = 70)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Vn_cwt.png'
    fig = plt.figure(figsize = (12, 12))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = smarb, left = smarl, top = smart, right = smarr)
    plt.title(r'N. Velocity $vs$ C.W. Thickness', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Wtl, Vnl, marker = 'o', s = 80, color = VeloColor)
    plt.scatter(Wtl[int((len(Wtl)-1)/2)], Vnl[int((len(Vnl)-1)/2)], marker = 'o', s = 96, color = 'blue')
    plt.xlim(cwtmin,cwtmax)
    plt.xlabel(r'$\delta\ (\si{\micro\metre})$', size = lfont)
    plt.xticks(np.arange(0,0.5,0.1))
    plt.ylim(vmin,vmax)
    if LoV:
        plt.ylabel(r'$V_n\ (\si{\nano\metre\cdot \minute^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$V_n\ (\si{\micro\metre\cdot \minute^{-1}})$', size = lfont)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Vn_Phi.png'
    fig = plt.figure(figsize = (12, 12))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(bottom = smarb, left = smarl-0.035, top = smart, right = smarr-0.035)
    plt.title(r'N. Velocity $vs$ C.W. Normal', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Phl, Vnl, marker = 'o', s = 80, color = VeloColor)
    plt.xlabel(r'$\varphi\ (\mathrm{rad})$', size = lfont)
    plt.xlim(-np.pi/2,np.pi/2)
    plt.xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2],[r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$+\pi/4$', r'$+\pi/2$'])
    plt.ylim(vmin,vmax)
    if LoV:
        plt.ylabel(r'$V_x\ (\si{\nano\metre\cdot \minute^{-1}})$', size = lfont, labelpad = 15)
    else:
        plt.ylabel(r'$V_x\ (\si{\micro\metre\cdot \minute^{-1}})$', size = lfont, labelpad = 15)
    plt.axvline(x = -np.pi/2, color = 'gray', linewidth = 2, zorder = 0)
    plt.axvline(x = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.axvline(x = np.pi/2, color = 'gray', linewidth = 2, zorder = 0)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_Vn_Sigmae.png'
    plt.figure(figsize = (12, 12))
    plt.subplots_adjust(bottom = smarb, left = smarl, top = smart, right = smarr)
    plt.title(r'$V_n = f(\sigma_e)$', size = tfont, y=1.03)
    plt.tick_params(axis='both', labelsize = mfont)
    plt.scatter(Sel, Vnl, marker = 'o', s = 80, color = 'purple')
    plt.scatter(Sel[int((len(Sel)-1)/2)], Vnl[int((len(Vnl)-1)/2)], marker = 'o', s = 96, color = 'blue')
    plt.xlabel(r'$\sigma_e\ (\si{\mega\pascal})$', size = lfont)
    if LoV:
        plt.ylabel(r'$V_x\ (\si{\nano\metre\cdot \minute^{-1}})$', size = lfont)
    else:
        plt.ylabel(r'$V_x\ (\si{\micro\metre\cdot \minute^{-1}})$', size = lfont)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    pngout = 'Inistat_ExpStrRat_Sigmae.png'
    fig = plt.figure(figsize = (16, 16))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4, length=10)
    ax.yaxis.set_tick_params(width=4, length=10)
    plt.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.10)
    plt.scatter(Sel, Eel, marker = 'o', s = 120, color = VeloColor)
    plt.title('Expected Strain Rate '+r'$vs$'+' Global Stress', size = 56, y = 1.03)
    plt.tick_params(axis='both', labelsize = 48)
    if min(Eel) == max(Eel):
        plt.ylim(min(Eel)*0.95, max(Eel)*1.05)
    else:
        plt.ylim(-min(Eel)*0.05, max(Eel)+(max(Eel)-min(Eel))*0.05)
    if LoEel:
        plt.ylabel(r'$\dot\varepsilon^*\ (\times 10^{-3}\ \si{\minute^{-1}})$', size = 52, labelpad = 15)
    else:
        plt.ylabel(r'$\dot\varepsilon^*\ (\si{\minute^{-1}})$', size = 52, labelpad = 15)
    plt.xlabel(r'$\sigma_e\ (\si{\mega\pascal})$', size = 52, labelpad = 7)
    plt.axhline(y = 0, color = 'gray', linewidth = 4, zorder = 0)
    plt.savefig(pngout)
    plt.close()

    return mX[0]-X0[0]

##############################################

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-T', '--turgor', type = float,
        help = 'default turgor pressure (MPa).', default = 0.495)
    parser.add_argument('-p', '--phi', type = float,
        help = 'Φ = cell wall extensibility (min⁻¹·MPa⁻¹).', default = 2.51E-3)
    parser.add_argument('-y', '--sigy', type = float,
        help = 'σy = stress yield threshold (MPa).', default = 11.18)
    parser.add_argument('-s', '--secondsbystep', type = float,
        help = 'time in seconds for 1 step.', default = 60)
    parser.add_argument('-e', '--expgro', type = float,
        help = 'expected growth rate (µm·h⁻¹).', default = 2.5)
    parser.add_argument('-v', '--voluble', action = 'store_true',
        help = 'write to stdout while working.')
    parser.add_argument('infile', type = str,
        help = 'path to input file.')
    
    args = parser.parse_args()
    
    try:
        instr = open(args.infile, 'r') # initial input stream
    except IOError:
        print('Cannot open', args.infile)
        sys.exit(1)
     
    paradic={}

    for arg, value in sorted(vars(args).items()):
        paradic[arg]=['constant',value]
    
    # READ INPUT FILE
    
    datdic = read_data_file(paradic,instr) # Dictionary of parameters and initial data

    used = []
    SetList = [nam for nam in datdic['Set']
               if nam not in used and (used.append(nam) or True)]
    
    if args.voluble:
        print('Dataset:')
        for nam in SetList:
            print(' ', nam)
        print()
    
    totdat = len(datdic['Set'])

    for nam in SetList:
    
        if args.voluble:
            print('  Drawing', nam)
    
        if len(paradic) > 0:
            print('    Parameters:')
            for para in paradic:
                if args.voluble:
                    print('   ', para, end = ':')
                    for val in paradic[para]:
                        if val != 'constant':
                            print('', val, end = '')
                    print()
                if (para == 'cwt' or para == 'sigy' or para == 'phi') and paradic[para][0] == 'file' :
                    # file is tab-delimited, header line begins with 's' and
                    # data lines are made of values of s and cwt or Sy or Phi
                    parfil = open(paradic[para][1],'r')
                    fils = []
                    filp = []
                    for li in parfil:
                        li = li[:-1]
                        pa = li.split('\t')
                        if pa[0][0] != 's':
                            fils.append(float(pa[0]))
                            filp.append(float(pa[1]))
                    paradic[para] = ['file',fils,filp,paradic[para][1]]
            print()
            
        # CREATE LIST OF Phi VALUES
        if 'phi' in paradic:
            phlst = [paradic['phi']]
        else:
            phlst=[['constant',args.phi]]
    
        # CREATE LIST OF Sigma_y VALUES
        if 'sigy' in paradic:
            sylst = [paradic['sigy']]
        else:
            sylst=[['constant',args.sigy]]

        ParaPhi=phlst[0]
        ParaSy=sylst[0]

        if ParaPhi[0]=='constant':
            PhiStr='{0:.2e}'.format(ParaPhi[1])
            PhiUni=PhiStr+' min$^{-1}$ MPa$^{-1}'
            PhiTTY=PhiStr+' min⁻¹ MPa⁻¹'
        else:
            PhiStr=ParaPhi[3]
            PhiUni='['+PhiStr+']'
            PhiTTY='['+PhiStr+']'
        
        if ParaSy[0]=='constant':                
            SyStr='{0:05.2f}'.format(ParaSy[1])
            SyUni=SyStr+' MPa'
            SyTTY=SyStr+' MPa'
        else:
            SyStr=ParaSy[3]
            SyUni='['+SyStr+']'
            SyTTY='['+SyStr+']'
    
        # BUILD INITIAL DATASET
        print('\n'+nam, end = ': ')
        print('Φ = '+PhiTTY, end = '')
        print(', σy = '+SyTTY, flush = True)
        R0 = []
        X0 = []
        s0 = []
        Ph0 = []
        Ks0 = []
        Kt0 = []
        for i in range(totdat):
            if datdic['Set'][i]==nam and datdic['R'][i]>=0:
                R0.append(datdic['R'][i])
                X0.append(datdic['X'][i])
                s0.append(datdic['s'][i])
                Ph0.append(datdic['Phi'][i])
                Ks0.append(datdic['Kappa_s'][i])
                Kt0.append(datdic['Kappa_t'][i])

    draw_charts(s0,X0,R0,Ph0,Ks0,Kt0,paradic['turgor'][1],paradic['cwt'],phlst[0],sylst[0],paradic['expgro'][1],paradic['secondsbystep'][1])
    
if __name__ == "__main__":
    main()

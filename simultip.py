#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
Simulate the tip growth.
Input: a tab-delimited file with data for each mean contour point.
Output: a tab-delimited file showing cell shape dynamics.
'''

'''
© Bernard Billoud / Morphogenesis of Macro-Algae team (CNRS/Sorbonne Université)

Bernard.Billoud@sb-roscoff.fr

This software is a computer program whose purpose is to simulate tip
growth under the viscoplastic model.

This software is governed by the CeCILL-B license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/or redistribute the software under the terms of the CeCILL-B
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
knowledge of the CeCILL-B license and that you accept its terms.
'''

from tiputil import *

##############################################

def main():
    
    # MANAGE COMMAND-LINE ARGUMENTS
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-u', '--unique',  type = str,
        help = 'use only the specified dataset ('' = use all).', default = '')
    parser.add_argument('-S', '--smooth', type = float,
        help = 'smoothing factor of contour spline.', default = 0.0)
    parser.add_argument('-T', '--turgor', type = float,
        help = 'default turgor pressure (MPa).', default = 0.495)
    parser.add_argument('-w', '--wall', type = float,
        help = 'cell wall (maximum) thickness (µm).', default = 0.591)
    parser.add_argument('-m', '--minwall', type = float,
        help = 'cell wall minimum thickness (µm).', default = 0.0362)
    parser.add_argument('-z', '--halfzone', type = float,
        help = 'width of half-thickness zone (µm).', default = 16.81)
    parser.add_argument('-n', '--numsteps', type = int,
        help = 'number of simulation steps (no max if 0).', default = 2000)
    parser.add_argument('-o', '--overhang', type = float,
        help = 'maximum advance (no max if 0; auto if < 0).', default = 25)
    parser.add_argument('-d', '--delta', type = float,
        help = 'write output every Δ steps or µm overhang.', default = 1)
    parser.add_argument('-s', '--secondsbystep', type = float,
        help = 'time in seconds for 1 step.', default = 60)
    parser.add_argument('-g', '--growthbysub', type = float,
        help = 'growth by substep (nm).', default = 1.0)
    parser.add_argument('-p', '--phi', type = float,
        help = 'Φ = cell wall extensibility (min⁻¹·MPa⁻¹).', default = 2.51E-3)
    parser.add_argument('-r', '--rangeforphi', type = float,
        help = 'Φ range (min⁻¹·MPa⁻¹).', default = 0.5E-3)
    parser.add_argument('-i', '--iforphi', type = float,
        help = 'Φ increase (min⁻¹·MPa⁻¹).', default = 0.5E-3)
    parser.add_argument('-y', '--sigy', type = float,
        help = 'σy = stress yield threshold (MPa).', default = 11.18)
    parser.add_argument('-a', '--rangefory', type = float,
        help = 'σy range (MPa).', default = 1)
    parser.add_argument('-j', '--jfory', type = float,
        help = 'σy increase (MPa).', default = 1)
    parser.add_argument('-e', '--expgro', type = float,
        help = 'expected growth rate (µm·h⁻¹).', default = 2.5)
    parser.add_argument('-R', '--Reffile', type = str,
        help = 'Reference file for contour.', default = '')
    parser.add_argument('-f', '--fileout', type = str,
        help = 'file for output.', default = 'simultip_out_'+str(os.getpid())+'.tab')
    parser.add_argument('-E', '--expdotepstar', type = str,
        help = 'name of Sigmae_ExpDotEpsilonStar file (default: do not write).', default ='')
    parser.add_argument('-c', '--charts', action = 'store_true',
        help = 'draw png charts for initital state.')
    parser.add_argument('-b', '--beforeafter', action = 'store_true',
        help = 'draw before/after as pdf.')
    parser.add_argument('-k', '--kinematics', action = 'store_true',
        help = 'produce (many) png images to follow growth.')
    parser.add_argument('-Z', '--zoom', action = 'store_true',
        help = 'produce zoomed kinematics.')
    parser.add_argument('-v', '--voluble', action = 'store_true',
        help = 'write to stdout while working.')
    parser.add_argument('-V', '--verbose', action = 'store_true',
        help = 'write too much to stdout while working.')
    parser.add_argument('infile', type = str,
        help = 'path to input file.')
    
    args = parser.parse_args()
    
    try:
        instr = open(args.infile, 'r') # initial input stream
    except IOError:
        print('Cannot open', args.infile)
        sys.exit(1)
    
    if args.Reffile != '':
        try:
            refstr = open(args.Reffile, 'r') # reference input stream
        except IOError:
            print('Cannot open', args.Reffile)
            sys.exit(1)
    
    try:
        fout = open(args.fileout, 'w') # file out
    except IOError:
        print('Cannot open', args.fileout)
        sys.exit(1)

    if args.expdotepstar != '':
        try:
            edes = open(args.expdotepstar, 'w')
        except IOError:
            print('Cannot open '+args.expdotepstar)
            sys.exit(1)
    
    deltastep = 0
    nextover = 0.0
    if args.overhang <= 0:
        if args.numsteps <= 0:
            print('Cannot simulate with overhang ≤ 0 and numsteps = ', args.numsteps)
            sys.exit(1)
        if args.delta > args.numsteps:
            print('Cannot simulate with delta = ',args.delta, '> numsteps = ', args.numsteps)
            sys.exit(1)
        deltastep = int(args.delta)
    else:
        if args.delta > args.overhang:
            print('Cannot simulate with delta = ',args.delta, '> overhang = ', args.overhang)
            sys.exit(1)
        nextover = args.delta
     
    paradic={}
    if args.voluble or args.verbose:
        print('\nParameters:')
    for arg, value in sorted(vars(args).items()):
        paradic[arg]=['constant',value]
        if args.voluble or args.verbose:
            print(' ', arg, ' = ', value)
    if args.voluble or args.verbose:
        print()

    # READ INPUT FILE(S)
    
    datdic = read_data_file(paradic,instr) # Dictionary of parameters and initial data
    
    if args.Reffile == '':
        refdic = {}
    else:
        refdic = read_ref_file(instr)   # Dictionary of reference data
    
    used = []
    SetList = [nam for nam in datdic['Set']
               if nam not in used and (used.append(nam) or True)]
    
    if args.voluble or args.verbose:
        print('Dataset:')
        for nam in SetList:
            print(' ', nam)
        print()

    # LAUNCH SIMULATION
    
    totdat = len(datdic['Set'])
    
    fout.write('Dataset\tPhi\tSigma_y\tTime\tG\tlogRk\tlogRd\tlogRe\n')
    
    for nam in SetList:
    
        if args.voluble or args.verbose:
            print('  Simulating', nam)
    
        if len(paradic) > 0:
            if args.voluble or args.verbose:
                print('    Parameters:')
            for para in paradic:
                if args.voluble or args.verbose:
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
            if args.voluble or args.verbose:
                print()
            
        if args.beforeafter:
            pdfout = PdfPages('Simulated_'+nam+'_{0:02d}x{1:02d}_{2}.pdf'.format(int(2*paradic['rangeforphi'][1]/paradic['iforphi'][1]+1.5),int(2*paradic['rangefory'][1]/paradic['jfory'][1]+1.5),os.getpid()))
    
        # BUILD INITIAL DATASET
        R0 = []
        X0 = []
        s0 = []
        Ph0 = []
        Ks0 = []
        Kt0 = []
        for i in range(totdat):
            if datdic['Set'][i] == nam and datdic['R'][i] >= 0:
                R0.append(datdic['R'][i])
                X0.append(datdic['X'][i])
                s0.append(datdic['s'][i])
                Ph0.append(datdic['Phi'][i])
                Ks0.append(datdic['Kappa_s'][i])
                Kt0.append(datdic['Kappa_t'][i])
        X0 = [ x-max(X0) for x in X0 ] # x=0 at tip
        
        # BUILD REFERENCE DATASET
        if args.Reffile!= '':
            Rr = []
            Xr = []
            sr = []
            Ksr = []
            totref = len(refdic['Set'])
            for i in range(totref):
                if refdic['R'][i] >= 0:
                    Rr.append(refdic['R'][i])
                    Xr.append(refdic['X'][i])
                    sr.append(refdic['s'][i])
                    Ksr.append(refdic['Kappa_s'][i])
            greylab = 'Translated'
            pinklab = 'Reference'
        else:
            Rr = R0
            Xr = X0
            sr = s0
            Ksr = Ks0
            greylab = ''
            pinklab = 'Translated'
    
        nptr = len(sr)
        if args.voluble or args.verbose:
            print('   ',nptr,'points in reference contour')
                
        deltR0 = max(X0)-max(Xr)
        for i in range(nptr) :
            Xr[i] = Xr[i]+deltR0
    
        # CREATE LIST OF Φ VALUES
        if paradic['phi'][0] == 'file' :
            phlst = [['file',paradic['phi'][1],paradic['phi'][2],paradic['phi'][3]]]
        elif paradic['iforphi'][1] <= 0 or paradic['rangeforphi'][1] <= 0:
                phlst = [['constant',paradic['phi'][1]]]
        else:
            phlst = []
            Phi_ext = round(paradic['phi'][1]-paradic['rangeforphi'][1], 7)
            while Phi_ext <= round(paradic['phi'][1]+paradic['rangeforphi'][1], 7) :
                phlst.append(['constant',Phi_ext])
                Phi_ext += paradic['iforphi'][1]
                Phi_ext = round(Phi_ext, 7)
    
        # CREATE LIST OF σy VALUES
        if paradic['sigy'][0] == 'file' :
            sylst = [['file',paradic['sigy'][1],paradic['sigy'][2],paradic['sigy'][3]]]
        elif paradic['jfory'][1] <= 0 or paradic['rangefory'][1] <= 0:
            sylst = [['constant',paradic['sigy'][1]]]
        else:
            sylst = []
            Sy = round(paradic['sigy'][1]-paradic['rangefory'][1], 7)
            while Sy <= round(paradic['sigy'][1]+paradic['rangefory'][1], 7) :
                sylst.append(['constant',Sy])
                Sy += paradic['jfory'][1]
                Sy = round(Sy, 7)

        # LOOP THROUGH VALUES OF Φ AND σy
        lRda = []
        for ParaPhi in phlst:
            
            if ParaPhi[0] == 'constant':
                PhiOut='{:.3e}'.format(ParaPhi[1])
                if ParaPhi[1] > 1E-1:
                    PhiStr = '{0:04.2f}'.format(ParaPhi[1])
                    PhiUni = PhiStr+' min$^{-1}$ MPa$^{-1}$'
                    PhiTTY = PhiStr+' min⁻¹·MPa⁻¹'
                else:
                    PhiStr = '{0:04.2f}'.format(ParaPhi[1]*1000)
                    PhiUni = PhiStr+'$\\times 10^{-3}$ min$^{-1}$ MPa$^{-1}$'
                    PhiTTY = PhiStr+'×10⁻³ min⁻¹·MPa⁻¹'
                    
            else:
                PhiStr = ParaPhi[3]
                PhiUni = '['+PhiStr+']'
                nphu = ''
                for k in PhiUni:
                    if k == '_':
                        nphu = nphu+'\\'
                    nphu = nphu+k
                PhiUni = nphu
                PhiTTY = '['+PhiStr+']'
                PhiOut = PhiTTY

            lRdl = []

            for ParaSy in sylst:
    
                if ParaSy[0] == 'constant':                
                    SyStr = '{0:05.2f}'.format(ParaSy[1])
                    SyUni = SyStr+' MPa'
                    SyTTY = SyStr+' MPa'
                else:
                    SyStr = ParaSy[3]
                    SyUni = '['+SyStr+']'
                    nphu = ''
                    for k in SyUni:
                        if k == '_':
                            nphu = nphu+'\\'
                        nphu = nphu+k
                    SyUni = nphu
                    SyTTY = '['+SyStr+']'
                
                if args.voluble or args.verbose:
                    print('\n'+nam, end = ': ')
                    print('Φ = '+PhiTTY, end = '')
                    print(', σy = '+SyTTY, flush = True)
                '''
                print(Ks0[int((len(Ks0)-1)/2)-5:int((len(Ks0)-1)/2)+6])
                '''
                
                # PREDICT TIP ADVANCE FOR FIRST STEP
                if args.charts:
                    # DRAW CHARTS FOR INITIAL STATE
                    igro = draw_charts(s0, X0, R0, Ph0, Ks0, Kt0,
                                       paradic['turgor'][1], paradic['cwt'],
                                       ParaPhi, ParaSy, paradic['expgro'][1],
                                       paradic['secondsbystep'][1])
                else:
                    igro = tipadvance(Ks0, Kt0, paradic['turgor'][1],
                                      paradic['cwt'], ParaPhi[:3],
                                      ParaSy[:3]
                                      )*paradic['secondsbystep'][1]/60
                
                npt0 = len(s0)
                nm = 0
                px0 = (-R0[1], X0[1])
                pxm = ( R0[0], X0[0])
                px1 = ( R0[1], X0[1])
                '''
                print('Time = 0', npt0, curvature(px0, pxm, px1), px0, pxm, px1)
                print(npt0, 'points R0 = ', R0[int((npt0-1)/2)], flush = True)
                '''
                if args.kinematics:
                    if not os.path.isdir('Global'):
                        os.mkdir('Global')
                    pngout = 'Global/Sim_'+nam+'_G_p'+PhiStr+'s'+SyStr+'_t{0:06d}.png'.format(0)
                    mR0 = [-r for r in R0]
                    mRr = [-r for r in Rr]
                    XpD = X0
                    RpD = R0
                    nR = R0
                    mnR = mR0
                    nX = X0
                    rrg = (max(R0)-min(R0))*1.05
                    if args.overhang < 0:
                        xrg = (max(X0)-min(X0))*1.5
                    else:
                        xrg = (max(X0)+args.overhang-min(X0))*1.05
                    frg = max(rrg, xrg)
                    rmd = 0
                    plt.figure(figsize = (8, 8))
                    ax = plt.subplot()
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_fontsize(25)
                    plt.subplots_adjust(left = 0.151, right = 0.93, bottom = 0.113, top = 0.892)
                    if args.Reffile!= '':
                        plt.plot(R0, XpD, lw = 3, color = 'darkgrey', zorder = 1, label = greylab)
                        plt.plot(mR0, XpD, lw = 3, color = 'darkgrey', zorder = 1)
                    plt.plot(Rr, XpD, lw = 3, color = 'purple', zorder = 2, label = pinklab)
                    plt.plot(mRr, XpD, lw = 3, color = 'purple', zorder = 2)
                    plt.plot(R0, X0, lw = 3, color = 'green', zorder = 3, label = 'Initial')
                    plt.plot(mR0, X0, lw = 3, color = 'green', zorder = 3)
                    plt.plot(nR, nX, lw = 3, color = 'blue', zorder = 4, label = 'Model')
                    plt.plot(mnR, nX, lw = 3, color = 'blue', zorder = 4)
                    plt.title(r'$\Phi$ = '+PhiUni+'; $\sigma_y$ = '+SyUni+'\nTime = '+s2hhmmss(0)+'; Growth = {0:05.2f} '.format(0)+r'$\si{\micro\metre}$',size=25)
                    plt.xlim(-20, 20)
                    plt.ylim(max(X0)-13, max(X0)+27)
                    plt.xlabel(r'$r\ (\si{\micro\metre})$', size = 30)
                    plt.ylabel(r'$x\ (\si{\micro\metre})$', size = 30)
                    leg = plt.legend(loc = 'lower right', fontsize = 15)
                    leg.draw_frame(False)
                    plt.savefig(pngout)
                    plt.close()

                    # Draw vertical plot
                    if not os.path.isdir('Vert'):
                        os.mkdir('Vert')
                    pngout = 'Vert/Sim_'+nam+'_V_p'+PhiStr+'s'+SyStr+'_t{0:06d}.png'.format(0)
                    plt.figure(figsize = (4.46, 8))
                    ax = plt.subplot()
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_fontsize(25)
                    plt.subplots_adjust(left = 0.28, right = 0.955, bottom = 0.115, top = 0.865)
                    if args.Reffile!= '':
                        plt.plot(R0, XpD, lw = 3, color = 'darkgrey', zorder = 1)
                        plt.plot(mR0, XpD, lw = 3, color = 'darkgrey', zorder = 1)
                    plt.plot(Rr, XpD, lw = 3, color = 'purple', zorder = 2)
                    plt.plot(mRr, XpD, lw = 3, color = 'purple', zorder = 2)
                    plt.plot(R0, X0, lw = 3, color = 'green', zorder = 3)
                    plt.plot(mR0, X0, lw = 3, color = 'green', zorder = 3)
                    plt.plot(nR, nX, lw = 3, color = 'blue', zorder = 4)
                    plt.plot(mnR, nX, lw = 3, color = 'blue', zorder = 4)
                    plt.title('')
                    plt.gcf().text(0.2, 0.95, 'Time = '+s2hhmmss(0), fontsize=25)
                    plt.gcf().text(0.2, 0.90, 'Growth = {0:05.2f} '.format(0)+r'$\si{\micro\metre}$', fontsize=25)
                    plt.xlim(-10, 10)
                    plt.ylim(max(X0)-13, max(X0)+27)
                    plt.xlabel(r'$r\ (\si{\micro\metre})$', size = 30)
                    plt.ylabel(r'$x\ (\si{\micro\metre})$', size = 30)
                    plt.savefig(pngout)
                    plt.close()
    
                    if args.zoom:
                        if not os.path.isdir('Zoom'):
                            os.mkdir('Zoom')
                        pngout = 'Zoom/Sim_'+nam+'_Z_p'+PhiStr+'s'+SyStr+'_t{0:06d}.png'.format(0)
                        Xhi = max(X0)+0.5
                        Xlo = Xhi-12
                        plt.figure(figsize = (8, 8))
                        ax = plt.subplot()
                        for label in ax.get_xticklabels() + ax.get_yticklabels():
                            label.set_fontsize(25)
                        plt.subplots_adjust(left = 0.151, right = 0.93, bottom = 0.113, top = 0.892)
                        if args.Reffile!= '':
                            plt.plot(R0, XpD, lw = 8, color = 'lightgrey', zorder = 1, label = greylab)
                            plt.plot(mR0, XpD, lw = 8, color = 'lightgrey', zorder = 1)
                        plt.plot(Rr, XpD, lw = 8, color = 'plum', zorder = 2, label = pinklab)
                        plt.plot(mRr, XpD, lw = 8, color = 'plum', zorder = 2)
                        plt.plot(nR, nX, lw = 8, color = 'lightblue', zorder = 3, label = 'Model')
                        plt.plot(mnR, nX, lw = 8, color = 'lightblue', zorder = 3)
                        plt.scatter(R0, XpD, marker = 'o', s = 16, color = 'darkgrey', zorder = 4)
                        plt.scatter(mR0, XpD, marker = 'o', s = 16, color = 'darkgrey', zorder = 4)
                        plt.scatter(Rr, XpD, marker = 'o', s = 16, color = 'purple', zorder = 5)
                        plt.scatter(mRr, XpD, marker = 'o', s = 16, color = 'purple', zorder = 5)
                        plt.scatter(nR, nX, marker = 'o', s = 16, color = 'blue', zorder = 6)
                        plt.scatter(mnR, nX, marker = 'o', s = 16, color = 'blue', zorder = 6)
                        plt.title(r'$\Phi$ = '+PhiUni+'; $\sigma_y$ = '+SyUni+'\nTime = '+s2hhmmss(0)+'; Growth = {0:05.2f} '.format(0)+r'$\si{\micro\metre}$',size=25)
                        plt.xlim(-6, 6)
                        plt.ylim(Xlo, Xhi)
                        plt.xlabel(r'$r\ (\si{\micro\metre})$', size = 30)
                        plt.ylabel(r'$x\ (\si{\micro\metre})$', size = 30)
                        leg = plt.legend(loc = 'lower center', fontsize = 15)
                        leg.draw_frame(False)
                        plt.savefig(pngout)
                        plt.close()
                        
                if args.kinematics or args.expdotepstar != '':
                    ''' Compute expected strain rate = f(stress) '''
                    # compute shifted contour
                    sbs = paradic['secondsbystep'][1]
                    iX = paradic['expgro'][1]/3600*sbs # expgro in µm·h⁻¹, iX = advance for 1 step
                    dR = [R0[0]]    # Growth similar to shift along X axis: R does not change
                    dX = [X0[0]+iX] # X increases by ix; Shifted dome made of (dR,dX) points
                    i = 1
                    while i < len(R0):
                        dR.append(R0[i])
                        dR.insert(0,R0[i])
                        dX.append(X0[i]+iX)
                        dX.insert(0,X0[i]+iX)
                        i += 1
                    nptd = len(dR)
                    tckspl, uc = interpolate.splprep([dR, dX], k = 3, s = 0) # -> spline
                
                    # interpolate shifted contour
                    splstp = 1/(2*len(dR))    # approx. spline step
                    t0 = findR0(tckspl, 0.5, splstp)
                    splstp = (1-t0)/(len(dR)) # correct spline step
                    trange = np.append(np.arange(t0, 1, splstp), [1]) # build t range
                    splpts = interpolate.splev(trange, tckspl)        # interpolate
                    npts = len(splpts[0])

                    # list resampled points
                    sR = []
                    sX = []
                    for i in range(npts):
                        sR.append(splpts[0][i])
                        sX.append(splpts[1][i])
                    sR[0]=0

                    iVnl = [iX*60/sbs]   # iVnl = adjusted Vn (in µm·min⁻¹) at each point
                    ci = 1              
                    while ci < len(s0):  
                        fwdpt = (R0[ci],X0[ci]+1.0) # Point 1 µm ahead
                        ctrpt = (R0[ci],X0[ci])     # central point
                        ctrph = Ph0[ci]             # angle between x axis and normal to cell wall
                        if -np.pi/2+1E-6<ctrph<np.pi/2-1E-6:
                            sj = 1
                            Vn = 0
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
                                Vn = eucdist(ctrpt,(splpt[0],splpt[1]))
                        else:
                            Vn = 0
                        iVnl.append(Vn*60/sbs)
                        ci+=1
                    
                    iSel = [] # Global stress σe (MPa)
                    iPSl = [] # ε* = Φ(σe-σy) computed using K κθ Vn /(σθ - ν σs) (min⁻¹)
                    for i in range(len(s0)):
                        s = s0[i]
                        Ks = Ks0[i]
                        Kt = Kt0[i]
                        cwt = para_func(paradic['cwt'],s)
                        Ss = paradic['turgor'][1]/(2*cwt*Kt)     # σs = T/(2 δ κθ)
                        St = Ss*(2-Ks/Kt)                        # σθ = T/(2 δ κθ) (2 - κs/κθ)
                        nu = (1-Ks/Kt)/2                         # ν = (1 - κs/κθ)/2
                        Se2 = nu*(St-Ss)**2+(1-nu)*(St**2+Ss**2) # σe² = ν(σθ - σs)²+(1 -ν)(σθ² + σs²)
                        if Se2 >= 0:
                            Se = np.sqrt(Se2)
                        else:
                            print('WARNING: s =',s,'σs =',Ss,'σθ =',St,'ν =',nu,'σe² =',Se2,'-> σe = 0')
                            Se = 0
                        # K = √(2((ν²−ν+1)(σs²+σθ²)+(ν²−4ν+1)σsσθ))
                        K = np.sqrt(2*((nu**2-nu+1)*(Ss**2+St**2)+(nu**2-4*nu+1)*Ss*St))

                        iSel.append(Se)
                        iPSl.append((K*Kt*iVnl[i])/(St-nu*Ss))
                    ''' DONE Compute expected strain rate = f(stress) '''
                    if args.expdotepstar != '':
                        edes.write('Sigma_e\tExpDotEpsStar\n')
                        for i in range(len(iSel)):
                            edes.write(str(iSel[i])+'\t'+str(iPSl[i])+'\n')
                        edes.close()
                        if args.numsteps == 0:
                            sys.exit()

                        LoPSl = False
                        if max(iPSl) < 0.1:
                            iPSl = [ p*1000 for p in iPSl ]
                            LoPSl = True

                        pngout = 'Sim_'+nam+'_ExpectStrainRate_'+str(os.getpid())+'.png'
                        fig = plt.figure(figsize = (8,8))
                        ax = fig.add_subplot(111)
                        for axis in ['top','bottom','left','right']:
                            ax.spines[axis].set_linewidth(4)
                        ax.xaxis.set_tick_params(width=3, length=7)
                        ax.yaxis.set_tick_params(width=3, length=7)
                        plt.subplots_adjust(left = 0.151, right = 0.93, bottom = 0.113, top = 0.892)
                        plt.scatter(iSel, iPSl, marker = 'o', s = 40, color = 'purple')
                        plt.title(nam+'\nExpected strain rate as a function of stress',size=25)
                        plt.tick_params(axis='both', labelsize = 25)
                        if min(iPSl) == max(iPSl):
                            plt.ylim(min(iPSl)*0.95, max(iPSl)*1.05)
                        else:
                            plt.ylim(min(iPSl)-(max(iPSl)-min(iPSl))*0.02, max(iPSl)+(max(iPSl)-min(iPSl))*0.02)
                        plt.xlim(min(iSel)-(max(iSel)-min(iSel))*0.02, max(iSel)+(max(iSel)-min(iSel))*0.02)
                        plt.xlabel(r'$\sigma_e\ (\si{\mega\pascal})$', size = 30)
                        if LoPSl:
                            plt.ylabel(r'$\dot{\epsilon}^*\ (\times 10^{-3}\ \si{\minute^{-1}})$', size = 30)
                        else:
                            plt.ylabel(r'$\dot{\epsilon}^*\ (\si{\minute^{-1}})$', size = 30)
                        plt.axhline(y = 0, color = 'gray', linewidth = 3, zorder = 0)
                        plt.savefig(pngout)
                        plt.close()
    
                # SIMULATE FOR THE CURRENT SET OF PARAMETERS
                # store previous values
                pR = list(R0)
                pX = list(X0)
                ps = list(s0)
                Phi = list(Ph0)
                Kappa_s = list(Ks0)
                Kappa_t = list(Kt0)
                fout.write(nam+'\t'+PhiOut+'\t'+SyStr+'\t'+str(0)+'\t'+str(0)+'\t'+str(0)+'E-3\t'+str(0)+'E-3\t'+str(0)+'E-3\n')
    
                simok = True
                loks = 0
                lokg = 0
                lokk = 0
                lokd = 0
                loke = 0
    
                domx = 0.0
                if args.overhang < 0:
                    i = 0
                    while i < len(Ph0) and -round(np.pi/2,3) < round(Ph0[i],3) < round(np.pi/2,3):
                        i += 1
                    i -= 1
                    domx = X0[0]-X0[i]
                    if args.verbose:
                        print('Dome length = ',domx)
    
                if args.overhang != 0:
                    nextover = args.delta
                    
                deltaX = 0.0
                expover = 0.0
                maxs = int(igro*paradic['secondsbystep'][1]/(paradic['growthbysub'][1]/1000)/60)
                vfac = maxs+1
                subs = 0
                step = 1
                tsec = paradic['secondsbystep'][1] # time in seconds
                if args.verbose:
                    print('Growth = ', igro, 'µm/min,', maxs, 'substeps', flush = True)
                
                while not ( (args.numsteps > 0 and step > args.numsteps) or (args.overhang > 0 and deltaX > args.overhang and (subs == 1 or maxs == 0)) or (domx > 0 and deltaX > domx and (subs == 1 or maxs == 0)) ):
    
                    if simok:
                        loks = step
                    else:
                        if (args.overhang == 0 and step%deltastep == 0) or (args.overhang!= 0 and deltaX > nextover):
                            fout.write(nam+'\t'+PhiOut+'\t'+SyStr+'\t'+str(step)+'\t{0:.5E}\t{1:.5E}\t{2:.5E}\t{3:.5E}\n'.format(lokg,lokk,lokd,loke))
                            nextover += args.delta
                        continue
                    # compute new position for each point
                    nR = []
                    nX = []
                    nptp = len(pR)
                    for i in range(nptp):
                        cwt = para_func(paradic['cwt'],ps[i])
                        vn = modelvn(Kappa_s[i], Kappa_t[i], cwt, paradic['turgor'][1], para_func(ParaPhi, ps[i]), para_func(ParaSy, ps[i]))
                        vn *= (paradic['secondsbystep'][1]/60.0)/vfac
                        vR = np.cos(np.pi/2+Phi[i])*vn
                        nR.append(pR[i]+vR)
                        vX = np.sin(np.pi/2+Phi[i])*vn
                        nX.append(pX[i]+vX)

                    # eliminate points too far from apex   
                    deltaX = nX[0]-X0[0]
                    base = min(X0)+deltaX
                    torm = []
                    for i in range(nptp):
                        if nX[i] < base:
                            torm.insert(0, i)
                    for i in torm:
                        nR.pop(i)
                        nX.pop(i)

                    # add points if necessary
                    addP = []
                    if len(nR) >= 2 and nX[-2] > nX[-1]:
                        while nX[-1] > base:
                            nR.append(nR[-1]+(nR[-1]-nR[-2]))
                            nX.append(nX[-1]+(nX[-1]-nX[-2]))
                            addP.append((nR[-1], nX[-1]))
                        if nX[-1] < base:
                            nR[-1] = nR[-2]+(nR[-1]-nR[-2])* (base-nX[-2])/(nX[-1]-nX[-2])
                            nX[-1] = base
                            if len(addP) > 0:
                                addP.pop(-1)
                            addP.append((nR[-1],nX[-1]))
                    nptn = len(nR)
                    if args.verbose:
                        print('Time = ', s2hhmmss(tsec), str(step)+':', subs, nptn, 'pts', 'deltaX = ', deltaX, 'base = ', base, 'removed', torm, 'added', addP)
                    if nptn < nptp/2 :
                        fout.write(nam+'\t'+PhiOut+'\t'+SyStr+'\t'+str(step)+'\t{0:.5E}\t{1:.5E}\t{2:.5E}\t{3:.5E}\n'.format(lokg,lokk,lokd,loke))
                        simok = False
                        continue
                    
                    '''
                    print('Time = ', str(datetime.timedelta(seconds = tsec)), str(step)+':',subs, (pR[0], pX[0]), '->', (nR[0], nX[0]), max(nX), '->', max(pX), '#', deltaX, '#', (pR[-1], pX[-1]), '->', (nR[-1], nX[-1]), flush = True)
                    '''
                    if args.verbose:
                        print('Time = ', s2hhmmss(tsec), str(step)+':', subs, nptn, 'pts', (nR[0], nX[0]), (nR[1], nX[1]), (nR[2], nX[2]), '...', (nR[-1], nX[-1]), flush = True)
    
                    # compute spline using symmetric set of new points
                    dR = [nR[0]]
                    dX = [nX[0]]
                    i = 1
                    while i < nptn:
                        dR.insert(0, -nR[i])
                        dX.insert(0, nX[i])
                        dR.append(nR[i])
                        dX.append(nX[i])
                        i += 1
                        nptd = len(dR)
                    try:
                        tckspl, uc = interpolate.splprep([dR, dX], k = 3, s = paradic['smooth'][1])  
                        # interpolate on half
                        splstp = 1/(2*npt0)                               # approx. spline step
                        t0 = findR0(tckspl, 0.5, splstp)                  # locate t for which R=0
                        splstp = (1-t0)/(npt0)                            # correct spline step
                        trange = np.append(np.arange(t0, 1, splstp), [1]) # build t range
                        splpts = interpolate.splev(trange, tckspl)        # interpolate
                        npts = len(splpts[0])
                    except SystemError:
                        npts = 0
                   
                    if npts < nptp/2 :
                        fout.write(nam+'\t'+PhiOut+'\t'+SyStr+'\t'+str(step)+'\t{0:.5E}\t{1:.5E}\t{2:.5E}\t{3:.5E}\n'.format(lokg,lokk,lokd,loke))
                        simok = False
                        continue
    
                    # list resampled and rescaled points
                    if args.verbose:
                        print('Time = ', s2hhmmss(tsec), str(step)+':', subs, npts, 'pts', (splpts[0][0], splpts[1][0]), (splpts[0][1], splpts[1][1]), (splpts[0][2], splpts[1][2]), '...', (splpts[0][-1], splpts[1][-1]), flush = True)
                    sR = []
                    sX = []
                    if paradic['smooth'][1] > 0:
                        srg = 5
                        tshrk = 0.0
                        for i in range(srg):
                            tshrk += (splpts[1][i]-nX[i])/srg
                        sxmin = min(splpts[1])
                        sxmax = max(splpts[1])
                        for i in range(npts):
                            sR.append(splpts[0][i])
                            sX.append(splpts[1][i]-tshrk*(splpts[1][i]-sxmin)/(sxmax-sxmin))
                        if args.verbose:
                            splder = interpolate.splev(trange, tckspl, der = 1) # compute derivatives
                            print('Time = ', s2hhmmss(tsec), str(step)+':', subs, npts, 'pts,', 'Tip = ', max(splpts[1]), 'Shrink = ', tshrk, '(dy/dx)(0) = {0:7f}/{1:7f} = {2:7f}'.format(splder[1][0], splder[0][0], splder[1][0]/splder[0][0]), flush = True)
                    else:
                        for i in range(npts):
                            sR.append(splpts[0][i])
                            sX.append(splpts[1][i])
                        if args.verbose:
                            splder = interpolate.splev(trange, tckspl, der = 1) # compute derivatives
                            print('Time = ', s2hhmmss(tsec), str(step)+':', subs, npts, 'pts,', 'Tip = ', max(splpts[1]), '(dy/dx)(0) = {0:7f}/{1:7f} = {2:7f}'.format(splder[1][0], splder[0][0], splder[1][0]/splder[0][0]), flush = True)
                    sR[0] = 0
                    if args.verbose:
                        print('Time = ', s2hhmmss(tsec), str(step)+':', subs, npts, 'pts from', (sR[0], sX[0]), 'to', (sR[-1], sX[-1]))
    
                    # COMPUTE s, φ, κs, κθ
                    nm =  0
                    n0 = -1
                    n1 =  1
                    px0, pxm, px1 = (-sR[n1], sX[n1]), (0.0, sX[nm]), (sR[n1], sX[n1])
                    nR = [pxm[0]] # = 0.0
                    nX = [pxm[1]]
                    merids = 0.0
                    ns = [merids]
                    phi = 0.0
                    Phi = [phi]
                    kappas = curvature(px0, pxm, px1)
                    Kappa_s = [kappas]
                    '''
                    if Kappa_s[0] < 0:
                        print('Time = ', s2hhmmss(tsec), str(step)+':', subs, npts, '!!!', nm, px0, pxm, px1, angle(px0, pxm, px1), 'κ_s = ', Kappa_s[0], flush = True)
                    '''
                    Kappa_t = [kappas] # at tip, κθ = κs
                    n0 += 1
                    nm += 1
                    n1 += 1
                    while n1 < npts:
                        px0, pxm, px1 = (sR[n0], sX[n0]), (sR[nm], sX[nm]), (sR[n1], sX[n1])
                        r, x = sR[nm], sX[nm]
                        nR.append(r)
                        nX.append(x)
                        merids += eucdist(px0, pxm)
                        ns.append(merids)
                        phi -= eucdist(px0, pxm)*kappas
                        Phi.append(phi)
                        kappas = curvature(px0, pxm, px1)
                        Kappa_s.append(kappas)
                        # κθ = 1 / rθ and r = rθ sin(φ)
                        kappat = abs(np.sin(phi))/r
                        Kappa_t.append(kappat)
                        '''
                        if kappas < 0:
                            print('Time = ', str(datetime.timedelta(seconds = tsec)), str(step)+':', subs, npts, '!!!', nm, px0, pxm, px1, angle(px0, pxm, px1), 'κ_s = ', kappas, flush = True)
                        '''
                        n0 += 1
                        nm += 1
                        n1 += 1
    
                    nptn = len(nR)
                    if args.verbose:
                        print('Time = ', s2hhmmss(tsec), step, npt0, 'intitial pts,', nptp, 'previous pts ->', npts, 'spline pts ->', nptn, 'new pts', flush = True)
                    '''
                    print('  Time = ', str(datetime.timedelta(seconds = tsec)), str(step)+':', subs, nptn)
                    '''
                    deltaX = nX[0]-X0[0]
                    '''
                    if args.voluble or args.verbose:
                        print('Time = ', str(datetime.timedelta(seconds = tsec)), str(step)+':', subs, len(R), 'points; R[mid] = ', R[int((len(R)-1)/2)], 'X[0] = ', X[0], 'X[-1] = ', X[-1], 'maxX = ', max(X))
                        print('Time = ', str(datetime.timedelta(seconds = tsec)), str(step)+':', subs, len(nR), 'points; nR[mid] = ', nR[int((len(nR)-1)/2)], 'nX[0] = ', nX[0], 'nX[-1] = ', nX[-1], 'maxnX = ', max(nX))
                    '''
                    
                    if (args.overhang <= 0 and step%deltastep == 0) or (args.overhang != 0 and deltaX > nextover):
                        # Compute residuals
                        nextover += args.delta
                        swd = 0.0
                        swe = 0.0
                        swk = 0.0
                        swn = 0.0
                        i = 0
                        while i < nptn-2 and i < nptr-2:
                            normf = np.exp(-np.log(2)*(ns[i])**2)
                            swd += normf*(nX[i]-(Xr[i]+deltaX))**2+(nR[i]-Rr[i])**2
                            swe += normf*(nX[i]-(Xr[i]+expover))**2+(nR[i]-Rr[i])**2
                            swk += normf*(Kappa_s[i]-Ksr[i])**2
                            swn += normf
                            i += 1
                        resid = np.log(swd/swn)
                        resie = np.log(swe/swn)
                        resik = np.log(swk/swn)
                        if args.verbose:
                            print('Time = '+s2hhmmss(tsec)+'; Growth = {0:.3f} µm; Residuals in {1} points log(rD) = {2:.5E}'.format(deltaX, nptn, resie), flush = True)

                        fout.write(nam+'\t'+PhiOut+'\t'+SyStr+'\t'+str(step)+'\t'+str(deltaX)+'\t{0:.5E}\t{1:.5E}\t{2:.5E}\n'.format(resik,resid,resie))
                        lokg = deltaX
                        lokk = resik
                        lokd = resid
                        loke = resie
                    
                    if args.kinematics and subs == maxs:
                        # Draw global plot
                        XpD = [x+deltaX for x in X0]
                        RpD = [x+deltaX for x in Xr]
                        pngout = 'Global/Sim_'+nam+'_G_p'+PhiStr+'s'+SyStr+'_t{0:06d}.png'.format(step)
                        mR0 = [-r for r in R0]
                        mRr = [-r for r in Rr]
                        mnR = [-r for r in nR]
                        plt.figure(figsize = (8, 8))
                        ax = plt.subplot()
                        for label in ax.get_xticklabels() + ax.get_yticklabels():
                            label.set_fontsize(25)
                        plt.subplots_adjust(left = 0.151, right = 0.93, bottom = 0.113, top = 0.892)
                        if args.Reffile!= '':
                            plt.plot(R0, XpD, lw = 3, color = 'darkgrey', zorder = 1, label = greylab)
                            plt.plot(mR0, XpD, lw = 3, color = 'darkgrey', zorder = 1)
                        plt.plot(Rr, RpD, lw = 3, color = 'purple', zorder = 2, label = pinklab)
                        plt.plot(mRr, RpD, lw = 3, color = 'purple', zorder = 2)
                        plt.plot(R0, X0, lw = 3, color = 'green', zorder = 3, label = 'Initial')
                        plt.plot(mR0, X0, lw = 3, color = 'green', zorder = 3)
                        plt.plot(nR, nX, lw = 3, color = 'blue', zorder = 4, label = 'Model')
                        plt.plot(mnR, nX, lw = 3, color = 'blue', zorder = 4)
                        plt.title(r'$\Phi$ = '+PhiUni+'; $\sigma_y$ = '+SyUni+'\nTime = '+s2hhmmss(tsec)+'; Growth = {0:05.2f} '.format(deltaX)+r'$\si{\micro\metre}$',size=25)
                        plt.xlim(-20, 20)
                        plt.ylim(max(X0)-13, max(X0)+27)
                        plt.xlabel(r'$r\ (\si{\micro\metre})$',size = 30)
                        plt.ylabel(r'$x\ (\si{\micro\metre})$',size = 30)
                        leg = plt.legend(loc = 'lower right',fontsize = 15)
                        leg.draw_frame(False)
                        plt.savefig(pngout)
                        plt.close()

                        # Draw vertical plot
                        pngout = 'Vert/Sim_'+nam+'_V_p'+PhiStr+'s'+SyStr+'_t{0:06d}.png'.format(step)
                        plt.figure(figsize = (4.46, 8))
                        ax = plt.subplot()
                        for label in ax.get_xticklabels() + ax.get_yticklabels():
                            label.set_fontsize(25)
                        plt.subplots_adjust(left = 0.28, right = 0.955, bottom = 0.115, top = 0.865)
                        if args.Reffile!= '':
                            plt.plot(R0, XpD, lw = 3, color = 'darkgrey', zorder = 1)
                            plt.plot(mR0, XpD, lw = 3, color = 'darkgrey', zorder = 1)
                        plt.plot(Rr, XpD, lw = 3, color = 'purple', zorder = 2)
                        plt.plot(mRr, XpD, lw = 3, color = 'purple', zorder = 2)
                        plt.plot(R0, X0, lw = 3, color = 'green', zorder = 3)
                        plt.plot(mR0, X0, lw = 3, color = 'green', zorder = 3)
                        plt.plot(nR, nX, lw = 3, color = 'blue', zorder = 4)
                        plt.plot(mnR, nX, lw = 3, color = 'blue', zorder = 4)
                        plt.title('')
                        plt.gcf().text(0.2, 0.95, 'Time = '+s2hhmmss(tsec), fontsize=25)
                        plt.gcf().text(0.2, 0.90, 'Growth = {0:05.2f} '.format(deltaX)+r'$\si{\micro\metre}$', fontsize=25)
                        plt.xlim(-10, 10)
                        plt.ylim(max(X0)-13, max(X0)+27)
                        plt.xlabel(r'$r\ (\si{\micro\metre})$', size = 30)
                        plt.ylabel(r'$x\ (\si{\micro\metre})$', size = 30)
                        plt.savefig(pngout)
                        plt.close()

                        if args.zoom:
                            # Draw zoom plot
                            pngout = 'Zoom/Sim_'+nam+'_Z_p'+PhiStr+'s'+SyStr+'_t{0:06d}.png'.format(step)
                            Xhi = max(nX)+0.5
                            Xlo = Xhi-12
                            plt.figure(figsize = (8, 8))
                            ax = plt.subplot()
                            for label in ax.get_xticklabels() + ax.get_yticklabels():
                                label.set_fontsize(25)
                            plt.subplots_adjust(left = 0.151, right = 0.93, bottom = 0.113, top = 0.892)
                            if args.Reffile!= '':
                                plt.plot(R0, XpD, lw = 8, color = 'lightgrey', zorder = 1, label = greylab)
                                plt.plot(mR0, XpD, lw = 8, color = 'lightgrey', zorder = 1)
                            plt.plot(Rr, RpD, lw = 8, color = 'plum', zorder = 2, label = pinklab)
                            plt.plot(mRr, RpD, lw = 8, color = 'plum', zorder = 2)
                            plt.plot(nR, nX, lw = 8, color = 'lightblue', zorder = 3, label = 'Model')
                            plt.plot(mnR, nX, lw = 8, color = 'lightblue', zorder = 3)
                            plt.scatter(R0, XpD, marker = 'o', s = 16, color = 'darkgrey', zorder = 4)
                            plt.scatter(mR0, XpD, marker = 'o', s = 16, color = 'darkgrey', zorder = 4)
                            plt.scatter(Rr, RpD, marker = 'o', s = 16, color = 'purple', zorder = 5)
                            plt.scatter(mRr, RpD, marker = 'o', s = 16, color = 'purple', zorder = 5)
                            plt.scatter(nR, nX, marker = 'o', s = 16, color = 'blue', zorder = 6)
                            plt.scatter(mnR, nX, marker = 'o', s = 16, color = 'blue', zorder = 6)
                            plt.title('   '+r'$\Phi$ = '+PhiUni+'; $\sigma_y$ = '+SyUni+'\nTime = '+s2hhmmss(tsec)+'; Growth = {0:05.2f} '.format(deltaX)+r'$\si{\micro\metre}$',size=25)
                            plt.xlim(-6, 6)
                            plt.ylim(Xlo, Xhi)
                            plt.xlabel(r'$r\ (\si{\micro\metre})$', size = 30)
                            plt.ylabel(r'$x\ (\si{\micro\metre})$', size = 30)
                            leg = plt.legend(loc = 'lower center', fontsize = 15)
                            leg.draw_frame(False)
                            plt.savefig(pngout)
                            plt.close()
    
                    # new points become previous points for next loop
                    pR = nR
                    pX = nX
                    ps = ns
    
                    subs+= 1
                    if subs > maxs:
                        step += 1
                        tsec += paradic['secondsbystep'][1]
                        expover += (paradic['expgro'][1]/60.0)*(paradic['secondsbystep'][1]/60.0)
                        subs = 0
    
                if simok:
                    tsec -= paradic['secondsbystep'][1]
                    if args.beforeafter:
                        # draw before / after plot for this data / parameter set
                        XpD = [x+expover for x in X0]
                        RpD = [x+expover for x in Xr]
                        mnR = [-r for r in nR]
                        mR0 = [-r for r in R0]
                        mRr = [-r for r in Rr]
                        prrg = max(nR+R0)-min(mnR+mR0)
                        pxrg = max(nX+X0+RpD)-min(nX+X0+RpD)
                        pfrg = 1.1*max(prrg, pxrg)
                        pxmd = min(mnR+mR0) + prrg / 2
                        pymd = min(nX+X0) + pxrg / 2
                        
                        plt.figure(figsize = (8, 8))
                        if args.Reffile!= '':
                            plt.scatter(Rr, RpD, marker = 'o', s = 16, color = 'darkgrey', zorder = 1, label = 'Reference')
                            plt.scatter(mRr, RpD, marker = 'o', s = 16, color = 'darkgrey', zorder = 1)
                        plt.scatter(R0, XpD, marker = 'o', s = 16, color = 'purple', zorder = 2, label = 'Expected')
                        plt.scatter(mR0, XpD, marker = 'o', s = 16, color = 'purple', zorder = 2)
                        plt.scatter(R0, X0, marker = 'o', s = 16, color = 'green', zorder = 3, label = 'Initial')
                        plt.scatter(mR0, X0, marker = 'o', s = 16, color = 'green', zorder = 3)
                        plt.scatter(nR, nX, marker = 'o', s = 16, color = 'blue', zorder = 4, label = 'Final')
                        plt.scatter(mnR, nX, marker = 'o', s = 16, color = 'blue', zorder = 4)
                        i=0
                        line = plt.plot([nR[i], R0[i]], [nX[i], XpD[i]], linewidth = 2, color = 'plum', zorder = 0, label = 'Residuals')
                        i=5
                        while i < nptn:
                            line = plt.plot([nR[i], R0[i]], [nX[i], XpD[i]], linewidth = 2, color = 'plum', zorder = 0)
                            line = plt.plot([mnR[i], mR0[i]], [nX[i], XpD[i]], linewidth = 2, color = 'plum', zorder = 0)
                            i += 5
                        plt.tick_params(axis='both', labelsize = 15)
                        plt.title(r'$\Phi$ = '+PhiUni+'; $\sigma_y$ = '+SyUni+'; Time = '+s2hhmmss(tsec))
                        plt.xlim(pxmd-pfrg/2, pxmd+pfrg/2)
                        plt.ylim(pymd-pfrg/2, pymd+pfrg/2)
                        plt.xlabel(r'$r\ (\si{\micro\metre})$', size = 15)
                        plt.ylabel(r'$x\ (\si{\micro\metre})$', size = 15)
                        leg = plt.legend(loc = 'center right', fontsize = 15)
                        leg.draw_frame(False)
                        pdfout.savefig()

                        plt.figure(figsize = (8, 8))
                        ms0 = [-s for s in s0]
                        msr = [-s for s in sr]
                        mns = [-s for s in ns]
                        if args.Reffile!= '':
                            plt.plot(sr, Ksr, linewidth = 4, color = 'darkgrey', zorder = 1, label = 'Reference')
                            plt.plot(msr, Ksr, linewidth = 4, color = 'darkgrey', zorder = 1)
                        plt.plot(s0, Ks0, linewidth = 4, color = 'green', zorder = 3, label = 'Initial')
                        plt.plot(ms0, Ks0, linewidth = 4, color = 'green', zorder = 3)
                        plt.plot(ns, Kappa_s, linewidth = 4, color = 'blue', zorder = 4, label = 'Final')
                        plt.plot(mns, Kappa_s, linewidth = 4, color = 'blue', zorder = 4)
                        plt.tick_params(axis='both', labelsize = 15)
                        plt.title(r'$\Phi$ = '+PhiUni+'; $\sigma_y$ = '+SyUni+'; Time = '+s2hhmmss(tsec))
                        plt.xlabel(r'$s\ (\si{\micro\metre})$', size = 15)
                        plt.ylabel(r'$\kappa_s\ (\si{\micro\metre^{-1}})$', size = 15)
                        leg = plt.legend(loc = 'center right', fontsize = 15)
                        leg.draw_frame(False)
                        plt.axhline(y = 0, color = 'gray', linewidth = 2, zorder = 0)
                        plt.axvline(x = 0, color = 'gray', linewidth = 2, zorder = 0)
                        pdfout.savefig()

                    if args.voluble or args.verbose:
                        print('Time = '+s2hhmmss(tsec)+'; Growth = {0:.3f} µm; Residuals in {1} points log(rD) = {2:.5E}'.format(deltaX, nptn, resie), flush = True)
                    lRdl.append(resie)
                else:
                    if args.voluble or args.verbose:
                        print('Time = '+s2hhmmss(tsec)+'; Growth = {0:.3f} µm; Residuals in {1} points log(rD) = {2:.5E} ### WARNING: incomplete simulation: {4} min ###'.format(lokg, len(nX), lokd, loks), flush = True)
                    lRdl.append(lokd)
                fout.flush()

            lRda.append(lRdl)
        
        fout.close()
        if args.beforeafter:
            pdfout.close()

        if len(phlst)>1 and len(sylst)>1:
            # DARW HEATMAP
            istep = paradic['jfory'][1]
            jstep = paradic['iforphi'][1]
            pngout = 'HeatMap_{0}_Map.png'.format(os.getpid())
            fig = plt.figure(figsize = (12, 12))
            ax = fig.add_subplot(111)
            for axis in ['top','bottom','left','right']:
              ax.spines[axis].set_linewidth(4)
            ax.xaxis.set_tick_params(width=4, length=10)
            ax.yaxis.set_tick_params(width=4, length=10)
            plt.subplots_adjust(bottom = 0.14, left = 0.20, top = 0.90, right = 0.96)
            plt.title(nam+' simulation', size = 56, y=1.05)
            lrmi,lrma=min(map(min,lRda)),max(map(max,lRda))
            mynorm=matplotlib.colors.Normalize(vmin=lrmi, vmax=lrma, clip=False)
            #mycm=cm.ScalarMappable(norm=mynorm,cmap='gnuplot2') # 'hot' and 'gnuplot2' remain OK
            mycm=cm.ScalarMappable(norm=mynorm,cmap='hot')       # when changed to greyscale
            j=0
            for p in phlst:
                i=0
                for s in sylst:
                    rect = matplotlib.patches.Rectangle( (s[1]-istep/2,p[1]-jstep/2),width=istep, height=jstep, color=mycm.to_rgba(lRda[j][i]))
                    ax.add_patch(rect)
                    i+=1
                j+=1
            plt.tick_params(axis='both', labelsize = 36, direction='out')
            plt.xlim(sylst[0][1]-istep/2,sylst[-1][1]+istep/2)
            lsyl = len(sylst)
            smid = int((lsyl-1)/2)
            sstep = 1
            if lsyl>7:
                ntick = 7
                if (lsyl-1)/4-int((lsyl-1)/4)<(lsyl-1)/6-int((lsyl-1)/6):
                    ntick = 5
                sstep = int((lsyl-1)/(ntick-1))
            tl=[sylst[smid][1]]
            nl=[round(sylst[smid][1],5)]
            si = sstep
            while smid+si < len(sylst):
                tl.append(sylst[smid+si][1])
                nl.append(round(sylst[smid+si][1],5))
                tl.insert(0,sylst[smid-si][1])
                nl.insert(0,round(sylst[smid-si][1],5))
                si += sstep
            plt.xticks(tl)
            plt.xlabel(r'$\sigma_y\ (\si{\mega\pascal})$', size = 52)
            plt.ylim(phlst[0][1]-jstep/2,phlst[-1][1]+jstep/2)
            lphl = len(phlst)
            pmid = int((lphl-1)/2)
            pstep = 1
            if lphl>7:
                ntick = 7
                if (lphl-1)/4-int((lphl-1)/4)<(lphl-1)/6-int((lphl-1)/6):
                    ntick = 5
                pstep = int((lphl-1)/(ntick-1))
            if ParaPhi[1]<=1E-1:
                tl=[phlst[pmid][1]]
                nl=[round(phlst[pmid][1]*1000,5)]
                pi = pstep
                while pmid+pi < len(phlst):
                    tl.append(phlst[pmid+pi][1])
                    nl.append(round(phlst[pmid+pi][1]*1000,5))
                    tl.insert(0,phlst[pmid-pi][1])
                    nl.insert(0,round(phlst[pmid-pi][1]*1000,5))
                    pi += pstep
                plt.yticks(tl,nl)
                plt.ylabel(r'$\Phi\ (\times 10^{-3}\ \si{\minute^{-1}\cdot \mega\pascal^{-1}})$', size = 52)
            else:
                tl=[phlst[pmid][1]]
                nl=[round(phlst[pmid][1],5)]
                pi = pstep
                while pmid+pi < len(phlst):
                    tl.append(phlst[pmid+pi][1])
                    nl.append(round(phlst[pmid+pi][1],5))
                    tl.insert(0,phlst[pmid-pi][1])
                    nl.insert(0,round(phlst[pmid-pi][1],5))
                    pi += pstep
                plt.yticks(tl)
                plt.ylabel(r'$\Phi\ (\si{\minute^{-1}\cdot \mega\pascal^{-1}})$', size = 52)
            ax.figure.canvas.draw()
            plt.savefig(pngout)
            plt.close()

            pngout = 'HeatMap_{0}_Scale.png'.format(os.getpid())
            fig = plt.figure(figsize = (4, 12))
            ax = fig.add_subplot(111)
            for axis in ['top','bottom','left','right']:
              ax.spines[axis].set_linewidth(4)
            plt.tick_params(axis='both', labelsize = 40)
            ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
            ax.tick_params(axis='y',which='both',left='off',right='on',labelleft='off',labelright='on')
            ax.yaxis.set_tick_params(width=4, length=10)
            plt.subplots_adjust(bottom = 0.14, left = 0.30, top = 0.90, right = 0.60)
            plt.title('Color\nscale', size = 40, y=1.00)
            srg=(lrma-lrmi)/200
            for s in np.arange(lrmi,lrma+srg,srg):
                rect = matplotlib.patches.Rectangle( (0,s),width=1, height=srg, color=mycm.to_rgba(s))
                ax.add_patch(rect)
            plt.xlim(0,1)
            plt.ylim(lrmi,lrma)
            plt.xlabel('log(rD)', size = 52)
            plt.savefig(pngout)
            plt.close()
            
    if args.voluble or args.verbose:
        print('Done')

if __name__ == '__main__':
    
    main()

sys.exit(0)

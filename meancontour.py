#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
Compute the mean contour(s) of one or more series of contours.
Input: A file containing the list of contour images with some options,
    then the image files themselves.
Output: A pdf file with each contour analyzed and figures for the mean
    contour, and a tab-delimited file with data for each mean contour
    point.
'''

'''
© Bernard Billoud / Morphogenesis of Macro-Algae team (CNRS/Sornonnr Université)

Bernard.Billoud@upmc.fr

This software is a computer program whose purpose is to compute a
mean contour from a collection of apical cell contours.

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
import webcolors
from PIL import Image
from scipy import interpolate
from scipy.optimize import curve_fit

def mkparadic(t,sdict):
    '''
    make arguments dictionary from a previous dictionary
    :param sdict: the already existing dictionary
    :type sdict: dictionary
    :param t: a table of strings, alternatively parameter
              name and value
    :type t: list of character strings
    '''
    na = len(t)
    adict = sdict.copy()
    for i in range(0,na,2):
        if t[i] in adict:
            typ = type(adict[t[i]])
            adict[t[i]] = typ(t[i+1])
    return adict

def simplifycolor(col):
    '''
    simplify colors to a 4x4x4x2 space, where each color component is
    set to the closest value among [0,85,170,255] and opacity is
    either 0 or 255.
    :param col: the color to simplify
    :type col: a tuple, expected to be of length 4.
    :return: the simplified color
    :rtype: tuple of length 4 (r,g,b,o)

    .. note:: If length of col is > 4, then only 4 first components are
              used; if length of col is < 4, then missing components are
              set to 255.
    '''
    sco = []
    for i in range(3): # color components: r, g, b
        if len(col)>= i:
            sco.append(int((col[i]+42)/85)*85)
        else:
            sco.append(255)
    if len(col)>= 4: # opacity component
        sco.append(int(col[3]/128)*255)
    else:
        sco.append(255)
    return tuple(sco)

def buildpath(path0,prvd,dirl,extl,ctxy):
    '''
    Recursively build a path.

    :param path0: the state of the path when entering the function
    :type path0: list of pixels, each pixel is a 2-uple
    :param prvd: previous direction, first to test for the next step
    :type prvd: int in the range [0:7]
    :param dirl: directions: 0 = south, 1 = south-east, etc.
    :type dirl: list of 2-uples
    :param extl: potential path extremities
    :type extl: list of pixels, each pixel is a 2-uple
    :param ctxy: contour points
    :type extl: list of pixels, each pixel is a 2-uple
    :return: an empty path, or the completed path
    :rtype: list of pixels, each pixel is a 2-uple

    .. note:: In order to try to minimize the path length, make use of
              ddir list to test prvd first, then 1 step clockwise, 1
              step anti-clockwise, etc. and keep the first full path.
    .. todo:: Choose the shortest path instead of the first one.
    '''

    debug=False
    
    if debug:
        print('<',len(path0),'>',prvd,end = ' ')

    if len(path0) == 0: # dead end, retro-propagate
        path1 = path0
    else:
        lastp = path0[-1]
        if lastp in extl[1:]: # path end reached, retro-propagate
            path1 = path0
            
            if debug:
                print(path1,'END')
                
        else:
            tstn = 0
            ddir = [0,-1,1,-2,2,-3,3,4]
            maxt = len(ddir)
            path1 = []
            while ( tstn < maxt and
                    (len(path1) == 0 or path1[-1] not in extl[1:])
                  ):
                ndir = (prvd+ddir[tstn]+8)%8
                stpd = dirl[ndir]
                newp = (lastp[0]+stpd[0],lastp[1]+stpd[1])
                
                if debug:
                    print(' +',newp,end = '')
                    
                if newp in path0 or newp not in ctxy:
                    
                    if debug:
                        print(' |',end = ' ')
                        
                    path1 = [] # this is a dead end
                else:
                    
                    if debug:
                        print(' ~',end = ' ')

                    path1 = path0
                    path1.append(newp)

                    if debug:
                        print('->',ndir)

                    path1 = buildpath(path1,ndir,dirl,extl,ctxy)
                tstn+= 1
    if debug:
        if len(path1)>0:
            print(' [',path1[0],'-',path1[-1],']')
        else:
            print(' [ X ]')

    return path1

def sigmo(x, x0, alpha, delta, y0):
    '''
    Sigmoïd function
    :param x: Value for which the function is computed
    :type x: float
    :param x0: Center of sigmoïd, x coordinate
    :type x0: float
    :param alpha: Slope
    :type alpha: float
    :param delta: Amplitude
    :type delta: float
    :param y0: Center of sigmoïd, y coordinate
    :type delta: float

    :return: Value of the sigmoïd function at x
    :rtype: float
    '''
    return y0 + delta / (1 + np.exp(-(x-x0)/alpha))

################# MAIN #################

# MANAGE COMMAND-LINE ARGUMENTS

parser = argparse.ArgumentParser()

parser.add_argument('-c','--contour', type = str,
    help = 'contour color.',default = '#ff00aa')
parser.add_argument('-t','--tip', type = str,
    help = 'tip color.',default = '#55aaaa')
parser.add_argument('-s', '--smooth', type = float,
    help = 'smoothing of contour spline.', default = 1.5)
parser.add_argument('-w', '--window', type = float,
    help = 'window width in nm.', default = 250)
parser.add_argument('-d', '--density', type = float,
    help = 'density of intrapolated points (points/µm).', default = 5)
parser.add_argument('-r', '--range',type = float,
    help = 'meridional abscissa range',default = 30.0)
parser.add_argument('-v', '--verbose',action = 'store_true',
    help = 'write to stdout while working.')
parser.add_argument('-p', '--pdf',action = 'store_true',
    help = 'write to pdf.')
parser.add_argument('-D','--depth', type = int,
    help = 'maximum depth of stack.',default = 5000)
parser.add_argument('infile', type = str,
    help = 'path to input file.')

args = parser.parse_args()

if args.verbose:
    print('\nParameters:')
    for arg, value in sorted(vars(args).items()):
        print(' ',arg,'=',value)
    print()

sys.setrecursionlimit(args.depth)

sct = simplifycolor(webcolors.hex_to_rgb(args.contour))
stp = simplifycolor(webcolors.hex_to_rgb(args.tip))

try:
    instr = open(args.infile,'r')
except IOError:
    print('Cannot open',args.infile)
    sys.exit(1)
    
# READ INPUT FILE AND BUILD DATASETS

if args.verbose:
    print('Reading from '+args.infile+'\n')

datasetl = [] # keep dataset order
datasetdic = {}
datasetpar = {}
ln = 0

for line in instr:
    ln += 1
    tabl = line[:-1].split()
    if len(tabl) > 0:
        if tabl[0][0] == '#': # a comment
            continue
        elif tabl[0][0] == '*': # a dataset
            dataset = tabl[1]
            datasetl.append(dataset)
            datasetdic[dataset] = []
            datasetpar[dataset] = mkparadic(tabl[2:],vars(args))
        else: # an image in the current dataset
            if len(tabl)<2: # expect image file name and width in µm
                print('Warning: bad format line',ln)
            else:
                imgfile = tabl[0]
                imwd = float(tabl[1])
                para = mkparadic(tabl[2:],datasetpar[dataset])
                datasetdic[dataset].append((imgfile,imwd,para))

instr.close()

# PREPARE TAB OUTPUT FILE
outfile = args.infile.split('.')[0]+'_contour.tab'
try:
    tabout = open(outfile,'w')
except IOError:
    print('Cannot open output file',outfile)
    sys.exit(1)
tabout.write('Set\tR\tX\ts\tPhi\tKappa_s\tKappa_t\tNu\n')
    
# PROCESS IMAGE FILES FOR EACH DATASET

for dataset in datasetl:
    
    if args.verbose:
        print(' Processing images for dataset '+dataset+'\n')

    pc = False
    dpar = datasetpar[dataset]
    for arg in dpar:
        if dpar[arg] != vars(args)[arg]:
            if not pc :
                if args.verbose:
                    print(' Dataset-specific parameters:')
                pc = True
            if args.verbose:
                print(' *',arg,'=',dpar[arg])

    if args.pdf:
        # PREPARE PDF
        plotfile = dataset+'_plots.pdf'
        pdfout = PdfPages(plotfile)    

    # PREPARE S AND KAPPA_S DATASET LISTS
    dssl = [] # dataset s list
    dskl = [] # dataset kappa_s list

    for image in datasetdic[dataset]:

        try:
        
            # READ IMAGE FILE
            imgfile = image[0]
            imgbase = imgfile.split('/')[-1]
            img = Image.open(imgfile)
            (imw,imh) = img.size
            pxw = image[1]
            mpp = pxw/imw
            if dpar['verbose']:
                print('  ',imgfile,': ',imw,' x ',imh,';',
                      ' one pixel = ',round(mpp,4),' µm',sep = '')
            pixels = img.load()

            pc = False
            ipar = image[2]
            for arg in ipar:
                if ipar[arg] != dpar[arg]:
                    if not pc :
                        if dpar['verbose']:
                            print('   Image-specific parameters:')
                        pc = True
                    if dpar['verbose']:
                        print('   *',arg,'=',ipar[arg])


            # SET LIST OF CONTOUR PIXELS
            contxy = []
            tipxy = []
            for y in range(imh):
                for x in range(imw):
                    pxc = pixels[x,y]
                    spx = simplifycolor(pxc)

                    # uncomment to produce simplified image
                    '''
                    pixels[x,y] = spx
                    '''
                    if spx == sct:
                        contxy.append((x,y))
                    elif spx == stp:
                        contxy.append((x,y))
                        tipxy.append((x,y))
                        if ipar['verbose']:
                            print('   Tip drawn at',(x,y))
            # uncomment to see (simplified) image
            '''
            img.show()
            '''
            if ipar['verbose']:
                print('  ',len(contxy),'contour pixels;',end = ' ')

            # REMOVE ISOLATED CONTOUR PIXELS FROM LIST
            # AND FIND POTENTIAL PATH ENDS
            extl = []
            torm = []
            nb0n = 0
            for ppx in contxy:
                nbl = []
                for dy in range(-1,2):
                    for dx in range(-1,2):
                        if dx != 0 or dy != 0:
                            if (ppx[0]+dx,ppx[1]+dy) in contxy:
                                nbl.append((ppx[0]+dx,ppx[1]+dy))
                if len(nbl) == 0:
                    torm.append(ppx)
                    nb0n += 1
                elif len(nbl) == 1:
                    extl.append(ppx)
            for ppx in torm:
                contxy.remove(ppx)
            if ipar['verbose']:
                print('  ',nb0n,'isolated pixels removed')
                print('  ',len(extl),'path ends:',extl)

            if len(extl) != 2:
                print('   Cannot build contour path for',imgfile)
                continue

            # BUILD A CONTOUR PATH
            dirl = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
            pdir = 0
            if len(extl)>0:
                pathlst = [extl[0]]
                pathlst = buildpath(pathlst,pdir,dirl,extl,contxy)
            '''
            if ipar['verbose']:
                print('   Initial path uses',len(pathlst),'pixels')
            '''

            # SIMPLIFIY CONTOUR
            torm = []
            for ppx in pathlst:
                rem = False
                for dx in [-1,1]:
                    if(ppx[0]+dx,ppx[1]) in pathlst:
                        for dy in [-1,1]:
                            if(ppx[0],ppx[1]+dy) in pathlst:
                                rem = True
                if rem:
                    torm.append(ppx)
            for ppx in torm:
                pathlst.remove(ppx)

            if ipar['verbose']:
                print('   Path uses',len(pathlst),'pixels')

            if len(pathlst) < 2:
                print('   Cannot build contour path for',imgfile)
                continue
            
            '''
            for px in contxy:
                if px not in pathlst:
                    (x,y) = px
                    if px in torm:
                        pixels[x,y] = (85,255,85)
                    else:
                        pixels[x,y] = (255,85,85)
            img.show()
            '''
            # ORDER pixels
            mid = int(len(pathlst)/2)
            if(angle(pathlst[0],pathlst[mid],pathlst[-1]))<0:
                j = -1
                for i in range(mid+1):
                    (x,y) = pathlst[i]
                    pathlst[i] = pathlst[j]
                    pathlst[j] = (x,y)
                    j -= 1
                    
            # INTERPOLATE
            pathx = []
            pathy = []
            for pt0 in pathlst :
                pathx.append(pt0[0]*mpp) # x coordinate in µm
                pathy.append(pt0[1]*mpp) # y coordinate in µm
            ctrspl,uc = interpolate.splprep([pathx,pathy],
                                            s = ipar['smooth'])
            splstp = 1/((len(pathlst)+2)*mpp*ipar['density'])  # spline step
            trange = np.arange(0, 1, splstp)            # build t range
            tckspl = interpolate.splev(trange, ctrspl) # interpolate
            lpw = len(tckspl[0])
            if ipar['verbose']:
                print('   Interpolated to',lpw,'spline points')

            if len(tipxy) == 1:
                # FIND APEX ASSUMING NORMAL TO WALL FOLLOWS A SIGMOID FUNCTION
                # compute normal to axis
                tipX = tipxy[0][0]*mpp
                tipY = tipxy[0][1]*mpp
                pt0 = (tckspl[0][0],tckspl[1][0])
                apex = pt0
                soff = 0
                m2tip = eucdist(pt0,[tipxy[0][0]*mpp,tipxy[0][1]*mpp])
                slopx = 0
                sumd = 0     # cumulated length along path in µm
                distl = []   # list of successive values of sumd
                splptl = []  # spline points in the form (x,y), in µm
                splxpt = []  # spline points x coordinates in µm
                splypt = []  # spline points y coordinates in µm
                sloplx = []
                for n in range(lpw):
                    pt1 = (tckspl[0][n],tckspl[1][n])
                    splptl.append(pt1)
                    splxpt.append(pt1[0])
                    splypt.append(pt1[1])
                    sumd += eucdist(pt0,pt1)
                    if pt1[1] != pt0[1]:
                        slope = (pt1[0]-pt0[0])/(pt1[1]-pt0[1])
                        sloplx.append(-1/slope)
                    d2tip = eucdist(pt1,[tipxy[0][0]*mpp,tipxy[0][1]*mpp])
                    if d2tip < m2tip:
                        apex = pt1
                        m2tip = d2tip
                        soff = sumd
                        slopx = -1/slope
                    distl.append(sumd)
                    pt0 = pt1
                    '''
                    print(sumd)
                    '''
                # build phi = f(s)
                if ipar['verbose']:
                    print('   Normal direction at tip: y =',round(1.0/slopx,2),'x')
                A = midpoint(splptl[0],splptl[-1])
                B = splptl[int(lpw/2)]
                M = (A[0]+1,A[1]-slopx)
                if angle(M,A,B)>0:
                    dP = (1,-slopx)
                else:
                    dP = (-1,slopx)
                pt0 = splptl[0]
                n = 1
                angl = []
                while n<lpw:
                    pt1 = splptl[n]
                    pt2 = (pt0[0]+dP[0],pt0[1]+dP[1])
                    angl.append(angle(pt2,pt0,pt1))
                    pt0 = pt1
                    n+= 1
                cdir = 1.0
                if angl[0] < angl[-1]:
                    print('  ## WARNING',imgfile,'CDIR = -1')
                    cdir = -1.0
                    la = len(angl)
                    for i in range(la):
                        angl[i] = -angl[i]
                alph = 0.0
                npt = 0
                for n in range(lpw-1):
                    if -np.pi/4<angl[n]<np.pi/4 and angl[n]!=0.0:
                        alph += -(distl[n+1]-soff)/np.log(np.pi/(angl[n]+np.pi/2)-1)
                        npt += 1
                alph = alph / npt
                sigpar=[soff,alph,np.pi,-np.pi/2]
                if ipar['verbose']:
                    print('   Slope for sigmoid =',round(alph,3))
                    print('   Tip drawn at (',
                      round(tipxy[0][0]*mpp,2),', ',
                      round(tipxy[0][1]*mpp,2),'),',
                      ' approximated to (',
                      round(apex[0],2),', ',
                      round(apex[1],2),')',sep = '')
                    print('   Meridional abscissa range from',
                      round(-soff,2),'µm to',
                      round(distl[-1]-soff,2),'µm')

            else:
                # FIND APEX ASSUMING NORMAL TO WALL FOLLOWS A SIGMOID FUNCTION
                # compute normal to axis
                pt0 = (tckspl[0][0],tckspl[1][0])
                sumd = 0     # cumulated length along path in µm
                distl = []   # list of successive values of sumd
                splptl = []  # spline points in the form (x,y), in µm
                splxpt = []  # spline points x coordinates in µm
                splypt = []  # spline points y coordinates in µm
                sloplx = []
                for n in range(lpw):
                    pt1 = (tckspl[0][n],tckspl[1][n])
                    splptl.append(pt1)
                    splxpt.append(pt1[0])
                    splypt.append(pt1[1])
                    sumd += eucdist(pt0,pt1)
                    if pt1[1] != pt0[1]:
                        sloplx.append((pt1[0]-pt0[0])/(pt1[1]-pt0[1]))
                    distl.append(sumd)
                    pt0 = pt1
                    '''
                    print(sumd)
                    '''
                # build phi = f(s)
                slopx = np.median(sloplx)
                if ipar['verbose']:
                    print('   Main direction: y =',round(1.0/slopx,2),'x')
                A = midpoint(splptl[0],splptl[-1])
                B = splptl[int(lpw/2)]
                M = (A[0]+1,A[1]-slopx)
                if angle(M,A,B)>0:
                    dP = (1,-slopx)
                else:
                    dP = (-1,slopx)
                pt0 = splptl[0]
                n = 1
                angl = []
                while n<lpw:
                    pt1 = splptl[n]
                    pt2 = (pt0[0]+dP[0],pt0[1]+dP[1])
                    angl.append(angle(pt2,pt0,pt1))
                    pt0 = pt1
                    n+= 1
                cdir = 1.0
                if angl[0] < angl[-1]:
                    print('  ## WARNING',imgfile,'CDIR = -1')
                    cdir = -1.0
                    la = len(angl)
                    for i in range(la):
                        angl[i] = -angl[i]
                # fit sigmoid and find x0, etc.
                sigpar, covar = curve_fit(sigmo, distl[:-1], angl,
                         p0 = [(distl[0]+distl[-1])/2,-2,np.pi,-np.pi/2])
                '''
                print(sigpar)
                '''
                soff = sigpar[0]
                # compute apex coordinates
                tval = (soff-distl[0])/(distl[-1]-distl[0])
                apxa = interpolate.splev(tval,ctrspl)
                apex = (apxa[0].item(0),apxa[1].item(0))
                if ipar['verbose']:
                    print('   Tip estimated at (',
                      round(apex[0],1),', ',
                      round(apex[1],1),')',sep = '')
                    print('   Meridional abscissa range from',
                      round(-soff,2),'µm to',
                      round(distl[-1]-soff,2),'µm')

            # COMPUTE KAPPA_S
            meridsl = []
            kappasl = []
            sigmol = []
            n0,nm,n1 = 0,1,2
            while n1<lpw:
                px0,pxm,px1 = splptl[n0],splptl[nm],splptl[n1]
                merids = distl[nm]-soff
                meridsl.append(merids)
                kappas = curvature(px0,pxm,px1)
                kappasl.append(cdir*kappas)
                sigmol.append(sigmo(distl[nm],sigpar[0],sigpar[1],
                                              sigpar[2],sigpar[3]))
                '''
                print(merids,kappas)
                '''
                n0 += 1
                nm += 1
                n1 += 1

            if args.pdf:
                if ipar['verbose']:
                    print('   Plotting φ')
                # PLOT phi AND SIGMOID PARAMETERS
                plt.figure(figsize = (8,6))
                plt.plot(distl[1:],angl,color = 'purple',lw = 2)
                plt.title('Normal to contour - Axis angle for '+imgbase)
                plt.axvline(x = sigpar[0],color = 'gray',zorder = 0)
                plt.axhline(y = sigmo(sigpar[0],sigpar[0],sigpar[1],sigpar[2],
                                       sigpar[3]),color = 'gray',zorder = 0)
                plt.plot(meridsl+soff,sigmol,color = 'gray',zorder = 0)
                plt.xlim(0,distl[-2])
                plt.yticks(np.arange(-np.pi/2, np.pi, np.pi/2),
                                 [r'$-\pi$/2','0',r'+$\pi$/2'])
                plt.xlabel('l ('+r'$\mu$'+'m)')
                plt.ylabel(r'$\varphi$'+' (rad)')
                pdfout.savefig()
                plt.close()

                if ipar['verbose']:
                    print('   Drawing contour')
                # DRAW CONTOUR
                xrg = max(pathx)-min(pathx)
                yrg = max(pathy)-min(pathy)
                frg = 1.1*max(xrg,yrg)
                xmd = min(pathx) + xrg / 2
                ymd = min(pathy) + yrg / 2
                plt.figure(figsize = (8.0*181.0/175.0,8))
                plt.scatter(splxpt,splypt,marker = 'o',s = 7,color = 'red')
                #plt.scatter(pathx,pathy,marker = '.',s = 1,
                #            color = ipar['contour'])
                plt.scatter([apex[0]],[apex[1]],marker = '+',s = 7,color = 'blue')
                plt.title('Contour for '+imgbase+', smooth = '+str(ipar['smooth']))
                plt.xlim(xmd-frg/2,xmd+frg/2)
                plt.ylim(ymd+frg/2,ymd-frg/2)
                plt.xlabel('x ('+r'$\mu$'+'m)')
                plt.ylabel('y ('+r'$\mu$'+'m)')
                pdfout.savefig()
                plt.close()

                # COMPUTE DRAWABLE KAPPA_S
                dpt = int(len(meridsl)*(dpar['window']/2000)/(meridsl[-1]-meridsl[0]))
                if dpt<1:
                    dpt=1
                if dpt>500:
                    dpt=500
                dmeridsl = []
                dkappasl = []
                n0,nm,n1 = 0,dpt,2*dpt
                while n1<lpw:
                    px0,pxm,px1 = splptl[n0],splptl[nm],splptl[n1]
                    merids = distl[nm]-soff
                    dmeridsl.append(merids)
                    kappas = curvature(px0,pxm,px1)
                    dkappasl.append(cdir*kappas)
                    '''
                    print(dmerids,dkappas)
                    '''
                    n0 += 1
                    nm += 1
                    n1 += 1

                if ipar['verbose']:
                    print('   Plotting κ_s')
                # PLOT KAPPA_S
                xmax = max(abs(min(meridsl)),abs(max(meridsl)))
                plt.figure(figsize = (8,4.5))
                plt.plot(meridsl,kappasl,color = '#5555AA',lw = 1)
                plt.plot(dmeridsl,dkappasl,color = 'blue',lw = 2)
                plt.title('Meridional curvature for '+imgbase)
                plt.xlim(-xmax,xmax)
                plt.axhline(color = 'gray',zorder = 0)
                plt.axvline(color = 'gray',zorder = 0)
                plt.xlabel('s ('+r'$\mu$'+'m)')
                plt.ylabel(r'$\kappa_s$ ('+r'$\mu$'+'m'+r'$^{-1}$'+')')
                pdfout.savefig()
                plt.close()

            # STORE S AND KAPPA_S IN DATASET LIST
            for s in meridsl:
                dssl.append(s)
            for k in kappasl:
                dskl.append(k)

            # img.close() # do not close as load() has been performed(?)

            if ipar['verbose']:
                print('  Done with '+imgfile+'\n')
                
            # END OF PROCESS FOR ONE IMAGE
            
        except IOError:
            print('Cannot open',imgfile)
            sys.exit(2)
        
    # COMPUTE AVERAGE KAPPA_S
    if len(dssl) == 0:
        print(' Empty dataset, no output for',dataset)
        pdfout.close()
        continue

    if dpar['verbose']:
        print('  Averaging and symmetrizing κ_s for dataset '+dataset)
    smax = max(abs(min(dssl)),abs(max(dssl)))
    if smax<dpar['range'] and dpar['verbose']:
            print('  Warning: range too wide. Data available for',smax,'µm')
    elif 0<dpar['range']<smax:
        smax = dpar['range']
    ws = 1/dpar['density'] # window step
    ww = dpar['window']/2000

    n = 0
    kl = []
    i = 0
    for s in dssl:
        if -ww <= s <= ww:
            kl.append(dskl[i])
            n += 1
        i += 1
    msl = [0.0] # mean s
    mk = np.mean(kl)
    sk = np.std(kl)
    mkl = [mk] # mean kappa_l
    dkl = [mk-sk] # mean - sd
    ukl = [mk+sk] # mean + sd
    nmax = n
    nmin = n
    sumn = n
    
    for s0 in np.arange(ws,smax-ww+ws,ws):
        n = 0
        sns = 0.0
        sps = 0.0
        kl = []
        i = 0
        for s in dssl:
            if -s0-ww <= s <= -s0+ww:
                sns += s
                kl.append(dskl[i])
                n += 1
            if s0-ww <= s <= s0+ww:
                sps += s
                kl.append(dskl[i])
                n += 1
            i += 1
        if n>0:
            ms = (abs(sns)+abs(sps))/n
            if ms != msl[-1]:
                msl.append(ms)
                mk = np.mean(kl)
                sk = np.std(kl)
                mkl.append(mk)
                dkl.append(mk-sk)
                ukl.append(mk+sk)
                if n > nmax:
                    nmax = n
                if n < nmin:
                    nmin = n
                sumn += n
    if dpar['verbose']:
        print(' ',len(msl),'points with |s| <',round(msl[-1],2))
        print('  Average of',nmin,'to',nmax,
                     'points; mean =',round(2*sumn/len(msl),2),'points')
    
    if args.pdf:
        if dpar['verbose']:
            print('  Plotting average κ_s')
        nsl = []
        for s in msl:
            nsl.append(-s)
        # PLOT AVERAGE KAPPA_S
        plt.figure(figsize = (8,4.5))
        plt.plot(msl,mkl,color = 'blue',lw = 2)
        plt.plot(msl,dkl,color = '#5555AA')
        plt.plot(msl,ukl,color = '#5555AA')
        plt.plot(nsl,mkl,color = 'blue',lw = 2)
        plt.plot(nsl,dkl,color = '#5555AA')
        plt.plot(nsl,ukl,color = '#5555AA')
        plt.title('Average symmetric meridional curvature for '+dataset+'\nwindow = '+str(2000*ww)+' nm every '+str(ws*1000)+' nm')
        plt.xlim(-smax,smax)
        plt.axhline(color = 'gray',zorder = 0)
        plt.axvline(color = 'gray',zorder = 0)
        plt.xlabel('s ('+r'$\mu$'+'m)')
        plt.ylabel(r'$\kappa_s$ ('+r'$\mu$'+'m'+r'$^{-1}$'+')')
        pdfout.savefig()
        plt.close()

    # COMPUTE AVERAGE CONTOUR
    if dpar['verbose']:
        print('  Computing symmetric average contour')
    nas = len(msl)
    phi = 0.0
    avr = 0.0
    avx = 0.0
    avd = msl[0]
    avs = mkl[0] # Ks
    avt = mkl[0] # Kt
    avn = 0.0
    avfl = [phi]
    avrl = [avr]
    avxl = [avx]
    avdl = [avd]
    avsl = [avs]
    avtl = [avt]
    avnl = [avn]
    '''
    o = 0
    '''
    i=1
    while i < nas:
        deltas = msl[i]-avd
        ang = deltas*avs/2
        phi += ang
        avr += deltas*np.cos(phi)
        avx -= deltas*np.sin(phi)
        phi += ang
        avd = msl[i]
        avs = mkl[i]
        avt = np.sin(phi)/avr
        nu = (1-avs/avt)/2
        avfl.append(-phi)
        avrl.append(avr)
        avxl.append(avx)
        avdl.append(avd)        
        avsl.append(avs)
        avtl.append(avt)
        avnl.append(nu)
        '''
        if o < 10:
            print('##',i,round(sp,4),round(s1,4),'##',round(mk,6),
            round(ang,6),round(phi,6),'==',round(avr,6),round(avx,6),' [',np.arcsin(ang),']')
        '''
        i += 1
        '''
        o += 1
        '''
    
    lpw = len(avdl)
    if dpar['verbose']:
        print('  Average contour values computed for 2 x',lpw,'points')

    ### RESCALE 
    if dpar['verbose']:
        print('  Rescaling...',end='',flush=True)

    rsdl = avdl
    rssl = []
    rsnl = []
    i = 0
    while avsl[i] >= 0:
        Ks = avsl[i]
        Nu = avnl[i]
        #if Nu < 0.0:
        #    Nu = 0.0
        #if Nu > 0.5:
        #    Nu = 0.5
        rssl.append(Ks)
        rsnl.append(Nu)
        i += 1
    while i<lpw:
        rssl.append(0.0)
        rsnl.append(0.5)
        i += 1
    
    prsl = rssl
    ssdl = []

    # iterate
    for it in range(1000) :

        # use previous κs to compute φ, κθ, and new κs
        s = 0.0
        phi = 0.0
        r = 0.0
        Ks = prsl[0]
        nksl = [Ks]
        i = 1
        while i<len(prsl) :    
            deltas = rsdl[i]-s
            ang = deltas*Ks/2
            phi += ang
            r += deltas*np.cos(phi)
            phi += ang
            Kt = np.sin(phi)/r
            Ks = (1-2*rsnl[i])*Kt
            nksl.append(Ks)
            s = rsdl[i]
            Ks = prsl[i]
            i += 1
        '''
        print('it =',it+1,'φ =',phi,'factor =',(np.pi/2)/phi,'Ks',nksl[0],'...',nksl[-1])
        '''
        # rescale κs so that -π/2 ≤ φ ≤ π/2
        for i in range(len(nksl)):
            nksl[i] = nksl[i]*abs((np.pi/2)/phi)
    
        # evaluate
        ssd=0.0
        for i in range(len(nksl)):
            ssd+=(nksl[i]-prsl[i])**2
        if ssd>0:
            ssdl.append(np.log(ssd))

        prsl = nksl

        if len(ssdl)>=2 and ssdl[-1]==ssdl[-2]:
            break

    npts = len(nksl)
    tmpn = list(rsnl)
    s = 0.0
    Ks = nksl[0]
    r = 0.0
    x = 0.0
    phi = 0.0
    rssl = [Ks]
    rsxl = [x]
    rsrl = [r]
    rspl = [phi]
    rstl = [Ks]
    rsnl = [tmpn[0]]
    i = 1
    while i <npts:
        deltas = rsdl[i] - s
        ang = deltas*Ks/2
        phi += ang
        r += deltas*np.cos(phi)
        x -= deltas*np.sin(phi)
        phi += ang
        s = rsdl[i]
        Ks = nksl[i]
        Kt = np.sin(phi)/r
        Nu = tmpn[i]
        rssl.append(Ks)
        rsrl.append(r)
        rsxl.append(x)
        rspl.append(-phi)
        rstl.append(Kt)
        rsnl.append(Nu)
        i += 1

    npts = len(rsdl)
    ngsl = []
    narl = []
    ngrl = []
    ngpl = []
    ngfl = []
    basx = rsxl[-1]
    for i in range(npts):
        ngsl.append(-rsdl[i])
        narl.append(-avrl[i])
        ngrl.append(-rsrl[i])
        ngpl.append(-rspl[i])
        ngfl.append(-avfl[i])
        avxl[i] = avxl[i]-basx
        rsxl[i] = rsxl[i]-basx

    if dpar['verbose']:
        print(' done, 2 x',npts,'points',flush=True)

    if args.pdf:
        if dpar['verbose']:
            print('  Drawing average symmetric contour made of 2 x',
                len(avrl),'points')
        # DRAW AVERAGE CONTOUR
        rrg = 2*max(rsrl)
        xrg = max(rsxl)-min(rsxl)
        frg = 1.1*max(rrg,xrg)
        rmd = min(rsrl) + rrg / 2
        xmd = min(rsxl) + xrg / 2
        plt.figure(figsize = (8,8))
        #print("PLOT RX:",len(avrl),len(avxl),len(rsrl),len(rsxl))
        #plt.scatter(avrl,avxl,marker = '.',s = 1,color = 'gray')
        #plt.scatter(narl,avxl,marker = '.',s = 1,color = 'gray')
        plt.scatter(rsrl,rsxl,marker = 'o',s = 6, color = dpar['contour'])
        plt.scatter(ngrl,rsxl,marker = 'o',s = 6, color = dpar['contour'])
        plt.title('Average symmetric contour for '+dataset+'\nwindow = '+str(2000*ww)+' nm every '+str(ws*1000)+' nm')
        plt.xlim(-frg/2,frg/2)
        plt.ylim(xmd-frg/2,xmd+frg/2)
        plt.xlabel('r ('+r'$\mu$'+'m)')
        plt.ylabel('x ('+r'$\mu$'+'m)')
        pdfout.savefig()
        plt.close()

        if dpar['verbose']:
            print('  Plotting output φ')
        # PLOT OUTPUT PHI
        plt.figure(figsize = (8,4.5))
        #print("PLOT PHI:",len(avdl),len(avfl),len(rsdl),len(rspl))
        plt.plot(avdl,avfl,color = 'plum',lw = 2)
        plt.plot(ngsl,ngfl,color = 'plum',lw = 2)
        plt.plot(rsdl,rspl,color = 'purple',lw = 2)
        plt.plot(ngsl,ngpl,color = 'purple',lw = 2)
        plt.title('Normal to contour - Axis angle for '+dataset+'\nwindow = '+str(2000*ww)+' nm every '+str(ws*1000)+' nm')
        plt.xlim(-smax,smax)
        plt.axhline(y = 0,color = 'gray',zorder = 0)
        plt.axvline(x = 0,color = 'gray',zorder = 0)
        plt.axhline(y = -np.pi/2,color = 'gray',zorder = 0)
        plt.axhline(y = np.pi/2,color = 'gray',zorder = 0)
        plt.yticks(np.arange(-np.pi/2, np.pi, np.pi/2),
                         [r'$-\pi$/2','0',r'+$\pi$/2'])
        plt.xlabel('l ('+r'$\mu$'+'m)')
        plt.ylabel(r'$\varphi$'+' (rad)')
        pdfout.savefig()
        plt.close()

        if dpar['verbose']:
            print('  Plotting output κ_s and κ_θ')
        # PLOT OUTPUT KAPPA_S AND KAPPA_THETA
        plt.figure(figsize = (8,4.5))
        plt.plot(rsdl,rstl,color = 'green',lw = 2)
        plt.plot(ngsl,rstl,color = 'green',lw = 2)
        plt.plot(rsdl,rssl,color = 'blue',lw = 2)
        plt.plot(ngsl,rssl,color = 'blue',lw = 2)
        plt.title('Curvatures for '+dataset+'\nwindow = '+str(2000*ww)+' nm every '+str(ws*1000)+' nm')
        plt.xlim(-smax,smax)
        plt.axhline(color = 'gray',zorder = 0)
        plt.axvline(color = 'gray',zorder = 0)
        plt.xlabel('s ('+r'$\mu$'+'m)')
        plt.ylabel(r'$\kappa_i$ ('+r'$\mu$'+'m'+r'$^{-1}$'+')')
        pdfout.savefig()
        plt.close()

        if dpar['verbose']:
            print('  Plotting output ν')
        # PLOT OUTPUT NU
        plt.figure(figsize = (8,4.5))
        plt.plot(rsdl,rsnl,color = 'brown',lw = 2)
        plt.plot(ngsl,rsnl,color = 'brown',lw = 2)
        plt.title('Flow coupling for '+dataset+'\nwindow = '+str(2000*ww)+' nm every '+str(ws*1000)+' nm')
        plt.xlim(-smax,smax)
        plt.axhline(y = 0,color = 'gray',zorder = 0)
        plt.axhline(y = 0.5,color = 'gray',zorder = 0)
        plt.axvline(x = 0,color = 'gray',zorder = 0)
        plt.xlabel('s ('+r'$\mu$'+'m)')
        plt.ylabel(r'$\nu$')
        pdfout.savefig()
        plt.close()
    
        pdfout.close()
        if dpar['verbose']:
            print('  Plots for dataset',dataset,'stored in',plotfile)

    # OUTPUT VALUES FOR THIS DATASET
    if dpar['verbose']:
        print('  Writing output values to ',outfile,'... ',sep='',end='',flush=True)
    # PRINT X, Y, s, Phi, Kappa_s, Kappa_t, Nu

    nwrit = 0
    for j in range(npts-1):
        i = -1-j
        tabout.write(dataset+'\t'+str(ngrl[i])+'\t'+str(rsxl[i])
                    +'\t'+str(ngsl[i])+'\t'+str(ngpl[i])
                    +'\t'+str(rssl[i])+'\t'+str(rstl[i])
                    +'\t'+str(rsnl[i])+'\n')
        nwrit += 1
    for i in range(npts):
        tabout.write(dataset+'\t'+str(rsrl[i])+'\t'+str(rsxl[i])
                    +'\t'+str(rsdl[i])+'\t'+str(rspl[i])
                    +'\t'+str(rssl[i])+'\t'+str(rstl[i])
                    +'\t'+str(rsnl[i])+'\n')
        nwrit += 1
    if dpar['verbose']:
        print('done,',nwrit,'data written',flush=True)

    if dpar['verbose']:
        print(' Done with '+dataset+'\n')
                
    # END OF PROCESS FOR ONE DATASET

if args.verbose:
    print('Output for',args.infile,'stored in',outfile)
    print('Done with '+args.infile+'\n')
tabout.close()

sys.exit(0)

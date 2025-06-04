#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
from scipy.interpolate import griddata
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors
import os
import plumed
import argparse

#reuse same plumed kernel, to avoid multiple warnings
PLUMED_KERNEL=plumed.Plumed()
from functools import partial
plumed.read_as_pandas = partial(plumed.read_as_pandas, kernel=PLUMED_KERNEL)

#some other plotting functions
from matplotlib.pyplot import MultipleLocator
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from IPython.display import clear_output

#set bigger font sizes
SMALL_SIZE = 16
MEDIUM_SIZE = 17
BIG_SIZE = 20
plt.rc('font', size=SMALL_SIZE)        # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)   # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)   # fontsize of the figure title

def load_fes(fes_folder):
    filename = os.path.join(fes_folder,'fes-rew.dat')
    X = plumed.read_as_pandas(filename, usecols=[2]).to_numpy()
    nbins = int(np.sqrt(len(X)))
    fes = []
    return X.reshape(nbins,nbins), fes

def get_xy_min_max(filename,cvs):
    with open(filename) as f: 
        lines = f.readlines()
        for line in lines:
            if "SET min_%s"%cvs[0] in line:
                minss = float(line.split()[3])
            if "SET max_%s"%cvs[0] in line:
                maxss = float(line.split()[3])    
            if "SET min_%s"%cvs[1] in line:
                mindd = float(line.split()[3])
            if "SET max_%s"%cvs[1] in line:
                maxdd = float(line.split()[3])
    return (minss,maxss,mindd,maxdd)

def plot_ala2_fes(ax, fes, myextent,max_fes=None):
    if max_fes is None:
        max_fes = np.amax(fes)
    im = ax.imshow(fes, vmax=max_fes, origin='lower', extent=myextent)
    cb = fig.colorbar(im, ax=ax, label='Free energy [kJ/mol]')
    #ax.set_yticks([-2,0,2])
    ax.set_aspect('auto')
    ax.set_xlabel('sp: SOLVATION')
    ax.set_ylabel('sd: IONDISTANCE')
    #ax.set_title(title)
    return cb

def get_star(x,y,z):
    z=list(z)
    min_value=min(z)
    min_index=z.index(min_value)
    return x[min_index],y[min_index]

def get_star_text(sx1,sy1,axes,name,mycolor='black',mycolor2='white',addx=0.05,addy=0.25,size=19): 
    white_border = path_effects.Stroke(linewidth=3, foreground=mycolor2)              
    star = axes.scatter(sx1,sy1,s=120,c=mycolor,marker=(5,1),alpha=0.75)
    star.set_path_effects([white_border, path_effects.Normal()])
    text = axes.text(sx1+addx,sy1+addy,'%s'%name,size=size,weight='bold',c=mycolor,alpha=0.75)
    text.set_path_effects([white_border, path_effects.Normal()])

###change below
folder     = "./"
max_fes    = 96 #kJ/mol
pathlist   = ['sp_sd_bin50_block3']
cvnames    = ['s05','d05']
flag       = 'nocharge'
set_bar_ticks = True  # modify color bar ticks 
write_data    = False # write files for finding path
write_skip    = False # write files for finding path
draw_path     = False
draw_grid     = False
###change above

#colors = [(255 ,255 ,255),
#          (31 ,59 ,115),
#          (40 ,111,134), 
#          (53 ,152,146),
#          (73 ,171,142 ),
#          (114,192,118 ),
#          (167,214,85 ),
#          (198,217,83 ),         
#          (219,220,71 ),
#          (231,206,74 ),
#          (243,193,77 ),          
#          (255,180,80 ),
#          (240,160,75 ),          
#          (228,120,69 ),
#          (200, 80,57 ),          
#          (188, 47,46 )]  # R -> G -> B 

colors = [(255,255,255),
#          (31,59,115),
          (34,75,121),
          (40,108,133), 
          (46,141,146),
          (57,156,146),
          (70,168,143),
          (85,180,138),
          (118,194,116),         
          (184,216,81),
          (216,220,72),
          (250,223,63),          
          (255,207,69),
          (255,186,78),          
          (252,164,83),
          (237,133,74),          
#          (222,102,64),
          (214,87,59)
          ]  # R -> G -> B 
          
colornum=16
colors=list(tuple(i/255 for i in color) for color in colors)
colors.reverse()
cm = LinearSegmentedColormap.from_list('my_list', colors, N=colornum)

for k in pathlist:
    multiplot,axes = plt.subplots(1, 1)
    multiplot.set_size_inches((8, 8))
    
    x,y,z=np.loadtxt(os.path.join(folder,k,"fes-rew.dat"),unpack=True)[:3]
    z=z-z.min()
    z[z>max_fes]=max_fes
    #z[z<11]=11

    yi = np.linspace(min(y), max(y), 500)
    xi = np.linspace(min(x), max(x), 500)
    X,Y=np.meshgrid(xi,yi)
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='nearest')
    
    if write_data:
        ffile = open(os.path.join(folder,k,"fes-analysis.dat"),'w')
        ffile.write('#%16s%16s%16s\n'%('x','y','fes'))
        xx = list(xi)
        yy = list(yi)
        zz = list(zi)
        for i in range(len(yy)):
            for j in range(len(xx)):
                ffile.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],zz[i][j]))
        ffile.close()

    if write_skip:
        fskip1 = open(os.path.join(folder,k,"fes-analskip1.dat"),'w')
        fskip2 = open(os.path.join(folder,k,"fes-analskip2.dat"),'w')
        fskip3 = open(os.path.join(folder,k,"fes-analskip3.dat"),'w')        
        fskip1.write('#%16s%16s%16s\n'%('x','y','fes'))
        fskip2.write('#%16s%16s%16s\n'%('x','y','fes'))
        fskip3.write('#%16s%16s%16s\n'%('x','y','fes'))
        xx = list(xi)
        yy = list(yi)
        zz = list(zi)
        for i in range(len(yy)):
            for j in range(len(xx)):
                #for ani2zwi2can
                if -1<xx[j]<-0.5 and 0<yy[i]<2:
                    fskip1.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],max_fes))
                else:
                    fskip1.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],zz[i][j]))
                #for can2zwi2cat    
                if -1<xx[j]<-0.5 and 0<yy[i]<2:
                    fskip2.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],max_fes))
                elif -0.5<=xx[j]<0.4 and 1<yy[i]<1.2:
                    fskip2.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],max_fes))                
                else:
                    fskip2.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],zz[i][j]))
                #for can2cat    
                if  -1.30<=xx[j]<0.50 and 1.0<yy[i]<12:
                    fskip3.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],max_fes))
                elif 0.87<=xx[j]<0.91 and 4.0<yy[i]<12:
                    fskip3.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],max_fes))
                elif 0.85<=xx[j]<0.87 and 3.7<yy[i]<12:
                    fskip3.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],max_fes))                    
                elif 0.81<=xx[j]<0.85 and 2.3<yy[i]<12:
                    fskip3.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],max_fes))
                elif 0.77<=xx[j]<0.81 and 1.9<yy[i]<12:
                    fskip3.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],max_fes))                    
                elif 0.72<=xx[j]<0.77 and 1.9<yy[i]<12:
                    fskip3.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],max_fes))
                elif 0.67<=xx[j]<0.72 and 1.7<yy[i]<12:
                    fskip3.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],max_fes))                    
                elif 0.61<=xx[j]<0.67 and 1.7<yy[i]<12:
                    fskip3.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],max_fes))
                elif 0.55<=xx[j]<0.61 and 1.7<yy[i]<12:
                    fskip3.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],max_fes))                    
                elif 0.50<=xx[j]<0.55 and 1.3<yy[i]<12:
                    fskip3.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],max_fes))                    
                else:
                    fskip3.write(' %16.8f%16.8f%16.8f\n'%(xx[j],yy[i],zz[i][j]))                    
        fskip1.close()
        fskip2.close()
        fskip3.close()
    
    extent_opes = get_xy_min_max(os.path.join(folder,k,"fes-rew.dat"),cvnames)
    axes.set_xlim([extent_opes[0],extent_opes[1]])
    axes.set_ylim([extent_opes[2],extent_opes[3]])

    axes.contour(xi, yi, zi, colornum, colors='k', linewidths=0.8)
    mlt=axes.contourf(xi, yi, zi, colornum,cmap=cm,)
    axes.xaxis.set_major_locator(MultipleLocator(0.5))

    axes.set_xlabel(r'$s_p$'+': protonation',fontsize='20')
    if flag == 'charge':
        axes.set_ylabel(r'$s_c$'+': ion charge',fontsize='20')
        axes.yaxis.set_major_locator(MultipleLocator(0.5))
    else:
        axes.set_ylabel(r'$s_d$'+': ion distance'+" ("+r"$ \rm \AA)$",fontsize='20')
        axes.yaxis.set_major_locator(MultipleLocator(2))
        
    if draw_grid:
        axes.yaxis.set_major_locator(MultipleLocator(0.4))
        axes.xaxis.set_major_locator(MultipleLocator(0.2))
        axes.grid(c='white')

    cbar_ax = multiplot.add_axes([0.92, 0.11, 0.018, 0.77])
    if set_bar_ticks:
        mycbar = multiplot.colorbar(mlt, cax=cbar_ax, ticks=[xx*6 for xx in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]],
                            label=r'$\Delta$'+r'$F$'+': relative free energy (kJ/mol)')
        mycbar.ax.set_yticklabels(["%d"%int(xx*6) for xx in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]+['>96'])                   
    else:       
        multiplot.colorbar(mlt, cax=cbar_ax, label='$Delta$'+r'$F$'+': relative free energy (kJ/mol)')  
    
    if draw_path:
        path_skip=10
        px1,py1,pz1=np.loadtxt(os.path.join(folder,k,"ani2cat_2.dat"),unpack=True)[1:4]
        white_border = path_effects.Stroke(linewidth=2, foreground='white')              
        
        mypath = axes.scatter(px1[30:925:path_skip],py1[30:925:path_skip],c='black',s=5,label='minimum free energy path')
        mypath.set_path_effects([white_border, path_effects.Normal()])        
        
        star = axes.scatter([-1.04,-1.0,-0.96,-0.92,-0.88],[11.4,11.4,11.4,11.4,11.4],s=5,c='black')
        text = axes.text(-0.84,11.2,'Minimum free energy path',size=16,c='black',weight='medium')
        text = axes.text(-0.84,10.7,'from [A]nionic to [N]eutral to',size=16,c='black',weight='medium')
        text = axes.text(-0.84,10.2,'[Z]witterionic to [C]ationic Glycines',size=16,c='black',weight='medium')
        
        star = axes.scatter([-1.04,-1.0,-0.96,-0.92,-0.88],[12,12,12,12,12],s=5,c='gray')
        text = axes.text(-0.84,11.8,'One of the free energy paths',size=16,c='gray',weight='medium')
        
        #axes.legend(loc=2)
        px2,py2,pz2=np.loadtxt(os.path.join(folder,k,"ani2can_2.dat"),unpack=True)[1:4]
        mypath = axes.scatter(px2[124::path_skip],py2[124::path_skip],c='gray',s=5)
        mypath.set_path_effects([white_border, path_effects.Normal()])        
        
        px3,py3,pz3=np.loadtxt(os.path.join(folder,k,"can2zwi_h.dat"),unpack=True)[1:4]
        #axes.scatter(px3[118::path_skip],py3[118::path_skip],c='white',s=3)
        
        #anion
        sx1,sy1 = get_star(px1[:200],py1[:200],pz1[:200])
        get_star_text(sx1,sy1,axes,'A')
        #anion'
        sx1,sy1 = get_star(px1[:1],py1[:1],pz1[:1])
        get_star_text(sx1+0.025,sy1+1.4,axes,'A\'',mycolor='silver',mycolor2='black')
        #cationic'
        sx1,sy1 = get_star(px1[-2:],py1[-2:],pz1[-2:])
        get_star_text(sx1+0.025,sy1+2,axes,'C\'',mycolor='silver',mycolor2='black') 
        #zwitterion'
        sx3,sy3 = get_star(px3[-2:],py3[-2:],pz3[-2:])
        get_star_text(sx3,sy3,axes,'Z\'',mycolor='silver',mycolor2='black')        
        #canonical
        sx1,sy1 = get_star(px1[200:507],py1[200:507],pz1[200:507])
        get_star_text(sx1,sy1,axes,'N')
        #zwitterion
        sx1,sy1 = get_star(px1[507:622],py1[507:622],pz1[507:622])
        get_star_text(sx1,sy1,axes,'Z',addx=0.035,addy=0.32)
        #ZN
        sx1,sy1 = get_star(px1[563:564],py1[563:564],pz1[563:564])
        get_star_text(sx1,sy1-0.1,axes,'ZN',mycolor='gray')         
        #cationic
        sx1,sy1 = get_star(px1[860:1059],py1[860:1059],pz1[860:1059])
        get_star_text(sx1,sy1,axes,'C')
        #NA
        sx1,sy1 = get_star(px1[366:367],py1[366:367],pz1[366:367])
        get_star_text(sx1,sy1,axes,'NA',mycolor='gray',addx=-0.06,addy=-0.75)        
        #NC
        sx2,sy2 = get_star(px2[637:638],py2[637:638],pz2[637:638])
        get_star_text(sx2,sy2,axes,'NC',mycolor='gray',addx=-0.06,addy=-0.75) 
        #ZA
        sx2,sy2 = get_star(px2[236:237],py2[236:237],pz2[236:237])
        get_star_text(sx2+0.025,sy2-0.4,axes,'ZA',mycolor='gray',addx=-0.1,addy=-0.7) 
        #ZC
        sx1,sy1 = get_star(px1[751:752],py1[751:752],pz2[751:752])
        get_star_text(sx1,sy1-0.5,axes,'ZC',mycolor='gray',addx=0.07,addy=-0.4) 
        
    multiplot.savefig(os.path.join(os.path.join(folder,k),"%s_%s_%s.png"%(k,"fes_reweight","1")),
                dpi=300, bbox_inches='tight')

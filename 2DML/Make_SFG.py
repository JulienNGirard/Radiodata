
#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ***
    Make_SFG
    ***
"""

__author__ = 'Julien Girard'
__copyright__ = 'Copyright 2020'
__credits__ = ['Julien Girard']
__maintainer__ = 'Julien'
__email__ = 'julien.girard@cea.fr'
__status__ = 'Production'
__all__ = [
    'Make_SFG'
]

import numpy as np
import galsim
from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import random
%matplotlib inline

Catalogue='catalogue_SFGs_complete_wide1.fits'
Nsources=300      # Number of objects to consider
Npix=128        # Size of final postage stamp
FFTBIGSIZE=81488  # Max siez of support
np.random.seed(42)# Initializing seed for reproducibility
pixelscale=0.24   # Size of pixel in arcsec on the sky. Helps sampling the images w.r.t. catalogues.
                  # Important, ask a radioastronomer

# Internal cuisine, do not touch this
big_fft_params=galsim.GSParams(maximum_fft_size=FFTBIGSIZE)
stampimage = galsim.ImageF(Npix, Npix, scale=pixelscale)
b=galsim.BoundsI(1,Npix,1,Npix)
stamp=stampimage[b]

catalogsfg=fits.open(Catalogue)                                                
catalogsfg.info()
cat1=catalogsfg[1]
catdatasfg=cat1.data

flux1400sfg=catdatasfg['I1400']  # flux density at 1400 MHz
sizesfg=catdatasfg['size']       # angular size on the sky (in arcsec)
e1=catdatasfg['e1']              # first ellipticity
e2=catdatasfg['e2']              # second ellipticity

# Filtering objects that are larger than 10 pixels and smaller than 70 pixel on the sky.
filterobj=np.logical_and(sizesfg > 10*pixelscale, sizesfg <70*pixelscale)
filterobj2=np.where(filterobj == True)[0]
Ntotobj=len(filterobj2)
print(Ntotobj)

# Among all sources, draw 300.
randidx=np.random.choice(filterobj2,Nsources)

# Generate Star-forming galaxies (T-RECS) and plotting
tabgal=[]
tabgal2=np.empty((Nsources,Npix,Npix))
cnt=0
for iobj in randidx:
    gauss_gal=galsim.Gaussian(fwhm=sizesfg[iobj],flux=flux1400sfg[iobj])
    gal = galsim.Exponential(half_light_radius=gauss_gal.half_light_radius, flux=flux1400sfg[iobj], gsparams=big_fft_params)
    ellipticity = galsim.Shear(e1=e1[iobj],e2=e2[iobj])
    gal = gal.shear(ellipticity)
    gal2=gal.drawImage(stamp,scale=pixelscale)
    tabgal.append(gal2.array)
    tabgal2[cnt,:,:]=gal2.array
    cnt=cnt+1
    del gal


np.savez("Cat-SFG-%s.npz"%(Nsources),Nsources=Nsources,listgal=np.array(tabgal2),flux1400sfg=flux1400sfg[randidx],sizesfg=sizesfg[randidx],randidx=randidx,e1=e1[randidx],e2=e2[randidx])
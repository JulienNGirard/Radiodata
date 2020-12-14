
#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ***
    Make_AGN
    ***
"""

__author__ = 'Julien Girard'
__copyright__ = 'Copyright 2020'
__credits__ = ['Julien Girard']
__maintainer__ = 'Julien'
__email__ = 'julien.girard@cea.fr'
__status__ = 'Production'
__all__ = [
    'Make_AGN'
]

import numpy as np
import galsim
from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import random
%matplotlib inline

Catalogue='catalogue_AGNs_complete_wide1.fits'
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

flux1400agn=catdataagn['I1400']
sizeagn=catdataagn['size']
rsagn=catdataagn['Rs']  # ratio between overall size and location of hotspots

# Filtering objects that are larger than 0.5 and smaller than 70 pixel on the sky.
filterobj=np.logical_and(sizeagn > 10*pixelscale, sizeagn <60*pixelscale)
filterobj2=np.where(filterobj == True)[0]

Ntotobj=len(filterobj2)
randidx=np.random.randint(0,Ntotobj,Nsources)

# Among all sources, draw 300.
randidx=np.random.choice(filterobj2,Nsources)

# Generate Star-forming galaxies (T-RECS) and plotting
# Active Galactic Nuclei
posang = random.uniform(0,2.*np.pi, Nsources)  # induce position angle variations

tabagn=[]
for idx,iobj in enumerate(randidx):
    lobe_flux=flux1400agn[iobj]*0.99    # 99% of the flux in lobes
    hs_flux=flux1400agn[iobj]-lobe_flux  # 1% of the flux in hotspots
    hs1_flux=hs_flux/3
    hs2_flux=hs_flux/3
    hs3_flux=hs_flux/3
    hs_offset=0.5*sizeagn[iobj]  #hotspot offset as a fraction of the whole object in arcsec
    lobe_offset=sizeagn[iobj] # offset of the lobe = size of agn
    
    # create realistic 
    lobe1=galsim.Gaussian(sigma=sizeagn[iobj]*0.25,flux=lobe_flux/2.,gsparams=big_fft_params)
    lobe2=galsim.Gaussian(sigma=sizeagn[iobj]*0.25,flux=lobe_flux/2.,gsparams=big_fft_params)

    lobe1=lobe1.shift(-lobe_offset/2,0) # shift the first lobe
    lobe2=lobe2.shift(lobe_offset/2,0)  # shift the second lobe
    
    gal  = lobe1+lobe2                              # add the two lobes together
    gal  = gal.rotate(posang[idx]*galsim.radians)   # rotate randomly the AGN
    gal2 = gal.drawImage(stamp,scale=pixelscale)    # update

    if rsagn[iobj]>=0.01:
        randomtilt=random.uniform(0.95,1.05)
        hs_ix_offset = hs_offset*np.sin(posang[idx]*galsim.radians*randomtilt)/pixelscale # convert arcsec offset in pixels
        hs_iy_offset = hs_offset*np.cos(posang[idx]*galsim.radians*randomtilt)/pixelscale # convert arcsec offset in pixels
     
        # Create the sub-image for this galaxy
        cen = np.array(gal2.array.shape)
        cenx,ceny=np.array(cen)/2

        # Add the hotspots as single pixel point sources
        gal2.array[int(cenx), int(ceny)] += hs1_flux

        fact=0.7
        gal2.array[np.int(cenx+hs_ix_offset*fact), np.int(ceny+hs_iy_offset*fact)] += hs2_flux
        gal2.array[np.int(cenx-hs_ix_offset*fact), np.int(ceny-hs_iy_offset*fact)] += hs3_flux
        print(np.int(cenx+hs_ix_offset),np.int(ceny+hs_iy_offset))
  
    tabagn.append(gal2.array)  

np.savez("Cat-AGN-%s.npz"%(Nsources),Nsources=Nsources,listgal=np.array(tabagn),flux1400agn=flux1400agn[randidx],sizeagn=sizeagn[randidx],randidx=randidx)
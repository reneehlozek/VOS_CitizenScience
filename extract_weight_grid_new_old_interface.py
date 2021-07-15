from __future__ import print_function, division, absolute_import

import sys, os
import ujson
import gc
import csv
import time

import numpy as np   # using 1.10.1
import pandas as pd  # using 0.13.1
import scipy
from os.path import basename, exists

# plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator
from argparse import ArgumentParser

# my own modules
#import aggregate_class_fast
#import weighting
#import get_info_fast
import itertools
import random


global NUMBINS  # make this a glocal variable

# scp -r /Users/Nora/Documents/research/TESS/planethunters/zooniverse_output/planet-hunters-tess-beta-classifications.csv nora@glamdring.physics.ox.ac.uk:/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/zooniverse_output/
# scp -r /Users/Nora/Documents/research/TESS/planethunters/code/PHT_extract/get_info_fast.py nora@glamdring.physics.ox.ac.uk:/mnt/zfsusers/nora/soft/PHT/extract
# scp -r /Users/Nora/Documents/research/TESS/planethunters/code/PHT_extract/aggregate_class_fast.py nora@glamdring.physics.ox.ac.uk:/mnt/zfsusers/nora/soft/PHT/extract 

# scp -r /Users/Nora/Documents/research/TESS/planethunters/code/PHT_extract/extract_weight_grid_new_old_interface.py nora@glamdring.physics.ox.ac.uk:/mnt/zfsusers/nora/soft/PHT/extract

# get the files from glamdring

# scp -r nora@glamdring.physics.ox.ac.uk:/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel12/analysis /Users/Nora/Desktop

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    with_mpi = True
except ImportError:
    mpi_rank = 0
    mpi_size = 1
    with_mpi = False
 
mpi_root = 0


t00 = time.time()


'''
This is a full aggregation up to the point of DB scanning. The output file of this code is the input file of the DB scan module. 
The variables that change per sector are defined at the top of the file

LCs
---
For sectors one and two each LC, of 27.8 days,  is chopped up into 4 lightcurves to make 4 individual subjects. 
For sectors 3 and 4 the LCs are chopped into 3 sections due to missing data. 


latest updates:
---------------
3/1/19 - rewrite the code so that you do not have to restart everythng everytime that something crashes - i.e. save the outputs as you go along. 
9/2/19 - calculation of SNR (until sector 4 this was stupidly not stored in the metadata and has to be recalcualted here)
       - extraction of the sim transit locations (in sims metadata under 'feedback')
       - comparison of the sims location to the markings for each subject - calculation of TP, TN, FP and FN 

29/3/19 - making the code faster at LSSTC workshop hackday
30/3/19 - put code into functions and start to parallelize
9/4/19 - fully parallelized, code works now. The output of this is the input for the DB scanning module which has the full output file. 

10/2/20 - change the weighting scheme
'''

active_workflow_id = 11235
active_workflow_major = 15
apply_weight = 3
rankfile_stem = 'subjects_ranked_by_weighted_'
normalise_weights = True
counts_out = True


#################################################################################

#################################################################################

# MODULES

def extract_info(classifications,args):
    """
    The purpose of this function is to extract all the relevant information from the data frame. 
    metadata : long string of information from the data frame - information is extracted from this.
    annotation_json : same as ^ (in json format i.e. can be queried like a dictionary)
    planet_classification : whether someone said yes/no
    sector : which sector
    user_label : information about the user
    created_day : when the classification was made
    chunk : which image (some sectors have 3 chunks (sector 2-4), others 4)
    subject_type : True => Sims, False => not sims  (double check this)
    sim : 
    TIC_ID : TIC ID of the subject
    TIC_LC : only applciable for sims
    min_pix : min pix for the data 
    max_pix : max pix for the data 
    min_time : min time for that chunks (also indicator of which chunk it is. )
    max_time : max time for that chunks (also indicator of which chunk it is. )
    SNR : the SNR of SIMS LCS.

    input
    -----
    classifications: a pandas data frame of raw data - this is chopped up into sections if run in glamdring so that it is processed on different nodes makign the code run a lot faster.

    output 
    -----
    classifications: a pandas data frame with new columns with extarcted data. 

    This funtion takes the classification data frame - the inition data or a subveset of it - and exracts
    all the import information that does not depend on the other things in the dataframe.
    i.e. this is the step before the grouping and the aggregating. 
    By making it into a function the process can be parallelized. 

    """

    classifications['metadata'] = [ujson.loads(q) for q in classifications.metadata]
    classifications['annotation_json'] = [ujson.loads(q) for q in classifications.annotations]
    
    # get the classifications of all of them - this is a list if they classified (yes), and a an empty list if they didn't (no)

    # Get subject info into a format we can actually use
    classifications['subject_json'] = [ujson.loads(q) for q in classifications.subject_data]
    
    #make a new column where None classifications are called NoClass
    classifications['planet_classification0'] = [q[0]['value'] for q in classifications.annotation_json]  
    
    classifications['planet_classification'] = classifications.apply(get_info_fast.yes_no_none, axis=1)
                      
    #delete the old planet_classification0 column
    del classifications['planet_classification0']

    #-----------------------------------------------
    # only look at classifications from sector...
    #-----------------------------------------------

    print ("length of classifications before sector selection {}".format(len(classifications)))
    classifications['sector'] = [get_info_fast.get_sector(q) for q in classifications['subject_ids subject_json'.split()].iterrows()]            # get the actual subject
    
    print (list(classifications.sector)[0:40])

    #classifications = classifications[classifications.sector == sector]
    
    print ("length of classifications AFTER sector selection {}".format(len(classifications)))
    sys.stdout.flush()
    
    if len(classifications) == 0:
        print ("LENGTH OF DF IS ZERO THERFORE SKIP")
        sys.stdout.flush()
        return classifications # just return the empty dataframe 

    #----------------------------------------------------------------------------
    # Filter out classifications by me... (I don't do them properly when testing, sometimes neither does Grant so maybe filter him out too)
    #----------------------------------------------------------------------------
    
    classifications = classifications[classifications.user_name != "nora.eisner"]  # me
    #classifications = classifications[classifications.user_name != "mrniaboc"]     # Grant 
    
    #---------------------------------
    # Only use LIVE classifications
    #---------------------------------
    
    #would that we could just do q['live_project'] but if that tag is missing for
    #any classifications (which it is in some cases) it crashes
    
    classifications['live_project']  = [get_info_fast.get_live_project(q) for q in classifications.metadata]
    
    ## if this line gives you an error you've read in this boolean as a string
    ## so need to convert "True" --> True and "False" --> False
    class_live = classifications[classifications.live_project].copy()
    n_class_thiswf = len(classifications)
    n_live = sum(classifications.live_project)
    n_notlive = n_class_thiswf - n_live
    #print(" Removing %d non-live classifications..." % n_notlive)
    
    classifications = pd.DataFrame(class_live)  # new panda dataframe with only the live classifications
    
    del class_live
    gc.collect()  # delete the deleted stuff (i.e. empty trash)
    
    #-------------------------------------------------------------
    # discard classifications not in the active workflow  
    #-------------------------------------------------------------
    
    #print("Picking classifications from the active workflow (id %d, version %d.*)" % (active_workflow_id, active_workflow_major))
    # use any workflow consistent with this major version, e.g. 6.12 and 6.23 are both 6 so they're both ok
    # also check it's the correct workflow id
    #the_active_workflow = [int(q) == active_workflow_major for q in classifications.workflow_version]
    #this_workflow = classifications.workflow_id == active_workflow_id
    #in_workflow = this_workflow & the_active_workflow
    ## note I haven't saved the full DF anywhere because of memory reasons, so if you're debugging:
    ## classifications_all = classifications.copy()
    #classifications = classifications[in_workflow]  # the new classification now only has the dedicated WF and version
    
    #-------------------------------------------------------------
    #print ("Number of yes' and no's before removal of systematics...")
    #print (classifications['planet_classification'].value_counts())
    #-------------------------------------------------------------
    
    classifications['user_label'] = [get_info_fast.get_alternate_sessioninfo(q) for q in classifications['user_name metadata'.split()].iterrows()]
    classifications['created_day'] = [q[:10] for q in classifications.created_at]

    #-----------------------------------
    # Classification yes/no information 
    #-----------------------------------
    #print("Getting classification info...")
    
    # Get annotation info into a format we can actually use
    
    # marked transits == there is a transit. 
    # nothing marked == there are no transits.
    
    #-----------------------------------
    
    # create a weight parameter but set it to 1.0 for all classifications (unweighted) - may change later
    classifications['weight'] = [1.0 for q in classifications.workflow_version]
    # also create a count parameter, because at the time of writing this .aggregate('count') was sometimes off by 1
    classifications['count'] = [1 for q in classifications.workflow_version]
    
    #print (classifications['subject_ids'])
    # ALL IMPORTANT STEP!!
    all_data = np.array([get_info_fast.get_all_data(q) for q in classifications['subject_ids subject_json'.split()].iterrows()])            # get the actual subject
    

    # the output of this is a list of these things:
    # take the transpose of them so that we can call each row something differnt in the panda df
    classifications['subject_type'] = all_data.T[0]
    classifications['sim'] = all_data.T[1]
    classifications['TIC_ID'] = all_data.T[2]
    classifications['TIC_LC'] = all_data.T[3]
    classifications['min_pix'] = all_data.T[4]
    classifications['max_pix'] = all_data.T[5]
    classifications['min_time'] = all_data.T[6]
    classifications['max_time'] = all_data.T[7]
    classifications['SNR'] = all_data.T[8]
    classifications['radius'] = all_data.T[9]
    classifications['temperature'] = all_data.T[10]
    classifications['Tmag'] = all_data.T[11]


    classifications['candidate'] = [get_info_fast.get_filename(q) for q in classifications['subject_ids subject_json TIC_ID TIC_LC sim min_time'.split()].iterrows()]

    #---------------------------------
    # Location of the Marked Transits #
    #---------------------------------
    
    # convert the xvals and widths (if in pixels) to times
    # but don't convert for the sims because we need those units to be the same as in the metadata and I don't want to have to convert both.

    # change the min and max times because they were wrong...

    if sector == 1:
        print ("Sector 1")
        classifications['xvals_pixels'] = [get_info_fast.get_marking(q) for q in classifications.annotation_json]
        classifications['width_pixels'] = [get_info_fast.get_width(q) for q in classifications.annotation_json]

        classifications.loc[(classifications.min_time < 1),'min_time']  = 0.1
        classifications.loc[(classifications.min_time > 6) & (classifications.min_time < 8),'min_time']  = 6.8
        classifications.loc[(classifications.min_time > 13) & (classifications.min_time < 15),'min_time']= 13.9
        classifications.loc[(classifications.min_time > 19) & (classifications.min_time < 22),'min_time']= 20.1

        classifications.loc[(classifications.max_time > 7) & (classifications.max_time < 9),'max_time']=   7.85
        classifications.loc[(classifications.max_time > 14) & (classifications.max_time < 17),'max_time'] = 14.75
        classifications.loc[(classifications.max_time > 20) & (classifications.max_time < 21),'max_time'] = 21.75
        classifications.loc[(classifications.max_time > 27),'max_time'] = 28

        classifications['xvals'], classifications['width'] = zip(*classifications.apply(get_info_fast.pixel_to_time, axis = 1))
    
    elif sector == 2:
        print ("Sector 2")
        classifications['xvals_pixels'] = [get_info_fast.get_marking(q) for q in classifications.annotation_json]
        classifications['width_pixels'] = [get_info_fast.get_width(q) for q in classifications.annotation_json]
        
        classifications.loc[(classifications.min_time < 1),'min_time']  = 0.1
        classifications.loc[(classifications.max_time > 7) & (classifications.max_time < 9),'max_time']=   7.8
    
        classifications.loc[(classifications.min_time > 6) & (classifications.min_time < 8),'min_time']  = 6.75
        classifications.loc[(classifications.max_time > 14) & (classifications.max_time < 17),'max_time'] = 13.1
    
        classifications.loc[(classifications.min_time > 13) & (classifications.min_time < 15),'min_time']= 14.7
        classifications.loc[(classifications.max_time > 20) & (classifications.max_time < 21),'max_time'] = 21.7
    
        classifications.loc[(classifications.min_time > 19) & (classifications.min_time < 22),'min_time']= 20.25
        classifications.loc[(classifications.max_time > 27),'max_time'] = 27.55
        
        classifications['xvals'], classifications['width'] = zip(*classifications.apply(get_info_fast.pixel_to_time, axis = 1))
        
        
    elif sector == 3:
        print ("Sector 3")
        classifications['xvals_pixels'] = [get_info_fast.get_marking(q) for q in classifications.annotation_json]
        classifications['width_pixels'] = [get_info_fast.get_width(q) for q in classifications.annotation_json]

        classifications['min_time'] = classifications['min_time'] + 4
        classifications['max_time'] = classifications['max_time'] + 4

        classifications['xvals'], classifications['width'] = zip(*classifications.apply(get_info_fast.pixel_to_time, axis = 1))
    
    
    elif sector == 4:
        print ("Sector 4")
        classifications['xvals_pixels'] = [get_info_fast.get_marking(q) for q in classifications.annotation_json]
        classifications['width_pixels'] = [get_info_fast.get_width(q) for q in classifications.annotation_json]

        classifications.loc[(classifications.min_time < 1),'min_time']= 0.1
        classifications.loc[(classifications.min_time > 8) & (classifications.min_time < 9),'min_time']= 10.4
        
        classifications.loc[(classifications.max_time > 7) & (classifications.max_time < 10),'max_time']= 8.1
        classifications.loc[(classifications.min_time > 16) & (classifications.min_time < 17),'min_time'] = 16.75
        classifications.loc[(classifications.max_time > 25),'max_time'] = 26.05
        
        classifications['xvals'], classifications['width'] = zip(*classifications.apply(get_info_fast.pixel_to_time, axis = 1))

    
    elif sector == 5:
        print ("Sector 5")
        classifications['xvals_pixels'] = [get_info_fast.get_marking(q) for q in classifications.annotation_json]
        classifications['width_pixels'] = [get_info_fast.get_width(q) for q in classifications.annotation_json]

        classifications.loc[(classifications.min_time > 0) & (classifications.min_time < 0.019),'min_time']  = 0.02
        classifications.loc[(classifications.max_time > 7.) & (classifications.max_time < 7.3),'max_time']= 7.4
    
        classifications.loc[(classifications.min_time > 6) & (classifications.min_time < 6.29),'min_time']  = 6.3
        classifications.loc[(classifications.max_time > 13) & (classifications.max_time < 14),'max_time'] = 12.35
    
        classifications.loc[(classifications.min_time > 12) & (classifications.min_time < 13),'min_time']= 13.55
        classifications.loc[(classifications.max_time > 19) & (classifications.max_time < 19.9),'max_time'] = 20
        
        classifications.loc[(classifications.min_time > 18) & (classifications.min_time < 18.79),'min_time']= 18.8
        classifications.loc[(classifications.max_time > 24.9) & (classifications.max_time < 26),'max_time'] = 26
        
        # ------
        
        classifications.loc[(classifications.min_time > 0.019) & (classifications.min_time < 0.022),'min_time']  = 0.02
        classifications.loc[(classifications.max_time > 7.3) & (classifications.max_time < 7.5),'max_time']= 7.4
    
        classifications.loc[(classifications.min_time > 6.29) & (classifications.min_time < 6.33),'min_time']  = 6.35
        classifications.loc[(classifications.max_time > 12) & (classifications.max_time < 13),'max_time'] = 13.2
    
        classifications.loc[(classifications.min_time > 13) & (classifications.min_time < 14),'min_time']= 13.2
        classifications.loc[(classifications.max_time > 19.9) & (classifications.max_time < 20.2),'max_time'] = 19.9
    
        classifications.loc[(classifications.min_time > 18.79) & (classifications.min_time < 18.85),'min_time']= 18.8
        classifications.loc[(classifications.max_time > 25.99) & (classifications.max_time < 26.5),'max_time'] = 26.1
        
        classifications['xvals'], classifications['width'] = zip(*classifications.apply(get_info_fast.pixel_to_time, axis = 1))
    
        
    elif sector == 6:
        print ("Sector 6")
        classifications['xvals_pixels'] = [get_info_fast.get_marking(q) for q in classifications.annotation_json]
        classifications['width_pixels'] = [get_info_fast.get_width(q) for q in classifications.annotation_json]
        
        classifications['min_time'] = classifications['min_time'] + 0.1
        classifications['max_time'] = classifications['max_time'] + 0.1
        
        classifications['xvals'], classifications['width'] = zip(*classifications.apply(get_info_fast.pixel_to_time, axis = 1))
    
        
    elif sector == 7:
        print ("Sector 7")
        classifications['xvals_pixels'] = [get_info_fast.get_marking(q) for q in classifications.annotation_json]
        classifications['width_pixels'] = [get_info_fast.get_width(q) for q in classifications.annotation_json]
        
        classifications.loc[(classifications.min_time > 0) & (classifications.min_time < 0.01),'min_time']  = 0.101
        classifications.loc[(classifications.max_time > 5) & (classifications.max_time < 6.9),'max_time']= 6.97
    
        classifications.loc[(classifications.min_time > 5) & (classifications.min_time < 5.88),'min_time']  = 5.87
        classifications.loc[(classifications.max_time > 12.5) & (classifications.max_time < 13.5),'max_time'] = 12
    
        classifications.loc[(classifications.min_time > 11.7) & (classifications.min_time < 11.8),'min_time']= 11.93
        classifications.loc[(classifications.max_time > 18) & (classifications.max_time < 18.6),'max_time'] = 18.67
    
        classifications.loc[(classifications.min_time > 15) & (classifications.min_time < 17.6),'min_time']= 17.69
        classifications.loc[(classifications.max_time > 23) & (classifications.max_time < 24.5),'max_time'] = 24.55
    
        # -------
    
        classifications.loc[(classifications.min_time > 0.01) & (classifications.min_time < 0.3),'min_time']  = 0.101
        classifications.loc[(classifications.max_time > 6.9) & (classifications.max_time < 7.2),'max_time']= 6.97
    
        classifications.loc[(classifications.min_time > 5.88) & (classifications.min_time < 6),'min_time']  = 5.87
        classifications.loc[(classifications.max_time > 11) & (classifications.max_time < 12),'max_time'] = 12
    
        classifications.loc[(classifications.min_time > 11.8) & (classifications.min_time < 12),'min_time']= 13
        classifications.loc[(classifications.max_time > 18.6) & (classifications.max_time < 18.9),'max_time'] = 18.87
    
        classifications.loc[(classifications.min_time > 17.6) & (classifications.min_time < 17.9),'min_time']= 17.69
        classifications.loc[(classifications.max_time > 24.5) & (classifications.max_time < 24.9),'max_time'] = 24.55
    
        classifications['xvals'], classifications['width'] = zip(*classifications.apply(get_info_fast.pixel_to_time, axis = 1))

   
    elif sector == 8:
        print ("Sector 8")
        classifications['xvals_pixels'] = [get_info_fast.get_marking(q) for q in classifications.annotation_json]
        classifications['width_pixels'] = [get_info_fast.get_width(q) for q in classifications.annotation_json]

        classifications.loc[(classifications.min_time > 0) & (classifications.min_time < 1),'min_time']  = 0.06
        classifications.loc[(classifications.max_time > 5) & (classifications.max_time < 8),'max_time']= 6.52
    
        classifications.loc[(classifications.min_time > 5) & (classifications.min_time < 7),'min_time']  = 5.5
        classifications.loc[(classifications.max_time > 10) & (classifications.max_time < 13),'max_time'] = 11.9
    
        classifications.loc[(classifications.min_time > 15) & (classifications.min_time < 18),'min_time']= 17.75
        classifications.loc[(classifications.max_time > 23) & (classifications.max_time < 26),'max_time'] = 24.7

        classifications['xvals'], classifications['width'] = zip(*classifications.apply(get_info_fast.pixel_to_time, axis = 1))
    
        
    elif ((sector == 9) and (args.interface == 'old')): # probbaly need another condition here becase there are two sector 9s ....
        print ("Sector 9")
        classifications['xvals_pixels'] = [get_info_fast.get_marking(q) for q in classifications.annotation_json]
        classifications['width_pixels'] = [get_info_fast.get_width(q) for q in classifications.annotation_json]

        classifications.loc[(classifications.min_time > 0) & (classifications.min_time < 2),'min_time']  = 1
        classifications.loc[(classifications.max_time > 5) & (classifications.max_time < 7),'max_time']= 6.95
    
        classifications.loc[(classifications.min_time > 5) & (classifications.min_time < 7),'min_time']  = 5.8
        classifications.loc[(classifications.max_time > 10) & (classifications.max_time < 13),'max_time'] = 12.4
    
        classifications.loc[(classifications.min_time > 13) & (classifications.min_time < 15),'min_time']= 14.4
        classifications.loc[(classifications.max_time > 19) & (classifications.max_time < 22),'max_time'] = 20.25
    
        classifications.loc[(classifications.min_time > 19) & (classifications.min_time < 20),'min_time']= 19.25
        classifications.loc[(classifications.max_time > 25) & (classifications.max_time < 28),'max_time'] = 25.35

        classifications['xvals'], classifications['width'] = zip(*classifications.apply(get_info_fast.pixel_to_time, axis = 1))

    else:

        print("Finding the location of the marked transits...")
    
        classifications['xvals'] = [get_info_fast.get_marking(q) for q in classifications.annotation_json]
        classifications['width'] = [get_info_fast.get_width(q) for q in classifications.annotation_json]


    sim = classifications.loc[(classifications['subject_type'] == True)].index
    classifications['simloc'] = pd.Series([get_info_fast.get_simloc(q) for q in classifications.subject_json[sim]],index = sim)
    classifications['simwidth'] = pd.Series([get_info_fast.get_sim_width(q) for q in classifications.subject_json[sim]],index = sim)

    def numsimtransits(row):

        try:
            return len(row['simloc'])
        except:
            return None
    
    #classifications['numsimtransits'] = classifications.apply(numsimtransits, axis = 1)

    #print (classifications['numsimtransits'])
    # run the function that tests whether the marked transit is correct or incorrect

    classifications = classifications.apply(get_info_fast.get_closest_new, axis = 1)

    sys.stdout.flush() 

    #print (classifications)

    return classifications


def plot_histogram_sys_init(classifications,args):

    '''
    This function plots the histgram which allows us to identify the cut off.
    '''

    notsim = classifications[(classifications['subject_type'] == False)]
    xval_sublist = list(notsim['xvals'])
    
    #mask = (xval_sublist == None)
    #xval_sublist = xval_sublist[~mask]

    #xvals_list = [item for sublist in xval_sublist if not pd.isnull(sublist).any() for item in sublist]
    
    xvals_list = np.hstack(xval_sublist)
    
    mask = (xvals_list == None)
    
    xvals_list = np.array(xvals_list[~mask], dtype=float)
    
    mask2 = np.where(np.isfinite(xvals_list))
    
    xvals_list = xvals_list[mask2]


    # -------------

    fig, ax = plt.subplots(figsize=(12,5))
    
    binboundaries = np.linspace(np.nanmin(xvals_list), np.nanmax(xvals_list), NUMBINS)


    ax.hist(list(xvals_list), density=True, bins=binboundaries)

    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    
    minorLocator = AutoMinorLocator()
    ax.yaxis.set_minor_locator(minorLocator)
    
    minorLocator = AutoMinorLocator()
    ax.xaxis.set_minor_locator(minorLocator)
    
    if sector < 9:
        plt.savefig('/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel0{}/analysis/marking_distribution.png'.format(sector), format='png')
        np.savetxt('/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel0{}/analysis/marking_distribution_data.txt'.format(sector), np.array([xvals_list]).T, delimiter = ' ')
        np.savetxt('/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel0{}/analysis/marking_distribution_binboundaries.txt'.format(sector), np.array([binboundaries]).T, delimiter = ' ')
        
    elif sector == 9:
        plt.savefig('/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel0{}/analysis_{}int/marking_distribution.png'.format(sector, args.interface), format='png')  

        np.savetxt('/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel0{}/analysis_{}int/marking_distribution_data.txt'.format(sector, args.interface), np.array([xvals_list]).T, delimiter = ' ')
        np.savetxt('/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel0{}/analysis_{}int/marking_distribution_binboundaries.txt'.format(sector, args.interface), np.array([binboundaries]).T, delimiter = ' ')
            
    else:
        plt.savefig('/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel{}/analysis/marking_distribution.png'.format(sector), format='png')
        np.savetxt('/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel{}/analysis/marking_distribution_data.txt'.format(sector), np.array([xvals_list]).T, delimiter = ' ')
        np.savetxt('/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel{}/analysis/marking_distribution_binboundaries.txt'.format(sector), np.array([binboundaries]).T, delimiter = ' ')
    
    return xvals_list, binboundaries



def histogram_sys_correction(classifications, xvals_list, LIMIT, binboundaries, args):

    '''
    This function gets rid of the systematics.
    '''

    print ("getting rid of systematics...")

    classifications['index'] =  classifications.index  # make a column of the index in the dataframe so that the two can be merged on this again later!
    
    #notsim = classifications[(classifications['subject_type'] == False)]

    #print (list(classifications[classifications.subject_type == True]['subject_ids'])[0:11])
    #print (len(list(classifications[classifications.subject_type == True]['subject_ids'])))
    #print (" - - - - - - - - - - - - -")
    #print (list(classifications[classifications.subject_type == False]['subject_ids'])[0:11])
    #print (len(list(classifications[classifications.subject_type == False]['subject_ids'])))
    #print ("________---------^^^^^^^^")

    #print (" - - - - - - - - - - - - -")

    df_sys = get_info_fast.get_marking_sysrem(classifications, LIMIT, binboundaries, xvals_list) # pass only one chunk through the function at a time

    length_before = len(classifications)

    classifications = pd.merge(classifications, df_sys, how='left', on=['index'], sort=False, suffixes=('_2', ''), copy=True)


    if length_before != (len(classifications)):
        print ("WARNING WARNING WARNING: the merging of dataframes changed the length from {}  to  {} ".format(length_before,(len(classifications))))

    del df_sys

    gc.collect()
    
    # notsim = classifications[(classifications['subject_type'] == False)]

    xval_sublist = list(classifications['xvals'])
    
    #mask = (xval_sublist == None)
    #xval_sublist = xval_sublist[~mask]

    #xvals_list = [item for sublist in xval_sublist if not pd.isnull(sublist).any() for item in sublist]
    
    xvals_list = np.hstack(xval_sublist)
    
    mask = (xvals_list == None)
    
    xvals_list = np.array(xvals_list[~mask], dtype=float)
    
    mask2 = np.where(np.isfinite(xvals_list))
    
    xvals_list = xvals_list[mask2]

    # ------

    xval_sublist2 = list(classifications['xval_sysrem'])

    xvals_list2 = np.hstack(xval_sublist2)
    
    mask3 = (xvals_list2 == None)
    
    xvals_list2 = np.array(xvals_list2[~mask3], dtype=float)
    
    mask4 = np.where(np.isfinite(xvals_list2))
    
    xvals_list2 = xvals_list2[mask4]


    # ----------------------
    # plot  histogram fo the markings with both removed and not removed systematics

    fig, ax = plt.subplots(figsize=(12,5))
    
    ax.hist(list(xvals_list), density=True, bins=binboundaries)
    ax.hist(list(xvals_list2), density=True, bins=binboundaries, alpha =0.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    
    minorLocator = AutoMinorLocator()
    ax.yaxis.set_minor_locator(minorLocator)
    
    minorLocator = AutoMinorLocator()
    ax.xaxis.set_minor_locator(minorLocator)
    

    if sector < 9:
        plt.savefig('/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel0{}/analysis/marking_distribution_removed.png'.format(sector), format='png')
    elif sector == 9:
        plt.savefig('/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel0{}/analysis_{}int/marking_distribution_removed.png'.format(sector, args.interface), format='png')
   
    else:
        plt.savefig('/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel{}/analysis/marking_distribution_removed.png'.format(sector), format='png')

    #plt.savefig('/Users/Nora/Documents/research/TESS/planethunters/zooniverse_output/marking_distribution_removed.png', format='png')

    return classifications


def weight_func(classifications,true_incr, false_incr, missed_incr):

    '''
    re-write this function so that it takes all the sims classifications into consideration, not just the ones from that one sector.
    
    Weighting
    ---------
    Weighting is currently based on simple yes/no of whetehr they marked anything - does not take into consideration whether what they marke dis correct
    Next step is to assess whether their marking is correct or not and base weighting on this. 
    '''

    #print (classifications)

    print ("START WEIGHTING FUNCTION")
    #sys.stdout.flush()

    #classifications = classifications[['subject_ids', 'planet_classification', 'weight', 'count', 'subject_type', 'sim', 'TIC_ID', 'TIC_LC', 'min_pix', 'max_pix','min_time', 'max_time','candidate', 'SNR','width','xvals', 'simloc', 'simwidth', 'width_sysrem','xval_sysrem', 'user_label','user_id','radius','temperature','Tmag', 'workflow_version', 'sim_status', 'missed_count', 'numsimtransits']]
    
    classifications.index.name = None

    classifications['seed'] = [0 for q in classifications.weight]
    classifications['is_gs'] = [0 for q in classifications.weight]

    print("Computing user weights...")

    is_known = classifications.subject_type == True

    #try:
    #    print ('sum')
    #    print (sum(is_known))
    #    print (len(is_known))
    #except:
    #    print ('sum failed')

    # is_candidate = np.invert(is_known)
    # if it's a non-gold-standard classification, mark it.
    classifications.loc[is_known, 'is_gs'] = 1

    # now that the data is aggregated (grouped by) the candidates, extract information that is the same for one candidate
    # note: the feedback information (even though it applies to all the same) has to be done earlier because it is needed in the single classification comparison
    
    # the weights going into this are all set to 1 this time
    # and the count is also set to 1
    # does not take into consideration the seed yet...
    # this returns the fraction of people who got each one right.

    sim_mask = classifications.subject_type == True
    
    classifications['weight'] = [1 for q in classifications.workflow_version]
    classifications['count'] = [1 for q in classifications.workflow_version]


    by_subj_sims = classifications[classifications.subject_type == True].groupby('subject_ids') # if it's known to be a planet, group by candidate
    

    sims_agg = by_subj_sims[list(classifications)].apply(aggregate_class_fast.get_difficulty)

    # replace NaN with 0.0
    sims_agg.fillna(value=0, inplace=True)
    
    # now we have multipliers for each sim subject, add them back to the classifications dataframe
    # add two new columns into the data frame that look at the yes and No probabilities 

    classifications_old = classifications.copy()

    #print (sims_agg)
    #print ("I HATE PYTHON IT'S THE WORST LADIDADIDA")

    #print (list(classifications_old['subject_ids'])[0:11])
    #print (type(list(classifications_old['subject_ids'])[0]))

    #print ("------dfakzdhsflakshdlfk-------")
    #print (list(sims_agg['subject_ids'])[0:11])

    sims_agg['subject_ids'] = sims_agg['subject_ids'].astype(int)

    #sims_agg.astype({'subject_ids': 'int'})
    #print (type(list(sims_agg['subject_ids'])[0]))

    #print (len(classifications_old), len(sims_agg))

    classifications = pd.merge(classifications_old, sims_agg, how='left', left_on='subject_ids', right_index=True, sort=False, suffixes=('_2', ''), copy=True)

    #print (len(classifications))
    try:
        classifications['subject_ids'] = classifications['subject_ids_2']
    except:
        print ("I hate python")

    del classifications_old

    # changing the seeds for each ok_class. Classification is correct then multiple the percentage of false positive with the ok incr.
    
    def calc_seed(row):

        total_seed = 0
        
        for idx in range(0,20):
            name = 'tr_{}'.format(idx)
            seed_name = 'True_' + name
            
            if row[name] == 1: # if they got it correct
                
                total_seed =+ (true_incr * (1/row[seed_name])) # seed name is difficulty 
            
            elif row[name] == 0:
            
                total_seed =+ (false_incr * (row[seed_name])) # seed name is difficulty 
            
            # None if 
            else:
                break # if not 0 or 1 then break out of the loop to save time on unecessary looping
                
        if row['missed_count'] != 0:
            try:
                total_seed =+ (row['missed_count']/row['avg_missed']) * missed_incr
            except:
                total_seed = total_seed

        return total_seed


    classifications['seed'] = classifications.apply(calc_seed, axis = 1)



    def num_markings(row):
        xvals = row['xvals']
        if type(xvals) == list:
            try:
                return len(xvals)
            except:
                print (type(xvals))
                print (xvals)
        else:
            return 0 
    
    
    def width_markings(row):
        width = row['width']
        if type(width) == list:
            try:
                return np.median(width)
            except:
                return None
        else:
            return None
    

    classifications['nummarkings'] = classifications.apply(num_markings, axis = 1)
    classifications['average_width_markings'] = classifications.apply(width_markings, axis = 1)



    # by this point we have generated a seed FOR EACH CLASSIFICATION, not each user.
    # this seed is based on whether their response agrees with the overall concensus.
    # more positive if it's correct but lots of people got it wrong
    # more negative if they got it wrong but everyone else got it right

    # then group classifications by user name, which will weight logged in as well as not-logged-in (the latter by session)
    by_user = classifications.groupby('user_label')
    
    
    # For each user, sum all of their seeds for each classification that they did. 
    # This goes into the exponent for the weight
    # This is their overall seed
    
    user_exp = by_user.seed.aggregate('sum')
    
    # then set up the DF that will contain the weights etc, and fill it
    user_weights = pd.DataFrame(user_exp)
    user_weights.columns = ['seed'] # label the summed seeds "seed"
    
    def calc_weighting(row):
         #for each user, sum of the calculated seed divided by the number of TRANSITS (not LCs) that they see
         #print ("total_seed : {}".format(row['total_seed']))
         #print ("total sim sum: {}".format(row['totalnumsimtransits']))
        
        c0 = 0.5
        n_gs = row['totalnumsimtransits']
        seed = row['total_seed']
        
        if (n_gs == 0) or (n_gs == None):
            #print ("No sims seen") t
            return c0
        else:
            
            weight1 = (np.min([3.0, np.max([0.05, c0*pow((1.0 + np.log10(n_gs)), (float(seed)/float(n_gs)))])]))
            #weight2 = (min([3.0, max([0.05,  seed  / n_gs ])]))
            
            return weight1
    

    
    user_weights['user_label'] = user_weights.index
    user_weights['totalnumsimtransits'] = by_user['num_sim'].aggregate('sum') #totoal number of transits the user saw
    user_weights['user_id'] = by_user['user_id'].head(1)
    user_weights['nclass_user'] = by_user['count'].aggregate('sum')
    # user_weights['n_gs'] = by_user['is_gs'].aggregate('sum')
    user_weights['total_seed'] = by_user['seed'].aggregate('sum')


    not_logged_in_mask = (user_weights['user_label'].str.contains("not-logged-in"))
    classifications_made_mask = user_weights['nclass_user'] > 15

    user_weights['weight_old'] = user_weights.apply(calc_weighting, axis = 1)


    weight = user_weights[~not_logged_in_mask & classifications_made_mask].seed/user_weights[~not_logged_in_mask & classifications_made_mask].totalnumsimtransits


    user_weights['total_average_width_markings'] = by_user['average_width_markings'].aggregate('median')
    user_weights['total_transits_marked'] = by_user['nummarkings'].aggregate('sum')


    # user_weights['weight'] = [assign_weight_old(q) for q in user_exp]
    normalise_weights = True
    # if you want sum(unweighted classification count) == sum(weighted classification count), do this
    if normalise_weights:
        user_weights['weight_unnorm'] = user_weights['weight_old'].copy()
        
        user_weights.weight_old *= float(len(classifications))/float(sum(user_weights.weight_old * user_weights.nclass_user))
    
    #user_weights.loc[user_weights['user_label'].str.contains("not-logged-in"), 'weight_old'] = 0.5
    #user_weights.loc[user_weights['nclass_user'] < 15, 'weight_old'] = 0.5

    # -------------

    # calculate the user weighting 
    # normalise to be centred on 1 (most people have a user weigthing of 1)

    weight = user_weights[~not_logged_in_mask & classifications_made_mask].seed/user_weights[~not_logged_in_mask & classifications_made_mask].totalnumsimtransits
    
    weight = user_weights.seed/user_weights.totalnumsimtransits
    weight_range = np.max(weight) - np.min(weight)
    weight_mean = np.mean(weight)
    user_weights['weight_normalised0'] = ((weight - weight_mean)/weight_range)
    
    user_weights['weight']   = user_weights['weight_normalised0'] * (2/np.max(user_weights['weight_normalised0'])) + 1
    user_weights['weight_3'] = user_weights['weight_normalised0'] * (3/np.max(user_weights['weight_normalised0'])) + 1
    user_weights['weight_4'] = user_weights['weight_normalised0'] * (4/np.max(user_weights['weight_normalised0'])) + 1
    user_weights['weight_5'] = user_weights['weight_normalised0'] * (5/np.max(user_weights['weight_normalised0'])) + 1
    
    user_weights.loc[user_weights['weight'] > 3, 'weight'] = 3
    user_weights.loc[user_weights['weight'] < 0.005, 'weight'] = 0.005
    user_weights.loc[user_weights['weight_3'] < 0.005, 'weight_3'] = 0.005
    user_weights.loc[user_weights['weight_4'] < 0.005, 'weight_4'] = 0.005
    user_weights.loc[user_weights['weight_5'] < 0.005, 'weight_5'] = 0.005
    
    user_weights['weight_normalised0'].fillna(1, inplace=True)
    user_weights['weight'].fillna(1, inplace=True)
    user_weights['weight_3'].fillna(1, inplace=True)
    user_weights['weight_4'].fillna(1, inplace=True)
    user_weights['weight_5'].fillna(1, inplace=True)


    # PUT IN EXTRA CONDITIONS ON THE WEIGHT TO SEE HOW THAT CHANGES THINGS. 

    def set_weight_to_low(row):

        if "not-logged-in" in row['user_label']:
            row['weight'] = 0.5
            row['weight_3'] = 0.5
            row['weight_4'] = 0.5
            row['weight_5'] = 0.5

        elif row['nclass_user'] < 10:
            row['weight'] = 0.5
            row['weight_3'] = 0.5
            row['weight_4'] = 0.5
            row['weight_5'] = 0.5

        if (row['total_transits_marked']/row['nclass_user']) > 4:
            return 0.005, 0.005, 0.005, 0.005
        
        elif row['total_average_width_markings'] > 3:
            return 0.005, 0.005, 0.005, 0.005
        
        else:
            return row['weight'], row['weight_3'], row['weight_4'], row['weight_5']
    

    user_weights['weight_strict'], user_weights['weight_3_strict'], user_weights['weight_4_strict'], user_weights['weight_5_strict'] = zip(*user_weights.apply(set_weight_to_low, axis = 1))


    # -------------

    classifications_old = classifications.copy()
    

    user_weights.index.name = 'Index weights'
    classifications_old.index.name = 'Index'

    classifications = pd.merge(classifications_old, user_weights, how='left',on='user_label', sort=False, suffixes=('_2', ''), copy=True)

    # -------------

    print ("length classification after merge")
    print (len(classifications))

    #print ((classifications))
    del classifications_old

    #gc.collect()
    
    nclass_mean   = np.mean(user_weights.nclass_user)
    nclass_median = np.median(user_weights.nclass_user)
    nclass_tot    = len(classifications)
    
    user_weights.sort_values(['nclass_user'], ascending=False, inplace=True)

    # save the counts files - was previously done later in the code but can do it here too.
    #if counts_out == True:
    #    print("Printing classification counts to %s..." % counts_out_file)
    #    user_weights['color'] = [get_info_fast.randcolor(q) for q in user_weights.index]
    #    user_weights.to_csv(counts_out_file)

    return classifications



#********************************************************************************
#################################################################################
#********************************************************************************

                            # Begin the main work

#********************************************************************************
#################################################################################
#********************************************************************************


if __name__ == '__main__':


    ap = ArgumentParser(description='Script to pLot TESS LCs for Zooniverse project')
    ap.add_argument('--sector', type=int, help='Sector to analyse')
    ap.add_argument('--limit', type=float, help='Sector to analyse')
    ap.add_argument('--interface', type=str)
    args = ap.parse_args()



    ###################### Define files and settings first ###################### 
    #############################################################################
    
    # DEFINE THE SECTOR THAT YOU WANT TO ANALYSE
    # ------------------------------------------
    sector = args.sector
    data_release = "Rel0{}".format(sector) # only needed when run on Nora's Computer.

    # ------------------------------------------
    
    if sector == 9:
        classfile_in = "/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/zooniverse_output/planet-hunters-tess-classifications_sec{}{}.h5".format(sector, args.interface) # input file
        classfile_in2 = "/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/zooniverse_output/planet-hunters-tess-classifications_sec{}{}.csv".format(sector, args.interface) # input file

    else:
        classfile_in = "/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/zooniverse_output/planet-hunters-tess-classifications_sec{}.h5".format(sector) # input file
        classfile_in2 = "/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/zooniverse_output/planet-hunters-tess-classifications_sec{}.csv".format(sector) # input file

    # --------

    if sector < 9:
        counts_out_file = '/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel0{}/analysis/class_counts_sector{}_fast.csv'.format(sector, sector)
        outpath = '/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel0{}/analysis'.format(sector)   
    elif sector == 9:
        counts_out_file = '/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel0{}/analysis_{}int/class_counts_sector{}_fast.csv'.format(sector, sector, args.interface)
        outpath = '/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel0{}/analysis_{}int'.format(sector, args.interface)
    else:
        counts_out_file = '/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel{}/analysis/class_counts_sector{}_fast.csv'.format(sector, sector)
        outpath = '/mnt/zfsusers/nora/kepler_share/kepler2/TESS/planethunters/Rel{}/analysis'.format(sector)

        
    
    
    #classfile_in = "/Users/Nora/Documents/research/TESS/planethunters/zooniverse_output/planet-hunters-tess-classifications_sec12.h5" # input file
    #counts_out_file = "/Users/Nora/Documents/research/TESS/planethunters/zooniverse_output/class_counts_sector_test_fast.csv"
    #outpath = "/Users/Nora/Documents/research/TESS/planethunters/zooniverse_output/"
    #outfile = '/planet_aggregations_sector_test_fast.csv'


    # if the folder doesn't exist, make it!
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    #################################################################################

    classification_df = pd.DataFrame()
    ## Master node
    ## -----------

    if mpi_rank == mpi_root:
    
        t00 = time.time()
        try:
            allclassifications = pd.read_hdf(classfile_in) # this step can take a few minutes for a big file
        except:
            print ("try csv")
            allclassifications = pd.read_csv(classfile_in2)

        # allclassifications = allclassifications[0:10000]

        t0 = time.time()
        print ("DF imported")
        sys.stdout.flush()
        print ("TIME to import the data: {}".format(t0-t00))
        sys.stdout.flush()
        
        # instead of using the whole data frame just cut into what we want - get this from the anaconda script 
        # sec = allclassifications[allclassifications["subject_data"].str.contains('"Sector":{}'.format(sector), case=False)]
        
        # split the pandas data frane into the number of chunks that we have nodes
        print ("splitting DF....")
        classifications_list = np.array_split(allclassifications, mpi_size) # this is now a list of smaller data frames
        nlc = len(classifications_list)
        
        print ("Done splitting DF....")
        
        ## Without MPI or running with a single node
        ## =========================================
        
        if (not with_mpi) or (mpi_size==1):
            print ('processing in {:d} chunks'.format(nlc))
            for f in classifications_list:
                res = extract_info(f,args)
                classification_df = pd.concat([classification_df, res])

        else:
            print ("not with MPI")
            ## Master node
            ## -----------
            
            if mpi_rank == 0:
                free_workers = list(range(1,mpi_size))
                active_workers = []
                n_finished_items = 0
        
                while classifications_list or active_workers:
                    ## Send a file
                    while classifications_list and free_workers:
                        f = classifications_list.pop()
        
                        w = free_workers.pop()
                        comm.send(f, dest=w, tag=0)
                        active_workers.append(w)
                        print('Sending chunk to worker {}'.format(w))
                    
                    ## Receive the results
                    for w in active_workers:
                        if comm.Iprobe(w, 2):
                            res = comm.recv(source=w, tag=2)
                
                            # put the data back into the main dataframe
                            classification_df = pd.concat([classification_df,res])
                            #print ("length 2 res {}".format(len(res)))
                            #print ("length 2 {}".format(len(classification_df)))
                            # once that's done free this worker
                            free_workers.append(w)
                            active_workers.remove(w)
                            n_finished_items += 1
                

                print('Worker {} FINISHED'.format(w))

                # before saving it remove the columns that we or longer want or need

                # classification_new = classification_df[['subject_ids', 'sector', 'weight', 'count', 'planet_classification', 'subject_type', 'candidate', 'TIC_ID', 'TIC_LC', 'SNR', 'min_pix', 'max_pix', 'min_time', 'max_time', 'sim','xvals', 'width','user_label','user_id','radius','temperature','Tmag', 'simloc', 'simwidth', 'workflow_version', 'correct_count', 'false_count', 'missed_count', 'numsimtransits']]

                # classification_new = classification_df[['subject_ids','sector']]
                try:
                    classification_new = classification_df.drop(columns=['annotation_json', 'annotations', 'created_at', 'created_day', 'live_project', 'metadata', 'subject_data', 'subject_json'])
                except:
                    classification_new = classification_df
                    print ("skip filt")

                print ("length after split:", len(classification_new))

                del classification_df
                
                NUMBINS = 84

                print ("pre hist 0 ")

                print (list(classification_new[classification_new.subject_type == True]['subject_ids'])[0:11])


                #i nsec = (np.array(all_DBcluster) > 0) & (np.array(all_DBcluster) <= 28)
                
                flat_lists, binboundarylists = plot_histogram_sys_init_per_camer(classification_new, args)

                #flat_list, binboundarylist = plot_histogram_sys_init(classification_new, args)
                #
                ## -----------
                #LIMIT = args.limit
                ## -----------

                ## now apply the specified cut-off limit to obtain a list of the times (in this case pixels) that we want to ommit.
                #
                ## for each chunk apply a function and return input in the pandas array based with the cut off values removed
                #
                #classifications = histogram_sys_correction(classification_new, flat_list, LIMIT,binboundarylist,args)
                classifications = classification_new
                #print ("post hist")
                #print (list(classifications[classifications.subject_type == True]['subject_ids'])[0:11])
                #print ("length after hist:", len(classifications))

                print ("START WEIGHTING")
                # Calculate the user weighting for each volunteer
                ##----------------------------------------------
                import itertools
                
                #true_incr = np.arange(3, 4, 0.5)    # 0.5,10.1, 1
                #missed_incr = np.arange(-3, 0, 1)
                #false_incr = np.arange(-3, 0, 1)
                
                true_incr = np.arange(3, 4, 0.5)    # 0.5,10.1, 1
                missed_incr = np.arange(-2, -1, 1)
                false_incr = np.arange(-2, -1, 1)

                combinations_list = []
                for c in itertools.product(*[true_incr, missed_incr, false_incr]):
                    combinations_list.append(c)

                def myfunction():
                    
                    return random.uniform(0, 1)
                    
                random.shuffle(combinations_list, myfunction)

                for combinations in combinations_list:
                    
                    outfile = '{}/planet_aggregations_sector{}_{}_{}_{}.csv'.format(outpath, sector, combinations[0], combinations[2], combinations[1])
                    
                    if not os.path.exists(outfile):

                        classifications2 = weight_func(classifications, combinations[0], combinations[2], combinations[1])
                        
                        print ("SAVING")
                        
                        pd.DataFrame(classifications2).to_csv(outfile)

                    else:
                        print ("skip")

                if n_finished_items < nlc:
                    print ('Failed on {} files'.format(nlc-n_finished_items))
            
                for w in free_workers:
                    comm.send(-1, dest=w, tag=0)


    ## Worker node
    ## -----------

    else:
        while True:
    
            filename = comm.recv(source=mpi_root, tag=0)
            
            if type(filename) == int:
                break
            #print ("TYPE2: {}".format(type(filename)))
            #sys.stdout.flush()

            res = extract_info(filename,args)
            comm.send(res, dest=mpi_root, tag=2)    

# end.

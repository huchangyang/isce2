#!/usr/bin/env python3
# Author: David Bekaert
# Hu Changyang, adopted from prepSlcALOS2.py for LT1 SLC


import os
import glob
import argparse
import shutil
import tarfile
import zipfile
from uncompressFile import uncompressfile
import xml.etree.ElementTree as ET

EXAMPLE = """example:
  prepSlcLT1.py -i download -o SLC -orbit orbits
"""

def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='Prepare LT1 SLC for processing (unzip/untar files, '
                                     'organize in date folders, generate script to unpack into isce formats).',
                                     epilog=EXAMPLE, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', dest='inputDir', type=str, required=True,
            help='directory with the downloaded SLC data')
    parser.add_argument('-o', '--output', dest='outputDir', type=str, required=False,
            help='output directory where data needs to be unpacked into isce format (for script generation).')

    parser.add_argument('-t', '--text_cmd', dest='text_cmd', type=str, default='source ~/.bash_profile;',
            help='text command to be added to the beginning of each line of the run files (default: %(default)s).')

    parser.add_argument('-p', '--polarization', dest='polarization', type=str,
            help='polarization in case if quad or full pol data exists (default: %(default)s).')

    parser.add_argument('-rmfile', '--rmfile', dest='rmfile',action='store_true', default=False,
            help='Optional: remove zip/tar/compressed files after unpacking into date structure '
                 '(default is to keep in archive folder)')
    
    parser.add_argument('-orbit', '--orbitfile', dest='orbitfile', type=str, default=None, required=False,
                        help='Optional: directory with the precise orbit file for LT1 SLC (default: %(default)s).')
    
    return parser


def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps = parser.parse_args(args = iargs)

    # parsing required inputs
    inps.inputDir = os.path.abspath(inps.inputDir)

    # parsing optional inputs
    if inps.outputDir:
        inps.outputDir = os.path.abspath(inps.outputDir)
    return inps


def get_Date(LT1_folder):
    """Grab acquisition date"""
    # will search for meta data of LT1 SLCS to get the acquisition date
    metadata_patterns = [
        os.path.join(LT1_folder, 'LT1A_*.meta.xml'),
        os.path.join(LT1_folder, 'LT1B_*.meta.xml')
    ]
    
    metadata = []
    for pattern in metadata_patterns:
        metadata.extend(glob.glob(pattern))
    
    # if nothing is found return a failure
    if len(metadata) > 0:
        for metadata_file in metadata:
            try:
                fp = open(metadata_file, 'r')
            except IOError as strerr:
                print("IOError: %s" % strerr)
                continue
        
            _xml_root = ET.ElementTree(file=fp).getroot()
            acquisitionDate = (str(_xml_root.find('.//productInfo/sceneInfo/sceneCenterCoord/azimuthTimeUTC').text)[:10].replace('-',''))
            if acquisitionDate:
                successflag = True
                fp.close()
                return successflag, acquisitionDate
            fp.close()
    
    # if it reached here it could not find the acquisitionDate
    successflag = False
    acquisitionDate = 'FAIL'
    return successflag, acquisitionDate


def get_LT1_name(infile):
    """Get the LT1 name from compress file in various format."""
    outname = None
    fbase = os.path.basename(infile)
    if 'LT1' in fbase:
        fbase = fbase.replace('_','-')
        # Remove common compression extensions
        outname = fbase.replace('.tar.gz', '').replace('.tar', '').replace('.zip', '').replace('.gz', '')
    else:
        fext = os.path.splitext(infile)[1]
        if fext in ['.tar', '.gz']:
            with tarfile.open(infile, 'r') as tar:
                file_list = tar.getnames()
        elif fext in ['.zip']:
            with zipfile.ZipFile(infile, 'r') as z:
                file_list = z.namelist()
        else:
            raise ValueError('unrecognized file extension: {}'.format(fext))
        meta_file = [i for i in file_list if 'meta.xml' in i][0]
        meta_file = os.path.basename(meta_file)
        outname = meta_file.replace('.meta.xml', '')
    return outname

def copy_orbit_file(orbit_dir, date, slc_dir):
    """Copy matching orbit file to SLC directory"""
    if not orbit_dir:
        return
    
    # search for orbit files containing the date (supports .txt and .xml formats)
    orbit_pattern = os.path.join(orbit_dir, f'*{date}*.txt')
    orbit_files = glob.glob(orbit_pattern)
    if not orbit_files:
        orbit_pattern = os.path.join(orbit_dir, f'*{date}*.xml')
        orbit_files = glob.glob(orbit_pattern)
    
    # if matching orbit files found, copy to SLC directory
    if orbit_files:
        for orbit_file in orbit_files:
            dest_file = os.path.join(slc_dir, os.path.basename(orbit_file))
            shutil.copy2(orbit_file, dest_file)
            print(f'Copied orbit file: {orbit_file} to {dest_file}')
    else:
        print(f'No matching orbit file found for date: {date}')


def main(iargs=None):
    '''
    The main driver.
    '''

    inps = cmdLineParse(iargs)

    # filename of the runfile
    run_unPack = 'run_unPackLT1'

    # loop over the different folder of LT1 zip/tar files and unzip them, make the names consistent
    file_exts = (os.path.join(inps.inputDir, '*.zip'),
                 os.path.join(inps.inputDir, '*.tar'),
                 os.path.join(inps.inputDir, '*.gz'))
    for file_ext in file_exts:
        # loop over zip/tar files
        for fname in sorted(glob.glob(file_ext)):
            ## the path to the folder/zip
            workdir = os.path.dirname(fname)

            ## get the output name folder without any extensions
            dir_unzip = get_LT1_name(fname)
            dir_unzip = os.path.join(workdir, dir_unzip)

            # loop over two cases (either file or folder): 
            # if this is a file, try to unzip/untar it
            if os.path.isfile(fname):
                # unzip the file in the outfolder
                successflag_unzip = uncompressfile(fname, dir_unzip)

                # put failed files in a seperate directory
                if not successflag_unzip:
                    dir_failed = os.path.join(workdir,'FAILED_FILES')
                    os.makedirs(dir_failed, exist_ok=True)
                    cmd = 'mv {} {}'.format(fname, dir_failed)
                    os.system(cmd)
                else:
                    # check if file needs to be removed or put in archive folder
                    if inps.rmfile:
                        os.remove(fname)
                        print('Deleting: ' + fname)
                    else:
                        dir_archive = os.path.join(workdir,'ARCHIVED_FILES')
                        os.makedirs(dir_archive, exist_ok=True)
                        cmd = 'mv {} {}'.format(fname, dir_archive)
                        os.system(cmd)


        # loop over the different LT1 folders and make sure the folder names are consistent.
        # this step is not needed unless the user has manually unzipped data before.
        LT1_folders = glob.glob(os.path.join(inps.inputDir, 'LT1*'))
        for LT1_folder in LT1_folders:
            # in case the user has already unzipped some files
            # make sure they are unzipped similar like the uncompressfile code
            temp = os.path.basename(LT1_folder)
            parts = temp.split(".")
            parts = parts[0].split('-')
            LT1_outfolder_temp = parts[0]
            LT1_outfolder_temp = os.path.join(os.path.dirname(LT1_folder),LT1_outfolder_temp)
            # check if the folder (LT1_folder) has a different filename as generated from uncompressFile (LT1_outfolder_temp)
            if not (LT1_outfolder_temp == LT1_folder):
                # it is different, check if the LT1_outfolder_temp already exists, if yes, delete the current folder
                if os.path.isdir(LT1_outfolder_temp):
                    print('Remove ' + LT1_folder + ' as ' + LT1_outfolder_temp + ' exists...')
                    # check if this folder already exist, if so overwrite it
                    shutil.rmtree(LT1_folder)


    # loop over the different LT1 folders and organize in date folders
    LT1_folders = glob.glob(os.path.join(inps.inputDir, 'LT1*'))                        
    for LT1_folder in LT1_folders:
        # get the date
        successflag, imgDate = get_Date(LT1_folder)       

        workdir = os.path.dirname(LT1_folder)
        if successflag:
            ## stripmapStack: YYYYMMDD/LT1*/LT1*.tiff
            # create the date folder
            SLC_dir = os.path.join(workdir,imgDate)
            os.makedirs(SLC_dir, exist_ok=True)

            # check if the folder already exist in that case overwrite it
            LT1_folder_out = os.path.join(SLC_dir,os.path.basename(LT1_folder))
            if os.path.isdir(LT1_folder_out):
                shutil.rmtree(LT1_folder_out)
                
            # copy orbit file to SLC directory
            copy_orbit_file(inps.orbitfile, imgDate, SLC_dir)

            # move the LT1 acqusition folder in the date folder
            cmd = 'mv ' + LT1_folder + '/* ' + SLC_dir
            os.system(cmd)
            os.rmdir(LT1_folder)
            print ('Succes: ' + imgDate)
        else:
            print('Failed: ' + LT1_folder)


    # now generate the unpacking script for all the date dirs
    dateDirs = sorted(glob.glob(os.path.join(inps.inputDir,'2*')))
    if inps.outputDir is not None:
        f = open(run_unPack,'w')
        for dateDir in dateDirs:
            LT1Files = glob.glob(os.path.join(dateDir, 'LT1*'))
            # if there is at least one frame
            if len(LT1Files)>0:
                acquisitionDate = os.path.basename(dateDir)
                slcDir = os.path.join(inps.outputDir, acquisitionDate)
                os.makedirs(slcDir, exist_ok=True)
                cmd = 'unpackFrame_LT1.py -i ' + os.path.abspath(dateDir) + ' -o ' + slcDir
                if inps.polarization:
                    cmd += ' --polarization {} '.format(inps.polarization)
                if inps.orbitfile:
                    cmd += ' --orbitfile ' + inps.orbitfile
                print (cmd)
                f.write(inps.text_cmd + cmd+'\n')
        f.close()
    return


if __name__ == '__main__':

    main()

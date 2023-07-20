"""
convert_meshes.py
Process the .obj files and make them waterproof and simplified.

From: https://github.com/NVlabs/acronym/issues/6
"""
import os
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

GRASPDIR = "./grasps"
OBJDIR = "./"
MANIFOLD_PATH = "./Manifold/build/manifold"
SIMPLIFY_PATH = "./Manifold/build/simplify"

## Grab the object file names from the grasp directory
hashes = []
for fname in os.listdir(GRASPDIR):
    tokens = fname.split("_")
    assert(len(tokens) == 3)
    hashes.append(tokens[1])

REGENENERATE = [
    '49a0a84ee5a91344b11c5cce13f76151',
    'feb146982d0c64dfcbf4f3f04bbad8',
    '202fd2497d2e85f0dd6c14adedcbd4c3',
    '47fc4a2417ca259f894dbff6c6a6be89',
    '2a7d62b731a04f5fa54b9afa882a89ed',
    '8123f469a08a88e7761dc3477e65a72',

    '202fd2497d2e85f0dd6c14adedcbd4c3'
    'feb146982d0c64dfcbf4f3f04bbad8',
    '47fc4a2417ca259f894dbff6c6a6be89',
    'feb146982d0c64dfcbf4f3f04bbad8',
    '2a7d62b731a04f5fa54b9afa882a89ed',
    'b54e412afd42fe6da7c64d6a7060b75b',


]

FAILED = [
    '330880146fd858e2cb6782d0f0acad78',
    'a24e0ea758d43657a5e3e028709e0474',
    '37d510f7958b941b9163cfcfcc029614',
    'd25b80f268f5731449c3792a0dc29860',
]

## Define function to process a single file
def process_hash(h):
    """Process a single object file by calling a subshell with the mesh processing script.

    Args:
        h (string): the hash denoting the file type
    """
    obj = OBJDIR + 'models/' + h + ".obj"
    temp_name = f"temp.{h}.watertight.obj"
    outfile = OBJDIR + "meshes/" + h + ".obj"
    
    
    if h in REGENENERATE:
        print(f'Regenerating: {h}')
    else:
        # File already done
        if os.path.isfile(outfile):
            # print(f'{outfile} already done')
            # Get rid of existing temp files
            if os.path.isfile(temp_name):
                os.remove(temp_name)
                print(f'removed: {temp_name}')
            return

        if not os.path.isfile(obj):
            return
    
    # Waterproof the object
    completed = subprocess.CompletedProcess(args=[], returncode=0)
    if not os.path.isfile(temp_name):
        completed = subprocess.run(["timeout", "-sKILL", "30", MANIFOLD_PATH, obj, temp_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    watertight_conversion = completed.returncode
    if completed.returncode != 0:
        print(f"Skipping object (manifold failed): {h}")
        return
            
    # Simplify the object
    completed = subprocess.run([SIMPLIFY_PATH, "-i", temp_name, "-o", outfile, "-m", "-r", "0.02"],  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # if os.path.isfile(temp_name):
    #     os.remove(temp_name)
        # print(f'removed: {temp_name}')

    if not os.path.isfile(outfile):
        print('Simplify failed, saving unsimplified mesh anyway')
        print(f'failed mesh: {outfile}')
        os.rename(temp_name, outfile)
        
out_name = 'meshes'
if not os.path.exists(os.path.join(OBJDIR, out_name)):
        raise ValueError('Make simplified dir')

## Issue the commands in a multiprocessing pool
with Pool(cpu_count()-2) as p:
    examples = list(
        tqdm(
            p.imap_unordered(process_hash, hashes),
            total=len(hashes)
        )
    )

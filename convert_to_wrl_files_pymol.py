import pymol
from pymol import cmd, preset
from biopandas.pdb import PandasPdb
import pymol
from pymol import cmd, preset
import glob
import os

import bpy
import mathutils

import os

# Useful page:
#    https://pymol.org/dokuwiki/doku.php?id=api:cmd:alpha
# https://pymolwiki.org/index.php/Selection_Algebra

# Use pymol -c option to run in headless mode.

# Wait for pymol to be ready
pymol.finish_launching()
filepath_input = '6w75.pdb'


class Confs(object):
    def __init__(self, **kwargs):
        self.filepath = kwargs.get('filepath')
        self.protein_surface = kwargs.get('protein_surface', False)
        self.protein_ribbon = kwargs.get("protein_ribbon", True)
        self.protein_sticks = kwargs.get("protein_sticks", False)

        self.protein_balls = kwargs.get("protein_balls", False)

        self.protein_vdw = kwargs.get("protein_vdw", False)

        self.ligand_surface = kwargs.get("ligand_surface", False)
        self.ligand_sticks = kwargs.get("ligand_sticks", True)
        self.ligand_balls = kwargs.get("ligand_balls", False)
        self.ligand_vdw = kwargs.get("ligand_vdw", False)

        self.near_ligand_surface = kwargs.get("near_ligand_surface", False)
        self.near_ligand_sticks = kwargs.get("near_ligand_sticks", False)
        self.near_ligand_balls = kwargs.get("near_ligand_balls", False)
        self.near_ligand_vdw = kwargs.get("near_ligand_balls", False)

        self.metals_vdw = kwargs.get("metals_vdw", False)

        self.remove_doubles = kwargs.get("remove_doubles", True)
        self.nanometers = kwargs.get("nanometers", False)

        self.vmd_exec_path = kwargs.get("vmd_exec_path", '')
        self.pymol_exec_path = kwargs.get("pymol_exec_path", '')
        self.prefer_vmd = kwargs.get("prefer_vmd", True)

        self.vmd_msms_repr = kwargs.get("vmd_msms_repr", True)


confs = Confs(filepath=filepath_input)


def reset_camera():
    # Make sure the camera is set to a neutral
    # position/orientation, so mesh matches pdb coordinates.
    cmd.set_view([1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0,
                  0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0,
                  40.0, 100.0, -20.0])


def split_object(target_object=None, source_object=None, max_iter=500, quiet=1, _self=cmd):
    """
DESCRIPTION

    Splits a multi-molecular object into one multi-state object

ARGUMENTS

    target_obj

        (string) name of target object

    source_obj

        (string) name of source object

    max_iter

        (int) maximum number of object to process; set to 0 to unlimit

    """
    if source_object is None:
        print(
            "Error: Please provide a source object.")
        return

    # ensure the user gave us one object; save for prefix

    obj_list = _self.get_object_list(target_object)

    if len(obj_list) > 1:
        print(
            " Error: Please provide only one object at a time.")
        return

    if target_object == None:
        target_object = _self.get_unused_name(source_object, alwaysnumber=0)

    # grab unused selection name

    s = cmd.get_unused_name("_store")

    # make original selection which we'll pare down

    cmd.select(s, source_object)

    count = 0

    while cmd.count_atoms(s) and count < max_iter:
        count += 1

        # create the object from the first molecular
        # object inside pfx
        cmd.create(pfx, "bm. first " + s, 1, count)

        # remove the first molecular object from
        # the source selection and re-iterate
        cmd.select(s, "%s and not bm. first %s" % (s, s))

    if not quiet:
        print(
            " Created new object %s." % target_object)


def reset_camera():
    # Make sure the camera is set to a neutral
    # position/orientation, so mesh matches pdb coordinates.
    cmd.set_view([1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0,
                  0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0,
                  40.0, 100.0, -20.0])
cmd.extend("split_object", split_object)
reset_camera()
with open('rcsb_pdb_ids_20200628065205.txt', 'r') as file:
    ids = file.read().split(',')

for id in ids:

    ppdb = PandasPdb().fetch_pdb(id)

    compounds = [i.strip().split(' ')[0] for i in
                 ppdb.df['OTHERS'][ppdb.df['OTHERS']['record_name'] == 'HETNAM']['entry']]
    pymol.finish_launching()
    cmd.fetch(id)
    chains = cmd.get_chains(selection="(all)", state=0, quiet=1)

    cmd.frame(0)
    for comp in compounds:

        all_protein_nuc_sel = "resn ala+arg+asn+asp+asx+cys+gln+glu+glx+gly+his+hsp+hyp+ile+leu+lys+met+pca+phe+pro+ser+thr+trp+tyr+val+dg+dc+dt+da+g+c+t+a+u+rg+rc+rt+ra+ru"
        water_sel = "resn wat+hoh+h2o+tip+tip3"
        all_metals_sel = "symbol " + "+".join(["fe", "ag", "co", "cu",
                                               "ca", "zn", "mg", "ni",
                                               "mn", "au"])

        cmd.save('to_convert/all_{}.wrl'.format(id))
        for chain in [c for c in chains if c != ""]:
            ligand_sel = "(chain " + chain + ") and (not " + all_protein_nuc_sel + " and not " + water_sel + ") and not (not symbol h+he+li+be+b+c+n+o+f+ne+na+mg+al+si+p+s+se+cl+br+f) and (not resn mse) and" + '(resn ' + comp.lower() + ')'
            protein_nuc_sel = "(chain " + chain + ") and (" + all_protein_nuc_sel + ",mse)"
            ligand_and_protein = "(chain " + chain + ") and (" + all_protein_nuc_sel + "+" + comp.lower() + ",mse)"
            cmd.hide("(all)")

            protein_and_ligand = (ligand_sel + " and " + protein_nuc_sel)
            cmd.select("protein_and_ligand{}".format(chain), ligand_and_protein)
            cmd.select("ligand_sel{}".format(chain), ligand_sel)
            cmd.select("protein_nuc_sel{}".format(chain), protein_nuc_sel)

            preset.ligand_cartoon(selection="protein_and_ligand{}".format(chain))
            reset_camera()
            cmd.save('to_convert/{}_protein_ligand_and_bond{}.wrl'.format(id, chain))
            cmd.hide("(" + "protein_nuc_sel{}".format(chain) + ")")
            cmd.save('to_convert/{}_ligand_bonds{}.wrl'.format(id, chain))

            preset.ligand_cartoon(selection="protein_and_ligand{}".format(chain))
            cmd.hide("(" + "ligand_sel{}".format(chain) + ")")
            preset.default(selection='all')
            cmd.hide("(all)")
            cmd.show("cartoon", "protein_nuc_sel{}".format(chain))
            reset_camera()
            cmd.save('to_convert/{}_protein{}.wrl'.format(id, chain))

            cmd.hide("(all)")
            cmd.show("sticks", "ligand_sel{}".format(chain))
            reset_camera()
            cmd.save('to_convert/{}_ligand{}.wrl'.format(id, chain))

            cmd.hide("(all)")
            print('e')

        cmd.reset()

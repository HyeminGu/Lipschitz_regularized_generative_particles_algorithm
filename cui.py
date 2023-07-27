#!/usr/bin/env python3

from sys import argv
import os

if "plot_result" in argv[1]:
    run_script ="python3 scripts/util/plot_result.py {filename}".format(filename=argv[2])
else:
    from scripts.util.input_args import input_params
    p, _ = input_params()
        
    run_script ="python3 scripts/{model_dir}/{model}.py".format(model_dir = p.generative_model, model=p.generative_model)
    
    for keybord_parameters in argv[1:]:
        run_script += " " + keybord_parameters

os.system(run_script)

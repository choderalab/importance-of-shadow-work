#!/usr/bin/env python


from simtk import unit
from openmmtools import mcmc
from yank.yamlbuild import YamlBuilder


def run_yank(yank_script_template_filepath):
    # Read in YAML script template. We will use it to set
    # the output folder for each experiment.
    with open(yank_script_template_filepath, 'r') as f:
        script_template = f.read()

    # Generate all combinations.
    for timestep in [1.5]*unit.femtosecond:
        for mcmc_move in [mcmc.LangevinDynamicsMove(timestep=timestep)]:

            # Create unique name name for the output of the experiments.
            # The " * 10" is just to avoid having dots in the directory name.
            experiment_dir = 'timestep{}-{}'.format(int(timestep * 10 / unit.femtosecond),
                                                    mcmc_move.__class__.__name__)
            script = script_template.format(experiments_dir=experiment_dir)

            # Build experiment for all molecules in the YAML script.
            yamlbuilder = YamlBuilder(script)
            for experiment in yamlbuilder.build_experiments():

                # Set the MCMC Move (YANK's default is LangevinDynamics).
                for phase in experiment.phases:
                    phase.sampler.mcmc_moves = mcmc_move

                # Run experiment.
                experiment.run()



if __name__ == '__main__':
    run_yank('freesolv_template.yaml')

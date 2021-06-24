import simtk.openmm as openmm
import simtk.unit as unit
from openmmtools.integrators import LangevinIntegrator, AlchemicalNonequilibriumLangevinIntegrator
import progressbar
from perses.tests.utils import createSystemFromIUPAC, get_data_filename
from perses.annihilation.new_relative import HybridTopologyFactory
from perses.rjmc.topology_proposal import TopologyProposal, SmallMoleculeSetProposalEngine, SystemGenerator
from perses.rjmc.geometry import FFAllAngleGeometryEngine
import openeye.oechem as oechem
import numpy as np


kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT

def simulate_hybrid(hybrid_system,functions, lambda_value, boxvec, positions, nsteps=5000, timestep=1.0*unit.femtoseconds, temperature=temperature, collision_rate=5.0/unit.picoseconds, splitting="V R O R V", platform_name="OpenCL"):
    platform = openmm.Platform.getPlatformByName(platform_name)
    integrator = LangevinIntegrator(temperature=temperature, timestep=timestep, splitting=splitting)
    context = openmm.Context(hybrid_system, integrator, platform)
    context.setPeriodicBoxVectors(*boxvec)
    for parameter in functions.keys():
        context.setParameter(parameter, lambda_value)
    context.setPositions(positions)
    integrator.step(nsteps)
    state = context.getState(getPositions=True)
    positions = state.getPositions()
    boxvec = state.getPeriodicBoxVectors()
    return positions, boxvec

def generate_solvated_hybrid_test_topology(mol_name="naphthalene", ref_mol_name="benzene"):

    import simtk.openmm.app as app
    from openmoltools import forcefield_generators

    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC, get_data_filename

    m, unsolv_old_system, pos_old, top_old = createSystemFromIUPAC(mol_name)
    refmol = createOEMolFromIUPAC(ref_mol_name)

    initial_smiles = oechem.OEMolToSmiles(m)
    final_smiles = oechem.OEMolToSmiles(refmol)
    pressure = 1.0 * unit.atmosphere
    barostat_period=50
    barostat = openmm.MonteCarloBarostat(pressure, temperature, barostat_period)
    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    modeller = app.Modeller(top_old, pos_old)
    modeller.addSolvent(forcefield, model='tip3p', padding=9.0*unit.angstrom)
    solvated_topology = modeller.getTopology()
    solvated_positions = modeller.getPositions()
    solvated_system = forcefield.createSystem(solvated_topology, nonbondedMethod=app.PME, removeCMMotion=False)

    gaff_filename = get_data_filename('data/gaff.xml')
    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'])
    geometry_engine = FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [initial_smiles, final_smiles], system_generator, residue_name=mol_name)

    #generate topology proposal
    topology_proposal = proposal_engine.propose(solvated_system, solvated_topology)

    #generate new positions with geometry engine
    new_positions, _ = geometry_engine.propose(topology_proposal, solvated_positions, beta)

    return topology_proposal, solvated_positions, new_positions

def generate_vacuum_hybrid_topology(mol_name="naphthalene", ref_mol_name="benzene"):
    import simtk.openmm.app as app
    from openmoltools import forcefield_generators

    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC, get_data_filename

    m, unsolv_old_system, pos_old, top_old = createSystemFromIUPAC(mol_name)
    refmol = createOEMolFromIUPAC(ref_mol_name)

    initial_smiles = oechem.OEMolToSmiles(m)
    final_smiles = oechem.OEMolToSmiles(refmol)

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    solvated_system = forcefield.createSystem(top_old, removeCMMotion=False)

    gaff_filename = get_data_filename('data/gaff.xml')
    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'])
    geometry_engine = FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [initial_smiles, final_smiles], system_generator, residue_name=mol_name)

    #generate topology proposal
    topology_proposal = proposal_engine.propose(solvated_system, top_old)

    #generate new positions with geometry engine
    new_positions, _ = geometry_engine.propose(topology_proposal, pos_old, beta)

    return topology_proposal, pos_old, new_positions

def check_alchemical_hybrid_elimination_bar(topology_proposal, old_positions, new_positions, ncmc_nsteps=50, n_iterations=50, platform_name="OpencL", splitting_string="V R O H O R V", eq_splitting_string="V R O R V", timestep=1.0*unit.femtosecond):

    #make the hybrid topology factory:
    factory = HybridTopologyFactory(topology_proposal, old_positions, new_positions)

    platform = openmm.Platform.getPlatformByName(platform_name)

    hybrid_system = factory.hybrid_system
    hybrid_topology = factory.hybrid_topology
    initial_hybrid_positions = factory.hybrid_positions

    outfile_prefix = splitting_string.replace(" ", "")
    #alchemical functions
    forward_functions = {
        'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
        'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
        'lambda_bonds' : 'lambda',
        'lambda_angles' : 'lambda',
        'lambda_torsions' : 'lambda'
    }

    reverse_functions = {param : param_formula.replace("lambda", "(1-lambda)") for param, param_formula in forward_functions.items()}


    w_f = np.zeros(n_iterations)
    w_r = np.zeros(n_iterations)

    sw_f = np.zeros(n_iterations)
    sw_r = np.zeros(n_iterations)

    #make the alchemical integrators:
    forward_integrator = AlchemicalNonequilibriumLangevinIntegrator(alchemical_functions=forward_functions, splitting=splitting_string, nsteps_neq=ncmc_nsteps, measure_shadow_work=True, timestep=timestep)
    forward_context = openmm.Context(hybrid_system, forward_integrator, platform)

    print("Minimizing for forward protocol...")
    forward_context.setPositions(initial_hybrid_positions)
    for parm in forward_functions.keys():
        forward_context.setParameter(parm, 0.0)

    openmm.LocalEnergyMinimizer.minimize(forward_context, maxIterations=10)
    
    boxvec = hybrid_system.getDefaultPeriodicBoxVectors()
    initial_state = forward_context.getState(getPositions=True, getEnergy=True)
    print("The initial energy after minimization is %s" % str(initial_state.getPotentialEnergy()))
    initial_forward_positions = initial_state.getPositions(asNumpy=True)
    equil_positions, equil_boxvec = simulate_hybrid(hybrid_system, forward_functions, 0.0, boxvec, initial_forward_positions, platform_name=platform_name, timestep=timestep, splitting=eq_splitting_string)

    print("Beginning forward protocols")
    #first, do forward protocol (lambda=0 -> 1)
    with progressbar.ProgressBar(max_value=n_iterations) as bar:
        for i in range(n_iterations):
            equil_positions, equil_boxvec = simulate_hybrid(hybrid_system, forward_functions, 0.0, equil_boxvec, equil_positions, platform_name=platform_name,timestep=timestep, splitting=eq_splitting_string)
            forward_context.setPositions(equil_positions)
            forward_context.setPeriodicBoxVectors(*equil_boxvec)
            forward_integrator.step(ncmc_nsteps)
            w_f[i] = forward_integrator.get_protocol_work(dimensionless=True)
            sw_f[i] = forward_integrator.get_shadow_work(dimensionless=True)
            forward_integrator.reset()
            bar.update(i)

    del forward_context, forward_integrator

    reverse_integrator = AlchemicalNonequilibriumLangevinIntegrator(alchemical_functions=reverse_functions, splitting=splitting_string, nsteps_neq=ncmc_nsteps, timestep=timestep, measure_shadow_work=True)

    print("Minimizing for reverse protocol...")
    reverse_context = openmm.Context(hybrid_system, reverse_integrator, platform)
    reverse_context.setPositions(initial_hybrid_positions)
    for parm in reverse_functions.keys():
        reverse_context.setParameter(parm, 1.0)
    openmm.LocalEnergyMinimizer.minimize(reverse_context, maxIterations=10)
    initial_state = reverse_context.getState(getPositions=True, getEnergy=True)
    print("The initial energy after minimization is %s" % str(initial_state.getPotentialEnergy()))
    initial_reverse_positions = initial_state.getPositions(asNumpy=True)
    equil_positions, equil_boxvec = simulate_hybrid(hybrid_system, reverse_functions, 1.0, equil_boxvec, initial_reverse_positions, nsteps=1000, platform_name=platform_name, timestep=timestep, splitting=eq_splitting_string)

    #now, reverse protocol
    print("Beginning reverse protocols...")
    with progressbar.ProgressBar(max_value=n_iterations) as bar:
        for i in range(n_iterations):
            equil_positions, equil_boxvec = simulate_hybrid(hybrid_system, reverse_functions, 1.0, equil_boxvec, equil_positions, platform_name=platform_name, timestep=timestep, splitting=eq_splitting_string)
            reverse_context.setPositions(equil_positions)
            reverse_context.setPeriodicBoxVectors(*equil_boxvec)
            reverse_integrator.step(ncmc_nsteps)
            w_r[i] = reverse_integrator.get_protocol_work(dimensionless=True)
            sw_r[i] = reverse_integrator.get_shadow_work(dimensionless=True)
            reverse_integrator.reset()
            bar.update(i)
    del reverse_context, reverse_integrator

    from pymbar import BAR
    [df, ddf] = BAR(w_f, w_r)
    print("df = %12.6f +- %12.5f kT" % (df, ddf))

    outfile_name = outfile_prefix + str(timestep.value_in_unit(unit.femtosecond)) + ".npy"

    output_data = {"reverse_protocol_work":  w_r, "reverse_shadow_work" : sw_r, "forward_protocol_work" : w_f, "forward_shadow_work" : sw_f}

    np.save(outfile_name, output_data)

if __name__=="__main__":
    import sys
    array_index = int(sys.argv[1])
    array_index -= 1
    allowed_timesteps = [0.5, 1.0, 1.5, 2.0, 2.5]
    if array_index < 5:
        eq_splitting_string = "V R O R V"
        ne_splitting_string = "V R O H O R V"
    else:
        eq_splitting_string = "R V O V R"
        ne_splitting_string = "R V O H O V R"
    timestep = allowed_timesteps[array_index % 5]
    init_objects = np.load("/cbio/jclab/home/pgrinaway/initobj.npy")
    top_pos = init_objects.item()
    topology_proposal = top_pos['top_prop']
    old_positions = top_pos['old_positions']
    new_positions = top_pos['new_positions']
    ts_with_units = timestep*unit.femtoseconds
    check_alchemical_hybrid_elimination_bar(topology_proposal, old_positions, new_positions, ncmc_nsteps=10000, n_iterations=100, platform_name="CUDA", timestep=ts_with_units, splitting_string=ne_splitting_string, eq_splitting_string=eq_splitting_string)
    #check_alchemical_hybrid_elimination_bar(topology_proposal_vac, old_positions_vac, new_positions_vac, ncmc_nsteps=100, n_iterations=50, platform_name="Reference")

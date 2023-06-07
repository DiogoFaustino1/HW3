# -*- coding: utf-8 -*-
"""
Assignment 3 - Problem 3 b) iv) 

 ========================================================================
   Instituto Superior Técnico - Aircraft Optimal Design - 2023
   
   96375 Filipe Valquaresma
   filipevalquaresma@tecnico.ulisboa.pt
   
   95782 Diogo Faustino
   diogovicentefaustino@tecnico.ulisboa.pt
 ========================================================================
"""

import numpy as np

import openmdao.api as om

from openaerostruct.geometry.utils           import generate_mesh
from openaerostruct.geometry.geometry_group  import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

# Create a dictionary to store options about the mesh
mesh_dict = {"num_y" : 101, # spanwise
             "num_x" : 5, # chordwise
             "wing_type" : "rect",
             "symmetry" : True,  # computes left half-wing only
             "num_chord_cp" : 2,
             "num_twist_cp" : 5
             } 

# Generate the aerodynamic mesh based on the previous dictionary
mesh = generate_mesh(mesh_dict)

# Create a dictionary with info and options about the aerodynamic
# lifting surface
surface = {
           # Wing definition
           "name" : "wing",         # name of the surface
           "symmetry" : True,       # if true, model one half of wing
                                    # reflected across the plane y = 0
           "S_ref_type": "projected",  # how we compute the wing area,
                                    # can be 'wetted' or 'projected'
           "span" : 11.0,
           "root_chord" : (16.2/11.0),
           "fem_model_type" : "tube",
           #"sweep" : 0,
           #"taper" : 1,
           "twist_cp" : np.zeros(10),
           "chord_cp" : np.ones(10),
           "mesh" : mesh,
           #"dor" : 0,
           # Aerodynamic performance of the lifting surface at
           # an angle of attack of 0 (alpha=0).
           # These CL0 and CD0 values are added to the CL and CD
           # obtained from aerodynamic analysis of the surface to get
           # the total CL and CD.
           # These CL0 and CD0 values do not vary wrt alpha.
           "CL0" : 0.0,    # CL of the surface at alpha=0
           "CD0" : 0.015,  # CD of the surface at alpha=0
           # Airfoil properties for viscous drag calculation
           "k_lam" : 0.05,  # percentage of chord with laminar
                            # flow, used for viscous drag
           "t_over_c_cp" : np.array([0.12, 0.08, 0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03]),
                         # thickness over chord ratio (2412)
           "c_max_t" : 0.3,  # chordwise location of maximum (NACA2412)
                             # thickness
           "with_viscous" : True,  # if true, compute viscous drag
           "with_wave" : False,    # if true, compute wave drag
}

# Create the OpenMDAO problem
prob = om.Problem()

# Create an independent variable component that will supply the flow
# conditions to the problem.
indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=63, units="m/s")
indep_var_comp.add_output("alpha", val=5.0, units="deg")
# indep_var_comp.add_output("Mach_number", val=0.84)
# indep_var_comp.add_output("re", val=1.0e6, units="1/m")
indep_var_comp.add_output("rho", val=1.00649, units="kg/m**3") # https://aerotoolbox.com/atmcalc/ assuming T_offset = 0
indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")

# Add this IndepVarComp to the problem model
prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

# Create and add a group that handles the geometry for the
# aerodynamic lifting surface
geom_group = Geometry(surface=surface)
prob.model.add_subsystem(surface["name"], geom_group)

# Create the aero point group, which contains the actual aerodynamic
# analyses
aero_group = AeroPoint(surfaces=[surface])
point_name = "aero_point_0"
prob.model.add_subsystem(point_name, aero_group,
                         promotes_inputs=["v", "alpha", "rho", "cg"]
)

name = surface["name"]

# Connect the mesh from the geometry component to the analysis point
prob.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")

# Perform the connections with the modified names within the
# 'aero_states' group.
prob.model.connect(name + ".mesh", point_name + ".aero_states." + name + "_def_mesh")

prob.model.connect(name + ".t_over_c", point_name + "." + name + "_perf." + "t_over_c")

# Import the Scipy Optimizer and set the driver of the problem to use
# it, which defaults to an SLSQP optimization method
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["tol"] = 1e-9

recorder = om.SqliteRecorder("aero.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options['record_derivatives'] = True
prob.driver.recording_options['includes'] = ['*']

# Setup problem and add design variables, constraint, and objective
prob.model.add_design_var("wing.chord_cp", lower=-0.001, upper=20)
prob.model.add_design_var("wing.twist_cp", lower=-50.0, upper=50.0)
prob.model.add_design_var("alpha", lower=-50.0, upper=50.0)
prob.model.add_constraint(point_name + ".wing_perf.CL", equals=0.5)
prob.model.add_objective(point_name + ".wing_perf.CD", scaler=1e4)

# Set up and run the optimization problem
prob.setup()

# (extra example, automatically reported by default)
#from openmdao.api import n2
#n2(prob)

# Perform optimization
prob.run_driver()

# Output some results
# print("Twist Distrbution =", prob['wing.twist_cp'][0])
print("alpha =", prob['aero_point_0.alpha'][0])
print("C_D =", prob['aero_point_0.wing_perf.CD'][0])
print("C_L =", prob['aero_point_0.wing_perf.CL'][0])
print("CM position =", prob['aero_point_0.CM'][1])

# Clean up
prob.cleanup()
# -*- coding: utf-8 -*-
"""
Assignment 3 - Mesh Convergence Study - CPU time (efficiency) comparison 

 ========================================================================
   Instituto Superior Técnico - Aircraft Optimal Design - 2023
   
   96375 Filipe Valquaresma
   filipevalquaresma@tecnico.ulisboa.pt
   
   95782 Diogo Faustino
   diogovicentefaustino@tecnico.ulisboa.pt
 ========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint
import time

# Create a dictionary to store options about the mesh
mesh_dict = {
    "num_y": 0,  # Vary the values of num_y
    "num_x": 0,  # Vary the values of num_x
    "wing_type": "rect",
    "symmetry": True,
    #"num_twist_cp": 10
}

# Create empty lists to store the values of num_x, num_y, C_D, and total iterations
num_y_values = [5, 7, 11, 21, 25, 51, 101, 201, 301, 401, 501]
C_D_values = []
CPU_times = []

for iy in range(len(num_y_values)):
    # Update the mesh dictionary with the current values of num_x and num_y
    num_y = num_y_values[iy]
    num_x = 5
    mesh_dict["num_y"] = num_y
    mesh_dict["num_x"] = num_x
    
    start = time.time()
    # Generate the aerodynamic mesh based on the updated mesh dictionary
    mesh = generate_mesh(mesh_dict)

    # Create a dictionary with info and options about the aerodynamic lifting surface
    surface = {
        # Wing definition
        "name": "wing",  # name of the surface
        "symmetry": True,  # if true, model one half of wing
        # reflected across the plane y = 0
        "S_ref_type": "projected",  # how we compute the wing area,
        # can be 'wetted' or 'projected'
        "span": 11.0,
        "root_chord": (16.2 / 11.0),
        "fem_model_type": "tube",
        #"sweep" : 0,
        #"taper" : 1,
        #"twist_cp" : [0,0,0,0,0,0,0,0,0,0],
        "mesh": mesh,

        # Aerodynamic performance of the lifting surface at
        # an angle of attack of 0 (alpha=0).
        # These CL0 and CD0 values are added to the CL and CD
        # obtained from aerodynamic analysis of the surface to get
        # the total CL and CD.
        # These CL0 and CD0 values do not vary wrt alpha.
        "CL0": 0.0,  # CL of the surface at alpha=0
        "CD0": 0.015,  # CD of the surface at alpha=0
        # Airfoil properties for viscous drag calculation
        "k_lam": 0.05,  # percentage of chord with laminar
        # flow, used for viscous drag
        "t_over_c_cp": np.array([0.12, 0.08, 0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03]),
        # thickness over chord ratio (2412)
        "c_max_t": 0.3,  # chordwise location of maximum (NACA2412)
        # thickness
        "with_viscous": True,  # if true, compute viscous drag
        "with_wave": False,  # if true, compute wave drag
    }

    # Create the OpenMDAO problem
    prob = om.Problem()

    # Create an independent variable component that will supply the flow conditions to the problem
    indep_var_comp = om.IndepVarComp()
    indep_var_comp.add_output("v", val=63, units="m/s")
    indep_var_comp.add_output("alpha", val=5.0, units="deg")
    indep_var_comp.add_output("rho", val=1.00649, units="kg/m**3")
    indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")
    prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

    # Create and add a group that handles the geometry for the aerodynamic lifting surface
    geom_group = Geometry(surface=surface)
    prob.model.add_subsystem(surface["name"], geom_group)

    # Create the aero point group, which contains the actual aerodynamic analyses
    aero_group = AeroPoint(surfaces=[surface])
    point_name = "aero_point_0"
    prob.model.add_subsystem(point_name, aero_group, promotes_inputs=["v", "alpha", "rho", "cg"])

    name = surface["name"]
    prob.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")
    prob.model.connect(name + ".mesh", point_name + ".aero_states." + name + "_def_mesh")
    prob.model.connect(name + ".t_over_c", point_name + "." + name + "_perf." + "t_over_c")

    prob.setup()
    prob.run_driver()
    end = time.time()

    # Extract the value of C_D and CPU time
    C_D = prob[point_name + ".wing_perf.CD"]
    CPU_time = end-start

    # Store the values in the lists
    C_D_values.append(C_D)
    CPU_times.append(CPU_time)

# Plotting the results
# fig, ax = plt.subplots(2, 1, figsize=(8, 10))


# Plot total iterations versus num_y

fig, ax = plt.subplots()
ax.plot(num_y_values, CPU_times, marker='o')
ax.set_xlabel('num_y')
ax.set_ylabel('CPU time')
ax.set_title('Variation of CPU time with num_y')
plt.show()
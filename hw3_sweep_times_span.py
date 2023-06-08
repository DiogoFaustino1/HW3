# -*- coding: utf-8 -*-
"""
Assignment 3 - Problem 3 c) - Function Sweep times Span 

 ========================================================================
   Instituto Superior Técnico - Aircraft Optimal Design - 2023
   
   96375 Filipe Valquaresma
   filipevalquaresma@tecnico.ulisboa.pt
   
   95782 Diogo Faustino
   diogovicentefaustino@tecnico.ulisboa.pt
 ========================================================================
"""

import openmdao.api as om

class SweepTimesSpan(om.ExplicitComponent):
    """
    Calculate sweep times span as an aerodynamic function.

    Parameters
    ----------
    sweep : float
        Wing sweep angle in degrees.
    span : float
        Wing span in meters.

    Returns
    -------
    sweep_times_span : float
        Sweep times span value.

    """

    def setup(self):
        self.add_input("sweep", val=1.0, units="deg")
        self.add_input("span", val=0.0, units="m")
        #self.add_input("mesh")
        self.add_output("sweep_times_span", val=0.0, units="deg*m")

        self.declare_partials("sweep_times_span", "sweep")
        self.declare_partials("sweep_times_span", "span")
    

    def compute(self, inputs, outputs):
        sweep = inputs["sweep"]
        span = inputs["span"]

        outputs["sweep_times_span"] = span/sweep  

    def compute_partials(self, inputs, partials):
        sweep = inputs["sweep"]
        span = inputs["span"]

        partials["sweep_times_span", "sweep"] = -span/sweep**2
        partials["sweep_times_span", "span"] = 1/sweep

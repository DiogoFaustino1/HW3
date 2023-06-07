import openmdao.api as om
from openmdao.devtools import iprofile as tool
import sympy as sym

class Paraboloid(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd', form='backward', step=1e-6, step_calc='abs', minimum_step=1e-6)

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (1 - x)**2 + 100*(y - x**2)**2

class constraint(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('g_xy', val=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        outputs['g_xy'] = x + y

class Paraboloid_analytical(om.ExplicitComponent):

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials(of='f_xy', wrt=['x','y'], method='exact')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (1 - x)**2 + 100*(y - x**2)**2

    def compute_partials(self, inputs, partials):
        x=inputs['x']
        y=inputs['y']
        partials['f_xy', 'x'] = -400*x*(-x**2 + y) + 2*x - 2
        partials['f_xy', 'y'] = -200*x**2 + 200*y

#methods = [('*compute*', (Paraboloid,Paraboloid_analytical)),('*constraint*', (constraint,))]
#tool.setup(methods=methods)
#tool.start()

########################################################################################################################################

# to add the constraint to the model, starting from (-1.2,1.0)
# build the model
prob = om.Problem()
prob.model.add_subsystem('parab', Paraboloid(), promotes_inputs=['x', 'y'])

# define the component whose output will be constrained
prob.model.add_subsystem('const', constraint(), promotes_inputs=['x', 'y'])

# Design variables 'x' and 'y' span components, so we need to provide a common initial
# value for them.
prob.model.set_input_defaults('x', -1.2)
prob.model.set_input_defaults('y', 1.0)

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('x', lower=-2, upper=2)
prob.model.add_design_var('y', lower=-2, upper=2)
prob.model.add_objective('parab.f_xy')
# to add the constraint to the model
prob.model.add_constraint('const.g_xy', upper=1)

prob.setup()
prob.run_driver();

# minimum value
print("Constrained optimization results starting from point (-1.2,1.0):")
print("Function value", prob.get_val('parab.f_xy'))
# location of the minimum
print("X value", prob.get_val('x'), "," , "Y value", prob.get_val('y'))
print("\n\n\n\n\n\n")
##########################################################################################################################################

#problem with analytical derivatives

# build the model
prob_1 = om.Problem()
prob_1.model.add_subsystem('parab_1', Paraboloid_analytical(), promotes_inputs=['x', 'y'])

# define the component whose output will be constrained
prob_1.model.add_subsystem('const_1', constraint(), promotes_inputs=['x', 'y'])

# Design variables 'x' and 'y' span components, so we need to provide a common initial
# value for them.
prob_1.model.set_input_defaults('x', 0.99962936)
prob_1.model.set_input_defaults('y', 0.99925735)

# setup the optimization
prob_1.driver = om.ScipyOptimizeDriver()
prob_1.driver.options['optimizer'] = 'SLSQP'

prob_1.model.add_design_var('x', lower=-2, upper=2)
prob_1.model.add_design_var('y', lower=-2, upper=2)
prob_1.model.add_objective('parab_1.f_xy')
# to add the constraint to the model
prob_1.model.add_constraint('const_1.g_xy', upper=1)

prob_1.setup()
prob_1.run_driver();
tool.stop()

# minimum value
print("Constrained optimization results with analytical derivatives:")
print("Function value", prob_1.get_val('parab_1.f_xy'))
print("X value", prob_1.get_val('x'), "," , "Y value", prob_1.get_val('y'))

#tool.stop()

#data = prob_1.check_partials(compact_print=True, show_only_incorrect=True)
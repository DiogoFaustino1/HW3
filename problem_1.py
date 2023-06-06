import openmdao.api as om
from openmdao.devtools import iprofile as tool
import sympy as sym

#Derivatives of multivariable function

x , y = sym.symbols('x y')
f = (1 - x)**2 + 100*(y - x**2)**2

#Differentiating partially w.r.t x
derivative_x = f.diff(x)
derivative_y = f.diff(y)

print('f\' wrt x =', derivative_x, '\nf\' wrt y =', derivative_y)

class Paraboloid(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd', form='backward', step=1e-3)

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (1 - x)**2 + 100*(y - x**2)**2

class Paraboloid_2(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (1 - x)**2 + 100*(y - x**2)**2


methods = [('*compute*', (Paraboloid,Paraboloid_2))]
tool.setup(methods=methods)
tool.start()

# build the model
prob = om.Problem()
prob.model.add_subsystem('parab', Paraboloid(), promotes_inputs=['x', 'y'])

# define the component whose output will be constrained
prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

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


prob.setup()
prob.run_driver();

# minimum value
print("Unconstrained optimization results:")
print("Function value", prob.get_val('parab.f_xy'))
# location of the minimum
x_opt=prob.get_val('x')
y_opt=prob.get_val('y')
print("X value", prob.get_val('x'), "," , "Y value", prob.get_val('y'))
print("\n\n\n\n\n\n")
########################################################################################################################################

#starting from the unconstrained optimal solution

# build the model
prob_1 = om.Problem()
prob_1.model.add_subsystem('parab', Paraboloid_2(), promotes_inputs=['x', 'y'])

# define the component whose output will be constrained
prob_1.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

# setup the optimization
prob_1.driver = om.ScipyOptimizeDriver()
prob_1.driver.options['optimizer'] = 'SLSQP'

prob_1.model.add_design_var('x', lower=-2, upper=2)
prob_1.model.add_design_var('y', lower=-2, upper=2)
prob_1.model.add_objective('parab.f_xy')


prob_1.model.set_input_defaults('x', x_opt)
prob_1.model.set_input_defaults('y', y_opt)
prob_1.model.add_constraint('const.g', upper=1)

prob_1.setup()
prob_1.run_driver();


# minimum value
print("Constrained optimization results starting from the unconstrained optimal solution:")
print("Function value", prob_1.get_val('parab.f_xy'))
# location of the minimum
print("X value", prob_1.get_val('x'), "," , "Y value", prob_1.get_val('y'), "," , "Constraint value", prob_1.get_val('x')+prob_1.get_val('y'))
tool.stop()
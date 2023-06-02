import openmdao.api as om
from openmdao.devtools import iprofile as tool
import sympy as sym

#Derivatives of multivariable function

x , y = sym.symbols('x y')
f = (1 - x)**2 + 100*(y - x**2)**2

#Differentiating partially w.r.t x
derivative_x = f.diff(x)
derivative_y = f.diff(y)

print('f\'(x)=', derivative_x, '\nf\'(y)=', derivative_y)


class Paraboloid(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
    """

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

        Minimum at: x = 6.6667; y = -7.3333
        """
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (1 - x)**2 + 100*(y - x**2)**2


class Paraboloid_analytical(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
    """

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('f_xy', ['x','y'])

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

        Minimum at: x = 6.6667; y = -7.3333
        """
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (1 - x)**2 + 100*(y - x**2)**2

    def compute_partials(self, inputs, partials):
        x=inputs['x']
        y=inputs['y']
        partials['f_xy', 'x'] = -400*x*(-x**2 + y) + 2*x - 2
        partials['f_xy', 'y'] = -200*x**2 + 200*y

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
prob.driver.options['optimizer'] = 'COBYLA'
prob.driver.options['maxiter'] = 100000

prob.model.add_design_var('x', lower=-2, upper=2)
prob.model.add_design_var('y', lower=-2, upper=2)
prob.model.add_objective('parab.f_xy')

tool.start()
prob.setup()
prob.run_driver();
tool.stop()

# minimum value
print("Unconstrained optimization results:")
print("Function value", prob.get_val('parab.f_xy'))
# location of the minimum
x_opt=prob.get_val('x')
y_opt=prob.get_val('y')
print("X value", prob.get_val('x'), "," , "Y value", prob.get_val('y'))

# to add the constraint to the model, starting from (-1.2,1.0)
prob.model.add_constraint('const.g', upper=1)

tool.start()
prob.setup()
prob.run_driver();
tool.stop()

# minimum value
print("Constrained optimization results starting from point (-1.2,1.0):")
print("Function value", prob.get_val('parab.f_xy'))
# location of the minimum
print("X value", prob.get_val('x'), "," , "Y value", prob.get_val('y'))

#starting from the unconstrained optimal solution
prob.model.set_input_defaults('x', x_opt)
prob.model.set_input_defaults('y', y_opt)

tool.start()
prob.setup()
prob.run_driver();
tool.stop()

# minimum value
print("Constrained optimization results starting from the unconstrained optimal solution:")
print("Function value", prob.get_val('parab.f_xy'))
# location of the minimum
print("X value", prob.get_val('x'), "," , "Y value", prob.get_val('y'))


#problem with analytical derivatives

# build the model
prob_1 = om.Problem()
prob_1.model.add_subsystem('parab', Paraboloid_analytical(), promotes_inputs=['x', 'y'])

# define the component whose output will be constrained
prob_1.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

# Design variables 'x' and 'y' span components, so we need to provide a common initial
# value for them.
prob_1.model.set_input_defaults('x', -1.2)
prob_1.model.set_input_defaults('y', 1.0)

# setup the optimization
prob_1.driver = om.ScipyOptimizeDriver()
prob_1.driver.options['optimizer'] = 'COBYLA'
prob_1.driver.options['maxiter'] = 100000

prob_1.model.add_design_var('x', lower=-2, upper=2)
prob_1.model.add_design_var('y', lower=-2, upper=2)
prob_1.model.add_objective('parab.f_xy')
# to add the constraint to the model
prob_1.model.add_constraint('const.g', upper=1)

tool.start()
prob_1.setup()
prob_1.run_driver();
tool.stop()

# minimum value
print("Constrained optimization results with analytical derivatives:")
print("Function value", prob_1.get_val('parab.f_xy'))
print("X value", prob_1.get_val('x'), "," , "Y value", prob_1.get_val('y'))

data = prob_1.check_partials(compact_print=True, show_only_incorrect=True)
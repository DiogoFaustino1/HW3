import sympy as sym
import openmdao.api as om

#Derivatives of multivariable function
 
x , y = sym.symbols('x y')
f = (1 - x)**2 + 100*(y - x**2)**2

#Differentiating partially w.r.t x
derivative_x = f.diff(x)
derivative_y = f.diff(y)

print('f\'(x)=', derivative_x, '\nf\'(y)=', derivative_y)

class Rosenbrock(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (1 - x)**2 + 100*(y - x**2)**2
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
        f(x,y) = (1 - x)**2 + 100*(y - x**2)**2

        Minimum at: x = 1; y = 1
        """
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (1 - x)**2 + 100*(y - x**2)**2


if __name__ == "__main__":

    model = om.Group()
    model.add_subsystem('comp', Rosenbrock(), promotes_inputs=['x', 'y'])
    model.add_subsystem('const', om.ExecComp('g = x + y'))
    # define the component whose output will be constrained 
    prob = om.Problem(model)
    prob.setup()

    prob.set_val('comp.x', 0.0)
    prob.set_val('comp.y', 0.0)

    prob.run_model()
    print(prob['comp.f_xy'])

    prob.set_val('comp.x', -1.0)
    prob.set_val('comp.y', 1.0)

    prob.run_model()
    print(prob.get_val('comp.f_xy'))


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
prob.model.add_objective('comp.f_xy')
# to add the constraint to the model
prob.model.add_constraint('const.g', upper=100)

prob.setup()
prob.run_driver()

# minimum value
print(prob.get_val('comp.f_xy'))
# location of the minimum
x_opt=prob.get_val('x')
print("Unconstrained crllll", prob.get_val('x'))
y_opt=prob.get_val('y')
print(prob.get_val('y'))


##############################################################################################

# Design variables 'x' and 'y' span components, so we need to provide a common initial
# value for them.
prob.model.set_input_defaults('x', x_opt)
prob.model.set_input_defaults('y', y_opt)

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'COBYLA'
prob.driver.options['maxiter'] = 100000


# Design variables 'x' and 'y' span components, so we need to provide a common initial
# value for them.
prob.model.set_input_defaults('x', -1.2)
prob.model.set_input_defaults('y', 1.0)

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['maxiter'] = 100000

#prob.model.add_design_var('x', lower=-2, upper=2)
#prob.model.add_design_var('y', lower=-2, upper=2)
#prob.model.add_objective('comp.f_xy')

prob.setup()
prob.run_driver()

# minimum value
print(prob.get_val('comp.f_xy'))
# location of the minimum
x_opt=prob.get_val('x')
print(prob.get_val('x'))
y_opt=prob.get_val('y')
print(prob.get_val('y'))
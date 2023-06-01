import openmdao.api as om


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


if __name__ == "__main__":

    model = om.Group()
    model.add_subsystem('parab_comp', Paraboloid())

    prob = om.Problem(model)
    prob.setup()

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

prob.setup()
prob.run_driver();

# minimum value
print("Unconstrained optimization results:")
print(prob.get_val('parab.f_xy'))
# location of the minimum
x_opt=prob.get_val('x')
print(prob.get_val('x'))
y_opt=prob.get_val('y')
print(prob.get_val('y'))

# to add the constraint to the model, starting from (-1.2,1.0)
prob.model.add_constraint('const.g', upper=1)

prob.setup()
prob.run_driver();

# minimum value
print("Constrained optimization results starting from point (-1.2,1.0):")
print(prob.get_val('parab.f_xy'))
# location of the minimum
print(prob.get_val('x'))
print(prob.get_val('y'))

#starting from the unconstrained optimal solution
prob.model.set_input_defaults('x', x_opt)
prob.model.set_input_defaults('y', y_opt)

prob.setup()
prob.run_driver();

# minimum value
print("Constrained optimization results starting from the unconstrained optimal solution:")
print("Function value", prob.get_val('parab.f_xy'))
# location of the minimum
print(prob.get_val('x'))
print(prob.get_val('y'))
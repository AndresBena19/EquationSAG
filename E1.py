from LinealEquationAG.SQAG import Equation

PRECISION_ = 4

# Define base parameters
generation_number = 600
point_precision_stop = 0.01
# Elitism number : This allow to select N amount of chromosomes from generation to pass the next
elitism_number = 20

crossover_percentage = 0.9
mutation_percentage = 0.2

limits_by_incognitos = {'x': [50, 250],
                        'y': [50, 250]}

equation = [{'x': 0.01, 'y': 0.02,  'ind_term': 4},
            {'x': 0.02, 'y': 0.05,  'ind_term': 9}]

equation_problem = Equation(limits_by_incognitos,
                            equation,
                            generation_number,
                            crossover_percentage,
                            mutation_percentage,
                            elitism_number,
                            point_precision_stop,
                            PRECISION_)

equation_problem.solve()

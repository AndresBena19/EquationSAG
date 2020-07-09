from LinealEquationAG.SQAG import Equation

PRECISION_ = 4

# Define base parameters
generation_number = 2000
point_precision_stop = 0.01
# Elitism number : This allow to select N amount of chromosomes from generation to pass the next
elitism_number = 20

crossover_percentage = 0.9
mutation_percentage = 0.3

limits_by_incognitos = {'A': [4980, 5000],
                        'P': [2990, 3020],
                        'V': [3000, 3020]}

equation = [{'A': 16.98, 'P': 9, 'V': 9, 'ind_term': 138900},
            {'A': 15.9, 'P': 8.72, 'V': 8.52, 'ind_term': 131220},
            {'A': 14.08, 'P': 8.2, 'V': 8.76, 'ind_term': 121280}]

equation_problem = Equation(limits_by_incognitos,
                            equation,
                            generation_number,
                            crossover_percentage,
                            mutation_percentage,
                            elitism_number,
                            point_precision_stop,
                            PRECISION_)

equation_problem.solve()

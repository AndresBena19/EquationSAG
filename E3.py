from LinealEquationAG.SQAG import Equation

PRECISION_ = 4

if __name__ == "__main__":
    # Define base parameters
    generation_number = 3000
    point_precision_stop = 0.01
    # Elitism number : This allow to select N amount of chromosomes from generation to pass the next
    elitism_number = 450

    crossover_percentage = 0.9
    mutation_percentage = 0.25

    limits_by_incognitos = {'x1': [0, 3],
                            'x2': [0, 3],
                            'x3': [0, 3],
                            'x4': [0, 3]}
    equation = [{'x1': 1, 'x2': -1, 'x3': 1, 'x4': 1, 'ind_term': 4},
                {'x1': 2, 'x2': 1, 'x3': -3, 'x4': 1, 'ind_term': 4},
                {'x1': 1, 'x2': -2, 'x3': 2, 'x4': -1, 'ind_term': 3},
                {'x1': 1, 'x2': -3, 'x3': 3, 'x4': -3, 'ind_term': 2}]

    equation_problem = Equation(limits_by_incognitos,
                                equation,
                                generation_number,
                                crossover_percentage,
                                mutation_percentage,
                                elitism_number,
                                point_precision_stop,
                                PRECISION_)

    equation_problem.solve()

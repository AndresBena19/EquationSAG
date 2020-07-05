import numpy as np
import copy


class Chromosome:

    def __init__(self, start, end, length, *args, **kwargs):
        self.metadata = np.random.uniform(low=start, high=end, size=length)
        self.system_equation = kwargs.get('system_equation')

    @property
    def fitness(self):
        fitness_val = 0
        for equation_object, ind_term in self.system_equation:
            result = equation_object * self.metadata
            absolute_result = abs(ind_term - sum(result))
            fitness_val += absolute_result
        return fitness_val


class GenerationProcess:

    def __init__(self, equation_schema, start, end, length, *args, **kwargs):
        self.equation_schema = equation_schema
        self.start = start
        self.end = end
        self.length = length
        self.generation_number = kwargs.get('number_generations')
        self.percentage_crossover = kwargs.get('percentage_crossover')
        self.percentage_mutation = kwargs.get('percentage_mutation')
        self.elitism_number = kwargs.get('elitism_number')
        self.generation = None

    def generate_generation(self, descendants=None):
        self.generation = [Chromosome(self.start,
                                      self.end,
                                      self.length,
                                      system_equation=self.generate_numpy_object_equation())
                           for _ in range(self.generation_number)]
        return self.generation

    def generate_numpy_object_equation(self):
        new_schema = []
        for schema in self.equation_schema:
            individual_equation = copy.copy(schema)
            independent_term = individual_equation.pop('ind_term')
            new_schema.append((np.array(list(individual_equation.values())), independent_term))
        return new_schema

    def elitism(self):
        return sorted(self.generation, key=lambda val: val.fitness)[:self.elitism_number]

    def selection(self):
        index_random_parent_one = np.random.choice(self.generation_number)
        index_random_parent_two = np.random.choice(self.generation_number)

        chromosome_one = self.generation[index_random_parent_one]
        chromosome_two = self.generation[index_random_parent_two]
        if chromosome_one.fitness > chromosome_two.fitness:
            parent_one = chromosome_one
        else:
            parent_one = chromosome_two
        index_random_parent = np.random.choice(self.generation_number)
        parent_two = self.generation[index_random_parent]
        return parent_one, parent_two

    def crossover(self, parent_one, parent_two):
        can_crossover = round(np.random.uniform(low=0.0, high=1.0), 2)
        descendant_one = parent_one
        descendant_two = parent_two
        if can_crossover < self.percentage_crossover:
            index_to_break = np.random.randint(1, self.length)
            descendant_one.metadata = np.concatenate(
                [parent_one.metadata[:index_to_break], parent_two.metadata[index_to_break:]])
            descendant_two.metadata = np.concatenate(
                [parent_two.metadata[:index_to_break], parent_one.metadata[index_to_break:]])

        return descendant_one, descendant_two

    def mutation(self, descendant_one, descendant_two):

        can_mutate = round(np.random.uniform(low=0.0, high=1.0), 2)
        if can_mutate < self.percentage_mutation:
            index_to_mutate = np.random.randint(1, self.length)
            new_data = np.random.uniform(self.start, self.end)
            descendant_to_mutate = np.random.choice([descendant_one, descendant_two])
            if descendant_to_mutate == descendant_one:
                descendant_to_mutate.metadata[index_to_mutate] = new_data
                descendant_one = descendant_to_mutate
            else:
                descendant_to_mutate.metadata[index_to_mutate] = new_data
                descendant_two = descendant_to_mutate

            return descendant_one, descendant_two
        return descendant_one, descendant_two


class Equation:

    def __init__(self, start, end,
                 incognito_number,
                 equation_object,
                 number_generations,
                 percentage_crossover,
                 percentage_mutation,
                 number_elitism):

        self.generation_number = number_generations
        self.number_elitism = number_elitism
        self.generation = GenerationProcess(equation_object,
                                            start,
                                            end,
                                            incognito_number,
                                            number_generations=number_generations,
                                            percentage_crossover=percentage_crossover,
                                            percentage_mutation=percentage_mutation,
                                            elitism_number=elitism_number)

    def solve(self):

        print("------------ Generating 1er Generation ----------------")
        self.generation.generate_generation()
        iterations = int((self.generation_number - self.number_elitism) / 2)

        for _ in range(1, self.generation_number):
            print("------------ Generating {} Generation ----------------".format(_))

            best_solutions = copy.deepcopy(self.generation.elitism())
            temp_generation = []

            for across in range(iterations):
                parent_one, parent_two = self.generation.selection()
                descendant_one, descendant_two = self.generation.crossover(parent_one, parent_two)
                descendant_one_mutated, descendant_two_mutated = self.generation.mutation(descendant_one,
                                                                                          descendant_two)
                temp_generation.extend([descendant_one_mutated, descendant_two_mutated])

            temp_generation.extend(best_solutions)
            self.generation.generation = temp_generation

            after_bes_fuck = self.generation.elitism()[0]
            print("Best local fitness {} : fitness {}".format(after_bes_fuck.metadata, after_bes_fuck.fitness))

        for best_result in self.generation.elitism():
            print("Result {} : fitness {}".format(best_result.metadata, best_result.fitness))


if __name__ == "__main__":
    generation_number = 2000
    elitism_number = 1000

    start_solution = 0
    end_solution = 3
    number_incognitos = 4

    crossover_percentage = 0.9
    mutation_percentage = 0.3

    equation_ = [{'x': 3, 'y': 8, 'z': 2, 'ind_term': 25},
                 {'x': 1, 'y': -2, 'z': 4, 'ind_term': 12},
                 {'x': -5, 'y': 3, 'z': 11, 'ind_term': 4}]

    equation__ = [{'A': 16.98, 'P': 9, 'V': 9, 'ind_term': 138900},
                {'A': 15.9, 'P': 8.72, 'V': 8.52, 'ind_term': 131220},
                {'A': 14.08, 'P': 8.2, 'V': 8.76, 'ind_term': 121280}]

    equation = [{'x1': 1, 'x2': -1, 'x3': 1, 'x4': 1, 'ind_term': 4},
                 {'x1': 2, 'x2': 1, 'x3': -3, 'x4': 1, 'ind_term': 4},
                 {'x1': 1, 'x2': -2, 'x3': 2, 'x4': -1, 'ind_term': 3},
                 {'x1': 1, 'x2': -3, 'x3': 3, 'x4': -3,  'ind_term': 2} ]

    equation_problem = Equation(start_solution,
                                end_solution,
                                number_incognitos,
                                equation,
                                generation_number,
                                crossover_percentage,
                                mutation_percentage,
                                elitism_number)

    equation_problem.solve()

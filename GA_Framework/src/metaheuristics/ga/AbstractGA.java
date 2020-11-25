package metaheuristics.ga;

import java.util.ArrayList;
import java.util.Random;

import problems.Evaluator;
import problems.qbfpt.qbf.QBFPT;
import solutions.Solution;

/**
 * Abstract class for metaheuristic GA (Genetic Algorithms). It consider the
 * maximization of the chromosome fitness.
 * 
 * @author ccavellucci, fusberti
 * @param <G>
 *            Generic type of the chromosome element (genotype).
 * @param <F>
 *            Generic type of the candidate to enter the solution (fenotype).
 */
public abstract class AbstractGA<G extends Number, F> {

	@SuppressWarnings("serial")
	public class Chromosome extends ArrayList<G> {
	}

	@SuppressWarnings("serial")
	public class Population extends ArrayList<Chromosome> {
	}

	/**
	 * flag that indicates whether the code should print more information on
	 * screen
	 */
	public static boolean verbose = true;

	private boolean performExtraMutations = false;
	private int extraMutationsCounter = 0;
	private boolean removeToMakeFeasible = false;

	/**
	 * a random number generator
	 */
	public static final Random rng = new Random(0);

	/**
	 * the objective function being optimized
	 */
	protected Evaluator<F> ObjFunction;

	/**
	 * maximum number of generations being executed
	 */
	protected int generations;

	/**
	 * the size of the population
	 */
	protected int popSize;

	/**
	 * the size of the chromosome
	 */
	protected int chromosomeSize;

	/**
	 * the probability of performing a mutation
	 */
	protected double mutationRate;

	/**
	 * the best solution cost
	 */
	protected Double bestCost;

	/**
	 * the best solution
	 */
	protected Solution<F> bestSol;

	/**
	 * the best chromosome, according to its fitness evaluation
	 */
	protected Chromosome bestChromosome;

	/**
	 * Creates a new solution which is empty, i.e., does not contain any
	 * candidate solution element.
	 * 
	 * @return An empty solution.
	 */
	public abstract Solution<F> createEmptySol();

	/**
	 * A mapping from the genotype (domain) to the fenotype (image). In other
	 * words, it takes a chromosome as input and generates a corresponding
	 * solution.
	 * 
	 * @param chromosome
	 *            The genotype being considered for decoding.
	 * @return The corresponding fenotype (solution).
	 */
	protected abstract Solution<F> decode(Chromosome chromosome);

	/**
	 * Generates a random chromosome according to some probability distribution
	 * (usually uniform).
	 * 
	 * @return A random chromosome.
	 */
	protected abstract Chromosome generateRandomChromosome();

	/**
	 * Determines the fitness for a given chromosome. The fitness should be a
	 * function strongly correlated to the objective function under
	 * consideration.
	 * 
	 * @param chromosome
	 *            The genotype being considered for fitness evaluation.
	 * @return The fitness value for the input chromosome.
	 */
	protected abstract Double fitness(Chromosome chromosome);

	protected abstract Integer findForbiddenValue(Chromosome chromosome);

	protected abstract void removeUntilFeasible(Chromosome c);

	/**
	 * Mutates a given locus of the chromosome. This method should be preferably
	 * called with an expected frequency determined by the {@link #mutationRate}.
	 * 
	 * @param chromosome
	 *            The genotype being mutated.
	 * @param locus
	 *            The position in the genotype being mutated.
	 */
	protected abstract void mutateGene(Chromosome chromosome, Integer locus);

	/**
	 * The constructor for the GA class.
	 * 
	 * @param objFunction
	 *            The objective function being optimized.
	 * @param generations
	 *            Number of generations to be executed.
	 * @param popSize
	 *            Population size.
	 * @param mutationRate
	 *            The mutation rate.
	 */
	public AbstractGA(Evaluator<F> objFunction, Integer generations, Integer popSize, Double mutationRate) {
		this.ObjFunction = objFunction;
		this.generations = generations;
		this.popSize = popSize;
		this.chromosomeSize = this.ObjFunction.getDomainSize();
		this.mutationRate = mutationRate;
	}

	public int getExtraMutationsCounter() {
		return extraMutationsCounter;
	}

	/**
	 * The GA mainframe. It starts by initializing a population of chromosomes.
	 * It then enters a generational loop, in which each generation goes the
	 * following steps: parent selection, crossover, mutation, population update
	 * and best solution update.
	 * 
	 * @return The best feasible solution obtained throughout all iterations.
	 */
	public Solution<F> solve(boolean isSUS, boolean isUniformCrossover, boolean isSteadyState, boolean forceMutations, boolean remove) {
		this.performExtraMutations = forceMutations;
		this.removeToMakeFeasible = remove;
		this.extraMutationsCounter = 0;

		long startTime = System.currentTimeMillis();
		long endTime;
		double totalTime;
		double maxTime = 1800.0; //30 min

		/* starts the initial population */
		Population population = initializePopulation();

		bestChromosome = getBestChromosome(population);
		bestSol = decode(bestChromosome);
		System.out.println("(Gen. " + 0 + ") BestSol = " + bestSol);

		/*
		 * enters the main loop and repeats until a given number of generations
		 */
		for (int g = 1; g <= generations; g++) {

			Population parents;
            if(isSUS){
                parents = selectParentsSUS(population);
            }else{
                parents = selectParents(population);
            }


            Population offsprings;
			if(isUniformCrossover){
                offsprings = uniformCrossover(parents);
            }else{
                offsprings = crossover(parents);
            }


			Population mutants = mutate(offsprings);

			Population newpopulation;

			if(isSteadyState){
                newpopulation = selectPopulationSteadyState(population, mutants);
            }else{
                newpopulation = selectPopulation(mutants);
            }

			population = newpopulation;

			bestChromosome = getBestChromosome(population);

			if (fitness(bestChromosome) > bestSol.cost) {
				bestSol = decode(bestChromosome);
				if (verbose)
					System.out.println("(Gen. " + g + ") BestSol = " + bestSol);
			}

			endTime   = System.currentTimeMillis();
			totalTime = (endTime - startTime)/(double)1000;

			//if it exceeded the time limit of 1800s (30 min), then break the loop
			if(totalTime >= maxTime) break;
		}

		return bestSol;
	}

	/**
	 * Randomly generates an initial population to start the GA.
	 * 
	 * @return A population of chromosomes.
	 */
	protected Population initializePopulation() {

		Population population = new Population();

		while (population.size() < popSize) {
			population.add(generateRandomChromosome());
		}

		return population;

	}

	/**
	 * Given a population of chromosome, takes the best chromosome according to
	 * the fitness evaluation.
	 * 
	 * @param population
	 *            A population of chromosomes.
	 * @return The best chromosome among the population.
	 */
	protected Chromosome getBestChromosome(Population population) {

		double bestFitness = Double.NEGATIVE_INFINITY;
		Chromosome bestChromosome = null;
		for (Chromosome c : population) {
			double fitness = fitness(c);
			if (fitness > bestFitness) {
				bestFitness = fitness;
				bestChromosome = c;
			}
		}

		return bestChromosome;
	}

	/**
	 * Given a population of chromosome, takes the worst chromosome according to
	 * the fitness evaluation.
	 * 
	 * @param population
	 *            A population of chromosomes.
	 * @return The worst chromosome among the population.
	 */
	protected Chromosome getWorseChromosome(Population population) {

		double worseFitness = Double.POSITIVE_INFINITY;
		Chromosome worseChromosome = null;
		for (Chromosome c : population) {
			double fitness = fitness(c);
			if (fitness < worseFitness) {
				worseFitness = fitness;
				worseChromosome = c;
			}
		}

		return worseChromosome;
	}

	/**
	 * Selection of parents for crossover using the tournament method. Given a
	 * population of chromosomes, randomly takes two chromosomes and compare
	 * them by their fitness. The best one is selected as parent. Repeat until
	 * the number of selected parents is equal to {@link #popSize}.
	 * 
	 * @param population
	 *            The current population.
	 * @return The selected parents for performing crossover.
	 */
	protected Population selectParents(Population population) {

		Population parents = new Population();

		while (parents.size() < popSize) {
			int index1 = rng.nextInt(popSize);
			Chromosome parent1 = population.get(index1);
			int index2 = rng.nextInt(popSize);
			Chromosome parent2 = population.get(index2);
			if (fitness(parent1) > fitness(parent2)) {
				parents.add(parent1);
			} else {
				parents.add(parent2);
			}
		}

		return parents;

	}

    /**
     * Selection of parents for crossover using the Stochastic universal selection (SUS)
     * method, considering 2 pointers
     *
     * @param population
     *            The current population.
     * @return The selected parents for performing crossover.
     */
    protected Population selectParentsSUS(Population population) {

        Population parents = new Population();

        double total = 0;
        double fitnessValue;
        int size = population.size();

        ArrayList<Double> fitnessList = new ArrayList<>();
        for(Chromosome chromosome : population){
            fitnessValue = fitness(chromosome);
            total += fitnessValue;
            fitnessList.add(fitnessValue);
        }

        while (parents.size() < popSize) {
            double pointer1 = rng.nextDouble();
            double pointer2 = pointer1 + 0.5;
            if (pointer2 > 1.0){
                pointer2 -= 1.0;
            }

            double start;
            double end = 0.0;
            for(int i=0; i < size; i++){
                start = end;
                end += fitnessList.get(i)/total;

                if( (start <= pointer1 && pointer1 <= end) ||
                    (start <= pointer2 && pointer2 <= end)){
                    parents.add(population.get(i));

                    if(parents.size() == popSize) break;
                }
            }
        }

        return parents;

    }

	/**
	 * The crossover step takes the parents generated by {@link #selectParents}
	 * and recombine their genes to generate new chromosomes (offsprings). The
	 * method being used is the 2-point crossover, which randomly selects two
	 * locus for being the points of exchange (P1 and P2). For example:
	 * 
	 *                        P1            P2
	 *    Parent 1: X1 ... Xi | Xi+1 ... Xj | Xj+1 ... Xn
	 *    Parent 2: Y1 ... Yi | Yi+1 ... Yj | Yj+1 ... Yn
	 * 
	 * Offspring 1: X1 ... Xi | Yi+1 ... Yj | Xj+1 ... Xn
	 * Offspring 2: Y1 ... Yi | Xi+1 ... Xj | Yj+1 ... Yn
	 * 
	 * @param parents
	 *            The selected parents for crossover.
	 * @return The resulting offsprings.
	 */
	protected Population crossover(Population parents) {

		Population offsprings = new Population();

		for (int i = 0; i < popSize; i = i + 2) {

			Chromosome parent1 = parents.get(i);
			Chromosome parent2 = parents.get(i + 1);

			int crosspoint1 = rng.nextInt(chromosomeSize + 1);
			int crosspoint2 = crosspoint1 + rng.nextInt((chromosomeSize + 1) - crosspoint1);

			Chromosome offspring1 = new Chromosome();
			Chromosome offspring2 = new Chromosome();

			for (int j = 0; j < chromosomeSize; j++) {
				if (j >= crosspoint1 && j < crosspoint2) {
					offspring1.add(parent2.get(j));
					offspring2.add(parent1.get(j));
				} else {
					offspring1.add(parent1.get(j));
					offspring2.add(parent2.get(j));
				}
			}

			if (performExtraMutations) {
				extraMutations(offspring1);
				extraMutations(offspring2);
			}

			if (removeToMakeFeasible) {
				removeUntilFeasible(offspring1);
				removeUntilFeasible(offspring2);
			}

			offsprings.add(offspring1);
			offsprings.add(offspring2);

		}

		return offsprings;

	}


	/**
	 * The crossover step takes the parents generated by {@link #selectParents}
	 * and recombine their genes to generate new chromosomes (offsprings). The
	 * method being used is the uniform crossover, which randomly selects the
	 * allele from parent P1 or parent P2 and exchange. For example:
     *
     *    Crossover Mask:   0  0  1  0  1  0      1
     *    Parent 1:         X1 X2 X3 X4 X5 X6 ... Xn
     *    Parent 2:         Y1 Y2 Y3 Y4 Y5 Y6 ... Yn
     *
     *    Offspring 1:      X1 X2 Y3 X4 Y5 X6 ... Yn
     *    Offspring 2:      Y1 Y2 X3 Y4 X5 X6 ... Xn
	 * @param parents
	 *            The selected parents for crossover.
	 * @return The resulting offsprings.
	 */
	protected Population uniformCrossover(Population parents) {
		Population offsprings = new Population();

		for (int i = 0; i < popSize; i = i + 2) {

			Chromosome parent1 = parents.get(i);
			Chromosome parent2 = parents.get(i + 1);

			Chromosome offspring1 = new Chromosome();
			Chromosome offspring2 = new Chromosome();

			for (int j = 0; j < chromosomeSize; j++) {
			    int whichParent = rng.nextInt(2);
				if (whichParent==1) {
					offspring1.add(parent2.get(j));
					offspring2.add(parent1.get(j));
				} else {
					offspring1.add(parent1.get(j));
					offspring2.add(parent2.get(j));
				}
			}

			if (performExtraMutations) {
				extraMutations(offspring1);
				extraMutations(offspring2);
			}

			if (removeToMakeFeasible) {
				removeUntilFeasible(offspring1);
				removeUntilFeasible(offspring2);
			}

			offsprings.add(offspring1);
			offsprings.add(offspring2);

		}

		return offsprings;
	}
	/**
	 * The mutation step takes the offsprings generated by {@link #crossover}
	 * and to each possible locus, perform a mutation with the expected
	 * frequency given by {@link #mutationRate}.
	 * 
	 * @param offsprings
	 *            The offsprings chromosomes generated by the
	 *            {@link #crossover}.
	 * @return The mutated offsprings.
	 */
	protected Population mutate(Population offsprings) {

		for (Chromosome c : offsprings) {
			for (int locus = 0; locus < chromosomeSize; locus++) {
				if (rng.nextDouble() < mutationRate) {
					mutateGene(c, locus);

					if (performExtraMutations) {
						extraMutations(c);
					}

					if (removeToMakeFeasible) {
						removeUntilFeasible(c);
					}
				}
			}
		}

		return offsprings;
	}

	private void extraMutations(Chromosome c) {
		while (fitness(c) <= -5000)
		{
			for (int locus = 0; locus < chromosomeSize; locus++) {
				if (rng.nextDouble() < mutationRate) {
					mutateGene(c, locus);
					extraMutationsCounter++;
				}
			}
		}
	}

	/**
	 * Updates the population that will be considered for the next GA
	 * generation. The method used for updating the population is the elitist,
	 * which simply takes the worse chromosome from the offsprings and replace
	 * it with the best chromosome from the previous generation.
	 * 
	 * @param offsprings
	 *            The offsprings generated by {@link #crossover}.
	 * @return The updated population for the next generation.
	 */
	protected Population selectPopulation(Population offsprings) {

		Chromosome worse = getWorseChromosome(offsprings);
		if (fitness(worse) < fitness(bestChromosome)) {
			offsprings.remove(worse);
			offsprings.add(bestChromosome);
		}

		return offsprings;
	}

    /**
     * Updates the population that will be considered for the next GA
     * generation. The method used for updating the population is the Steady-State. It replaces
     * the worst and a random individuals from the current population by the 2 best
     * individuals from the offspring
     *
     * @param offsprings
     *            The offsprings generated by {@link #crossover}.
     * @return The updated population for the next generation.
     */
    protected Population selectPopulationSteadyState(Population currPopulation,
                                                     Population offsprings) {

	    int indexRemoval = rng.nextInt(currPopulation.size());

        Chromosome currWorse = getWorseChromosome(currPopulation);
        Chromosome randomIndiv = currPopulation.get(indexRemoval);

        Chromosome newBest1 = getBestChromosome(offsprings);
        offsprings.remove(newBest1);

        Chromosome newBest2 = getBestChromosome(offsprings);

        double bestCurrFitness = fitness(bestChromosome);

        if (fitness(currWorse) < bestCurrFitness) {
            currPopulation.remove(currWorse);
            currPopulation.add(newBest1);
        }

        if (fitness(randomIndiv) < bestCurrFitness) {
            currPopulation.remove(randomIndiv);
            currPopulation.add(newBest2);
        }

        return currPopulation;
    }

}

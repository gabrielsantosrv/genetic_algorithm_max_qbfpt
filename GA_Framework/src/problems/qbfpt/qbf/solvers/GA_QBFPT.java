package problems.qbfpt.qbf.solvers;

import metaheuristics.ga.AbstractGA;
import problems.qbfpt.qbf.QBFPT;
import solutions.Solution;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

/**
 * Metaheuristic GA (Genetic Algorithm) for
 * obtaining an optimal solution to a QBFPT (Quadractive Binary Function --
 * {@link #QuadracticBinaryFunction}). 
 * 
 * @author ccavellucci, fusberti
 */
public class GA_QBFPT extends AbstractGA<Integer, Integer> {

	/**
	 * Constructor for the GA_QBFPT class. The QBFPT objective function is passed as
	 * argument for the superclass constructor.
	 * 
	 * @param generations
	 *            Maximum number of generations.
	 * @param popSize
	 *            Size of the population.
	 * @param mutationRate
	 *            The mutation rate.
	 * @param filename
	 *            Name of the file for which the objective function parameters
	 *            should be read.
	 * @throws IOException
	 *             Necessary for I/O operations.
	 */
	public GA_QBFPT(Integer generations, Integer popSize, Double mutationRate, String filename) throws IOException {
		super(new QBFPT(filename), generations, popSize, mutationRate);
	}

	/**
	 * {@inheritDoc}
	 * 
	 * This createEmptySol instantiates an empty solution and it attributes a
	 * zero cost, since it is known that a QBFPT solution with all variables set
	 * to zero has also zero cost.
	 */
	@Override
	public Solution<Integer> createEmptySol() {
		Solution<Integer> sol = new Solution<Integer>();
		sol.cost = 0.0;
		return sol;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see metaheuristics.ga.AbstractGA#decode(metaheuristics.ga.AbstractGA.
	 * Chromosome)
	 */
	@Override
	protected Solution<Integer> decode(Chromosome chromosome) {

		Solution<Integer> solution = createEmptySol();
		for (int locus = 0; locus < chromosome.size(); locus++) {
			if (chromosome.get(locus) == 1) {
				solution.add(new Integer(locus));
			}
		}

		ObjFunction.evaluate(solution);
		return solution;
	}


	/*
	 * (non-Javadoc)
	 * 
	 * @see metaheuristics.ga.AbstractGA#generateRandomChromosome()
	 */
	@Override
	protected Chromosome generateRandomChromosome() {

		Chromosome chromosome = new Chromosome();
		for (int i = 0; i < chromosomeSize; i++) {
			chromosome.add(rng.nextInt(2));
		}

		return chromosome;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see metaheuristics.ga.AbstractGA#fitness(metaheuristics.ga.AbstractGA.
	 * Chromosome)
	 */
	@Override
	protected Double fitness(Chromosome chromosome) {

		Solution<Integer> sol = decode(chromosome);
		//if is a infeasible solution, then returns -5000
		if(!((QBFPT)ObjFunction).isFeasible(sol))
		    return -5000.0;

        return sol.cost;

	}

	@Override
	protected Integer findForbiddenValue(Chromosome chromosome)
	{
		Solution<Integer> sol = decode(chromosome);
		return ((QBFPT)ObjFunction).findForbiddenValue(sol);
	}

	@Override
	protected void removeUntilFeasible(Chromosome c)
	{
		Integer x;
		while ((x = findForbiddenValue(c)) != null)
		// x está na solução mas forma tripla proibida com alguém
		{
			c.set(x, 0);
			// remover x da solução!
		}
	}


	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * metaheuristics.ga.AbstractGA#mutateGene(metaheuristics.ga.AbstractGA.
	 * Chromosome, java.lang.Integer)
	 */
	@Override
	protected void mutateGene(Chromosome chromosome, Integer locus) {

		chromosome.set(locus, 1 - chromosome.get(locus));

	}



	/**
	 * A main method used for testing the GA metaheuristic.
	 * @throws URISyntaxException 
	 * 
	 */
	public static void main(String[] args) throws IOException, URISyntaxException {
		List<String> instances = new ArrayList<String>();
		instances.add("qbf020");
		instances.add("qbf040");
		instances.add("qbf060");
		instances.add("qbf080");
		instances.add("qbf100");
		instances.add("qbf200");
		instances.add("qbf400");
		int[] instanceSize  = {20, 40, 60, 80, 100, 200, 400};
		
		for(int i=0; i<instances.size(); i++) {
			try {
				System.out.println("Instance " + instanceSize[i]);
				Integer pop1 = 100;
				Integer pop2 = 1000;
				Double mut1 = 1.0 / instanceSize[i];
				Double mut2 = 1.0 / 100.0;
				
				GA_QBFPT gaPadrao = new GA_QBFPT(100000, pop1, mut1, "instances/"+instances.get(i));
				GA_QBFPT gaPop = new GA_QBFPT(100000, pop2, mut1, "instances/"+instances.get(i));
				GA_QBFPT gaMut = new GA_QBFPT(100000, pop1, mut2, "instances/"+instances.get(i));
				GA_QBFPT gaEvol1 = new GA_QBFPT(100000, pop1, mut1, "instances/"+instances.get(i));
				GA_QBFPT gaEvol2 = new GA_QBFPT(100000, pop1, mut1, "instances/"+instances.get(i));

				System.out.println("Instanciou");
				
				FileWriter fileWriter = new FileWriter("results/"+instances.get(i)+".txt");
				fileWriter.append(" ======== Execução " + instances.get(i) + " ======= \n\n");

				System.out.println("Mexeu no arquivo");
				
				GA_QBFPT.executeInstance("GA Padrão",gaPadrao, fileWriter, false, false, false, false, false);
				GA_QBFPT.executeInstance("GA Pop",gaPop, fileWriter, false, false, false, false, false);
				GA_QBFPT.executeInstance("GA Mut",gaMut, fileWriter, false, false, false, false, false);
				GA_QBFPT.executeInstance("GA Evol1",gaEvol1, fileWriter, false, true, false, false, false);
				GA_QBFPT.executeInstance("GA Evol2",gaEvol2, fileWriter, false, false, true, false, false);

				System.out.println("Executou as variantes");

				fileWriter.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

	}
	
	public static void executeInstance(String title, GA_QBFPT ga, FileWriter fileWriter, boolean isSUS, boolean isUniformCrossover, boolean isSteadyState, boolean forceMutations, boolean remove) {
		long startTime = System.currentTimeMillis();
		Solution<Integer> bestSol = ga.solve(isSUS, isUniformCrossover, isSteadyState, forceMutations, remove);
		long endTime = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		double time = (double)totalTime/(double)1000;

		System.out.println("title = " + title);
		System.out.println("maxVal = " + bestSol);
		System.out.println("Time = "+ time +" seg");
		System.out.println("Extra muts = " + ga.getExtraMutationsCounter());
		
		if(fileWriter != null) {
			try {
				fileWriter.append(title+"\n");
				fileWriter.append("Best solution: "+ bestSol + "\n");
				fileWriter.append("Time: "+ time + "seg \n");
				fileWriter.append("Extra mutations: " + ga.getExtraMutationsCounter() + "\n\n");
			} catch (IOException e) {
				e.printStackTrace();
				System.out.println("Error writing in file: "+title);
			}
		}
	}

}

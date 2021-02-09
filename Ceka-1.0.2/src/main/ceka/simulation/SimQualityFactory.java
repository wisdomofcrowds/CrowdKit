/**
 * 
 */
package ceka.simulation;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;

/**
 * @author Jing
 *
 */
public class SimQualityFactory {
	
	/**
	 * generate uniform distribution
	 * @param n number of values
	 * @param lower lower boundary (inclusive)
	 * @param upper upper upper boundary (exclusive)
	 * @return sequence
	 */
	public static double [] sampleUniform(int n, double lower, double upper) {
		double [] vals = new double[n];
		JDKRandomGenerator randGen = new JDKRandomGenerator();
		long seed = System.nanoTime();
		randGen.setSeed(seed);
		UniformRealDistribution distrib = new UniformRealDistribution(randGen, lower, upper);
		for (int i = 0; i < n; i++)
			vals[i] = distrib.sample();
		return vals;
	}
	
	/**
	 * generate truncated Gaussian distribution
	 * @param n number of values
	 * @param mean mean
	 * @param sd   standard deviation
	 * @param lower lower boundary (inclusive)
	 * @param upper upper boundary (exclusive)
	 * @return sequence
	 */
	public static double [] sampleTruncatedGaussian(int n, double mean, double sd, double lower, double upper) {
		double [] vals = new double[n];
		JDKRandomGenerator randGen = new JDKRandomGenerator();
		long seed = System.nanoTime();
		randGen.setSeed(seed);
		NormalDistribution distrib = new NormalDistribution(randGen, mean, sd);
		int count  = 0;
		while (count < n) {
			double v = distrib.sample();
			if ((v >= lower) && (v < upper))
				vals[count++] = v;
		}
		return vals;
	}
	
	/*
	 * for test
	 */
	public static void main(String[] args) {
		int n = 20;
		double lower = 0.5;
		double upper = 0.7;
		double mean  = 0.6;
		double sd    = 0.1;
		double [] vals1 = SimQualityFactory.sampleUniform(n, lower, upper);
		int count = 0;
		for (double e : vals1)
			System.out.println("u[" + (count++) + "]=" + e);
		double [] vals2 = SimQualityFactory.sampleTruncatedGaussian(n, mean, sd, lower, upper);
		count = 0;
		for (double e : vals2)
			System.out.println("g[" + (count++) + "]=" + e);
	}
}

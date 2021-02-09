/**
 * 
 */
package ceka.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import ceka.converters.FileSaver;

/**
 * some stochastic function
 *
 */
public class Stochastics {
	/**
	 * fact n!
	 * @param n
	 * @return
	 */
	public static double fact(long n)
	{
		double x = 1;
		for (int i = 1; i <= n; i++)
			x *= i;
		return x;
	}
	
	/**
	 * combination m! / (n!(m-n)!)
	 * @param m
	 * @param n
	 * @return
	 */
	public static double combination(int m, int n)
	{
		//m!/n!(m-n)!
		return fact(m) / (fact(n) * fact(m-n));
	}
	
	/**
	 * use binomial model to integrate labels
	 * @param nW number of workers
	 * @param p probability to provide right answer
	 * @return
	 */
	public static double binomialIntegration(int nW, double p) {
		int N = nW/2;
		double r = 0.0;
		for (int i =0; i <= N; i++)
			r += (combination(2 * N + 1, i)*Math.pow(p, 2 * N + 1 - i)* Math.pow(1-p, i));
		return r;
	}
	
	/**
	 * class ComponentTuple is used for generating random sequence
	 * @author jzhang
	 *
	 */
	public class ComponentTuple {
		
		public ComponentTuple() {
			name = "err";
			prob = -1.0;
		}
		public void setValues(String str, double p) {
			name = str;
			prob = p;
		}
		public String name;
		public double prob;
	}
	
	/**
	 * generate sequence 
	 * @param seqLen the length of the sequence
	 * @param correct correct label and its proportion
	 * @param remain remain error label and their proportion
	 * @return
	 */
	public static ArrayList<String> generateRandomSequence(long seqLen, ComponentTuple correct
			, ArrayList<ComponentTuple> remain)
	{
		ArrayList<String> sequence = new ArrayList<String>();
		if (seqLen <= 0)
			return sequence;
		int lengthOfRemain = remain.size();
		double sigmaRemainProb = 0.0;
		for (int i = 0; i < lengthOfRemain; i++) {
			sigmaRemainProb += remain.get(i).prob;
		}
		if (!Misc.isDoubleSame(sigmaRemainProb, 1.0, 0.0001)) {
			log.error("the sum of the probs of remain is not equal to 1.0.");
			return sequence;
		}
		
		long correctNum = Math.round((double)seqLen * correct.prob);
		long remainNum = seqLen - correctNum;
		
		for (int i = 0; i < correctNum; i++) {
			sequence.add(correct.name);
		}
		
		long [] remainVector = new long[remain.size()];
		long sumRemain = 0;
		for (int i = 0; i < remain.size() - 1; i++) {
			remainVector[i] = Math.round((double)remainNum * remain.get(i).prob);
			sumRemain += remainVector[i];
		}
		remainVector[remain.size() - 1] = remainNum - sumRemain;
		
		for (int i = 0; i < remain.size(); i++) {
			for (int j = 0; j < remainVector[i]; j++)
				sequence.add(remain.get(i).name);
		}
		
		for (int i = 0; i < 7; i++)
			Collections.shuffle(sequence, new Random());
		
		assert sequence.size() == seqLen;
		
		return sequence;
	}

	
	@SuppressWarnings("unused")
	static private Logger log = LogManager.getLogger(Stochastics.class);
}

package ceka.consensus.ds;

import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.utils.IdDecorated;

public class DSExample  implements IdDecorated {

	/**
	 * create a DSExample
	 * @param e
	 * @param cateNum number of categories
	 */
	public DSExample (Example e, int cateNum) {
		example = e;
		this.cateNum = cateNum;
		cateProb = new double [this.cateNum];
		for (int i = 0; i < this.cateNum; i++)
			cateProb[i] = 1.0 / (double)this.cateNum;
		multiNoisyLabelSet = example.getMultipleNoisyLabelSet(0);
	}
	
	/**
	 * get probability of this example belonging to class c
	 * @param c
	 * @return probability
	 */
	public double getCateProb(int c){
		return cateProb[c];
	}
	
	/**
	 * set probability of this example belonging to class c
	 * @param c
	 * @param prob
	 */
	public void setCateProb(int c, double prob) {
		cateProb[c] = prob; 
	}
	
	public MultiNoisyLabelSet getMultiNoisyLabelSet() {
		return multiNoisyLabelSet;
	}
	
	@Override
	public String getId() {
		// TODO Auto-generated method stub
		return example.getId();
	}
	
	public void generateIntegratedLabel(String name) {
		int result = 0;
		double maxProbability = -1;
		for (int j = 0; j < cateNum; j++) {
			Double probability = cateProb[j];
			if (probability > maxProbability) {
				maxProbability = probability;
				result = j;
			}
			else if (probability == maxProbability) {
				if (Math.random() > 0.5) {
					maxProbability = probability;
					result = j;
				}
			}	
		}
		Label integratedLabel = new Label(null, new Integer(result).toString(), example.getId(), name);
		example.setIntegratedLabel(integratedLabel);
	}
	
	private Example example = null;
	private int cateNum = 2;
	private double [] cateProb = null; // probability for every class
	private MultiNoisyLabelSet multiNoisyLabelSet = null;
}

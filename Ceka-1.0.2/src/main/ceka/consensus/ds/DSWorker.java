/**
 * 
 */
package ceka.consensus.ds;

import ceka.core.MultiNoisyLabelSet;
import ceka.core.Worker;
import ceka.utils.IdDecorated;

/**
 * Worker for Dawid & Skene algorithm
 * @see Dawid, A. P., & Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates using the EM algorithm. Applied statistics, 20-28.
 */

public class DSWorker implements IdDecorated {
	
	class ConfusionMatrix
	{
		ConfusionMatrix(int cateNum)
		{
			this.cateNum = cateNum;
			element = new double[cateNum][];
			for (int i = 0; i < cateNum; i++)
				element[i] = new double [cateNum];
			for (int i = 0; i < cateNum; i++)
				for (int j = 0; j < cateNum; j++)
					element[i][j] = 0.0;
		}
		
		void uniform1()
		{
			double [] sum = new double [cateNum];
			for (int i = 0; i < cateNum; i++) {
				sum[i] = 0.0;
				for (int j = 0; j < cateNum; j++) {
					sum[i] += element[i][j];
				}
			}
			for (int i = 0; i < cateNum; i++) {
				for (int j = 0; j < cateNum; j++) {
					if (sum[i] != 0) {
						element[i][j] /= sum[i];
					} else {
						element[i][j] = Double.NaN;
					}
				}
			}
		}
		
		int  cateNum = 0;
		double [][] element = null;
	}
	
	/**
	 * create a DSWorker
	 * @param w
	 * @param numCate number of categories
	 */
	public DSWorker(Worker w, int numCate) {
		worker = w;
		this.numCate = numCate;
		this.cmatrix = new ConfusionMatrix(numCate);
		multiNoisyLabelSet = w.getMultipleNoisyLabelSet(0);
	}

	public void initializeConfusionMatrix() {
		for (int i = 0; i < numCate; i++)
			for (int j = 0; j < numCate; j++) {
				if (i == j)
					cmatrix.element[i][j] = 0.9;
				else
					cmatrix.element[i][j] = 0.1/(numCate - 1);
			}
	}
	
	public void randomInitializeConfusionMatrix () {
		for (int i = 0; i < numCate; i++){
			double [] rlist = new double[numCate];
			double sum = 0.0;
			for (int j = 0; j < numCate; j++) {
				rlist[j] = Math.random();
				sum += rlist[j];
			}
			for (int j = 0; j < numCate; j++) {
				cmatrix.element[i][j] = rlist[j] / sum;
			}
		}
	}
	
	public MultiNoisyLabelSet getMultiNoisyLabelSet() {
		return multiNoisyLabelSet;
	}
	
	public void setNewConfusionMatrix(ConfusionMatrix cm) {
		cmatrix = cm;
	}
	
	public String getId() {
		return worker.getId();
	}
	
	public double getCMValue(int i, int j) {
		return cmatrix.element[i][j];
	}
	
	public void printConfusionMatric() {
		System.out.print("[");
		for (int i = 0; i < numCate; i++) {
			for (int j = 0; j < numCate; j++) {
				System.out.print(cmatrix.element[i][j] + "  ");
			}
		}
		System.out.print("]\n");
	}
	
	public double[][] getCM() {
        return cmatrix.element;
    }
	
	private Worker worker = null;
	private ConfusionMatrix cmatrix = null;
	private int numCate = 2;
	private MultiNoisyLabelSet multiNoisyLabelSet = null;
}

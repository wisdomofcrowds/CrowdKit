package ceka.consensus.wtcmm;

import java.util.ArrayList;

import ceka.core.Label;
import ceka.core.Worker;
import ceka.utils.IdDecorated;

public class WTWorker implements IdDecorated {

	@Override
	public String getId() {
		return worker.getId();
	}

	public WTWorker(Worker w, int nCate) {
		worker = w;
		numCate = nCate;
		initializePi();
	}
	
	private void initializePi() {
		pi = new double[numCate][];
		for (int i = 0; i < numCate; i++)
			pi[i] = new double [numCate];
		double diagonal = 0.8;
		for (int i = 0; i < numCate; i++)
			for (int j = 0; j < numCate; j++) {
				if (i == j)
					pi[i][j] = diagonal;
				else
					pi[i][j] = (1-diagonal)/ (numCate - 1);
			}
	}
	
	public void printConfusionMatrix() {
		System.out.print("[");
		for (int i = 0; i < numCate; i++) {
			for (int j = 0; j < numCate; j++) {
				System.out.print(pi[i][j] + "  ");
			}
		}
		System.out.print("]\n");
	}
	
	public void m_step(ArrayList<WTExample> examples) {
		double [][] newpi = new double[numCate][];
		for (int i = 0; i < numCate; i++) {
			newpi[i] = new double [numCate];
			for (int j = 0; j < numCate; j++) 
				newpi[i][j] = 0;
		}
		for (WTExample example: examples) {
			Label l = worker.getMultipleNoisyLabelSet(0).getNoisyLabelByExampleId(example.getId());
			if (l != null)
				newpi[example.estimatedY][l.getValue()] += 1;
		}
		for (int k = 0; k < numCate; k++) {
			double s = 0;
			for (int q = 0; q < numCate; q++)
				s += newpi[k][q];
			if (s != 0)
				for (int d = 0; d < numCate; d++)
					pi[k][d] = newpi[k][d] / s;
		}
	}
	
	private int numCate = 2;
	private Worker worker = null;
	public double [][] pi = null;
}

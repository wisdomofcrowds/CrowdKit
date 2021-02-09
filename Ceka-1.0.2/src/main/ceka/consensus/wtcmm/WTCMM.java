package ceka.consensus.wtcmm;

import java.util.ArrayList;

import ceka.core.Category;
import ceka.core.Dataset;

public class WTCMM {

	public static final String NAME = "WTCMM";
	
	public WTCMM(int maxIter) {
		
		maxIteration = maxIter;
	}
	
	public void doInference(Dataset data) {
		initialize(data);
		double currentLikelihood = loglikelihood();
		System.out.println("log-likelihood = " + currentLikelihood);
		int count = 0;
		while (count < maxIteration) {
			e_step();
			m_step();
			currentLikelihood = loglikelihood();
			System.out.println("log-likelihood = " + currentLikelihood);
			count++;
		}
		data.assignIntegeratedLabel2WekaInstanceClassValue();
	}
	
	private double loglikelihood() {
		double like = 0;
		for (WTExample e: examples) {
			double l = e.computeLikelihood(workers, categories);
			//System.out.println("likelihood [" + e.getId() + "]="+l);
			like += Math.log(l);
		}
		return like;
	}
	
	private void e_step() {
		for (WTExample e: examples) {
			e.e_step(workers, categories);
		}
	}
	
	private void m_step() {
		// compute pk
		double [] probK = new double[categories.size()];
		for (int i = 0; i < examples.size(); i++)
			probK[examples.get(i).estimatedY]+=1;
		for (int i = 0; i < categories.size(); i++)
			categories.get(i).setProbability(probK[categories.get(i).getValue()]/ examples.size());
		for (int j = 0; j < workers.size(); j++)
			workers.get(j).m_step(examples);
		for (int i = 0; i < examples.size(); i++)
			examples.get(i).m_step();
	}
	
	private void initialize(Dataset data) {
		int numCate = data.getCategorySize();
		for (int k = 0; k < numCate; k++) {
			Category cate = data.getCategory(k);
			cate.setProbability(1.0/numCate);
			categories.add(cate);
			System.out.println("WTCMM add cate = "+ cate.getValue());
		}
		int numWorker = data.getWorkerSize();
		for (int j = 0; j < numWorker; j++) {
			WTWorker w = new WTWorker(data.getWorkerByIndex(j), numCate);
			workers.add(w);
		}
		int numExample = data.getExampleSize();
		for (int i = 0; i < numExample; i++) {
			WTExample e = new WTExample(data.getExampleByIndex(i), numCate);
			examples.add(e);
		}
	}
	
	private int maxIteration = 50;
	private ArrayList<Category> categories = new ArrayList<Category>();
	private ArrayList<WTWorker> workers = new ArrayList<WTWorker>();
	private ArrayList<WTExample> examples = new ArrayList<WTExample>();
}

/**
 * 
 */
package ceka.consensus.ds;

import java.util.ArrayList;

import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.utils.Misc;


/**
 * Article:<br>
 * Alexander Philip Dawid and Allan M Skene. Maximum likelihood estimation of observer
error-rates using the em algorithm. Applied statistics, pages 20 - 28, 1979.
 * @author Zhang
 */
public class DawidSkene {

	public static final String NAME = "DS";
	
	public DawidSkene(int maxIter) {
		maxIteration = maxIter;
	}
	
	public void doInference(Dataset data) {
		initialize(data);
		currentLikelihood = loglikelihood();
		int ncounter = 0;
		log.info("Initial Likelihood (" + ncounter + "): " + currentLikelihood);
		while ((Math.abs(currentLikelihood - oldLikelihood) > epsion) && (ncounter < maxIteration))
		{
			oldLikelihood = currentLikelihood;
			updateExampleProbabilities();
			updateCategoryPrior();
			updateWorkerConfusionMatrix();
			currentLikelihood = loglikelihood();
			ncounter++;
			log.info("Current Likelihood (" + ncounter + "): " + currentLikelihood);
		}
		for (int i = 0; i < examples.size(); i++)
			examples.get(i).generateIntegratedLabel(NAME);
		// this is important
		data.assignIntegeratedLabel2WekaInstanceClassValue();
	}
	
	private void initialize(Dataset data) {
		int numCate = data.getCategorySize();
		for (int j = 0; j < numCate; j++) {
			Category cate = data.getCategory(j).copy();
			cate.setProbability(1.0 / numCate);
			categories.add(cate);
		}
		int numWorker = data.getWorkerSize();
		for (int k = 0; k < numWorker; k++) {
			DSWorker w = new DSWorker(data.getWorkerByIndex(k), numCate);
			w.initializeConfusionMatrix();
			workers.add(w);
		}
		int numExample = data.getExampleSize();
		for (int i = 0; i < numExample; i++) {
			examples.add(new DSExample(data.getExampleByIndex(i), numCate));
		}
	}
	
	private void updateCategoryPrior() {
		for (int j = 0; j < categories.size(); j++) {
			Category cate = categories.get(j);
			double prob = 0.0;
			for (int i = 0; i < examples.size(); i++)
				prob += examples.get(i).getCateProb(cate.getValue());
			cate.setProbability(prob/(double)examples.size());
			// log.info("updateCategoryPrior | category: " + cate.getValue() + " Prior=" + cate.getProbability());
		}
	}
	
	private void updateWorkerConfusionMatrix() {
		for (int k = 0; k < workers.size(); k++) {
			DSWorker w = workers.get(k);
			MultiNoisyLabelSet assignedLabels = w.getMultiNoisyLabelSet();
			DSWorker.ConfusionMatrix cmatrix = w.new ConfusionMatrix(categories.size());
			
			for (int i = 0; i < assignedLabels.getLabelSetSize(); i++) {
				Label l = assignedLabels.getLabel(i);
				String exampleId = l.getExampleId();
				DSExample e = getDSExampleById(exampleId);
				// different from 1978 paper
				double [] probs = updateProbabilities(e, w.getId());
				if (probs == null)
					continue;
				for (int j = 0; j < categories.size(); j++)
					cmatrix.element[j][l.getValue()] += probs[j];
			}
			cmatrix.uniform1();
			w.setNewConfusionMatrix(cmatrix);
			//w.printConfusionMatric();
		}
		log.info("  ");
	}
	
	private double loglikelihood() {
		double like = 0.0;
		for (int i = 0; i < examples.size(); i++) {
			DSExample e = examples.get(i);
			for (int j = 0; j < categories.size(); j++) {
				for (int ni = 0; ni < e.getMultiNoisyLabelSet().getLabelSetSize(); ni++) {
					Label noisylabel = e.getMultiNoisyLabelSet().getLabel(ni);
					DSWorker w = getDSWorkerById(noisylabel.getWorkerId());
					double labelingProb = w.getCMValue(j, noisylabel.getValue());
					double categoryProb = e.getCateProb(j);
					if (labelingProb == 0 || categoryProb == 0 || Double.isNaN(labelingProb))
						continue;
					like += (Math.log(labelingProb) + Math.log(categoryProb));
				}
			}
		}
		return like;
	}
	
	private void updateExampleProbabilities() {
		for (int i = 0; i < examples.size(); i++) {
			DSExample e = examples.get(i);
			double [] probs = updateProbabilities(e, null);
			if (probs == null)
				continue;
			for (int j = 0; j < categories.size(); j++)
				e.setCateProb(j, probs[j]);
		}
	}
	
	private double [] updateProbabilities(DSExample e, String wkIdToIngore) {
		double denominator = 0.0;
		double [] nominator = new double[categories.size()];
		
		if ((wkIdToIngore != null) && (e.getMultiNoisyLabelSet().getLabelSetSize() == 1)) 
			if (e.getMultiNoisyLabelSet().getLabel(0).getWorkerId().equals(wkIdToIngore)) 
					return null;
		
		for(int j = 0; j < categories.size(); j++) {
			double cateM = categories.get(j).getProbability();
			int ns = e.getMultiNoisyLabelSet().getLabelSetSize();
			for (int i = 0; i < ns; i++) {
				Label l = e.getMultiNoisyLabelSet().getLabel(i);
				DSWorker w = getDSWorkerById(l.getWorkerId());
				if ((wkIdToIngore != null) && wkIdToIngore.equals(w.getId()))
					continue;
				double pi = w.getCMValue(j, l.getValue());
				if (Double.isNaN(pi))
					continue;
				cateM *= pi;	
			}
			nominator[j] = cateM;
			denominator += cateM;
		}
		
		if (denominator == 0) {
			log.debug("denominator = 0");
			return null;
		}
		
		double [] result = new double[categories.size()];
		for (int j = 0; j < categories.size(); j++) {
			Double probability = Misc.round(nominator[j] / denominator, 5);
			result[j] = probability;
		}
		return result;
	}
	
	private DSWorker getDSWorkerById(String id) {
		return Misc.getElementById(workers, id);
	}
	
	public ArrayList<DSWorker> getWorkers() {
        return workers;
    }
	
	private DSExample getDSExampleById(String id) {
		return Misc.getElementById(examples, id);
	}
	
	public void printAllConfusionMatrices() {
		for (DSWorker w: workers)
			w.printConfusionMatric();
	}
	
	private ArrayList<DSExample>  examples = new ArrayList<DSExample>();
	private ArrayList<DSWorker>   workers = new ArrayList<DSWorker>();
	private ArrayList<Category>   categories = new ArrayList<Category>();
	
	private double oldLikelihood = 0.0;
	private double currentLikelihood = 0.0;
	private double epsion = 10E-5;
	private int    maxIteration = 50;
	
	static private Logger log = LogManager.getLogger(DawidSkene.class);
}

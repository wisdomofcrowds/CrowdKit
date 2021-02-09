package ceka.consensus.wtcmm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import ceka.core.Category;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.utils.IdDecorated;
import ceka.utils.Misc;

public class WTExample implements IdDecorated {

	@Override
	public String getId() {
		return example.getId();
	}
	
	public WTExample(Example e, int nCate) {
		numCate = nCate;
		example = e;
		jointKZ = new JointKZ[numCate];
		for (int i = 0; i < numCate; i++)
			jointKZ[i] = new JointKZ();
		mnls = e.getMultipleNoisyLabelSet(0);
		for (int i = 0; i < mnls.getLabelSetSize(); i++) {
			Label label = mnls.getLabel(i);
			String wId = label.getWorkerId();
			estimatedZ.put(wId, new Integer(0));
			//w1.put(wId, new Double(0.5));
			for (int k = 0; k < numCate; k++) {
				jointKZ[k].category = k;
				jointKZ[k].z1.put(wId, new Double(0.0));
				jointKZ[k].z2.put(wId, new Double(0.0));
			}
		}
		initializePsi();
	}
	
	public double computeLikelihood(ArrayList<WTWorker> workers, ArrayList<Category> cates) {
		double [] likeK = new double[cates.size()];
		for (int k = 0; k < cates.size(); k++) {
			double prod = 1.0;
			ArrayList<String> wList = example.getWorkerIdList();
			for (String wId: wList) {
				WTWorker w = FindWTWorker(workers, wId);
				Label label = mnls.getNoisyLabelByWorkerId(wId);
				int valK = cates.get(k).getValue();
				int valD = label.getValue();
				double wj1 = w1;
				double wj2 = 1 - wj1;
				prod *= (wj1*w.pi[valK][valD] + wj2*psi[valK][valD]);
			}
			likeK[cates.get(k).getValue()] = cates.get(k).getProbability()*prod;
		}
		likelihood = 0;
		for (int k = 0; k < likeK.length; k++)
			likelihood += likeK[k];
		return likelihood;
	}
	
	public void e_step(ArrayList<WTWorker> workers, ArrayList<Category> cates) {
		for (int x = 0; x < cates.size(); x++) {
			int k = cates.get(x).getValue();
			double pk = cates.get(x).getProbability();
			ArrayList<String> wList = example.getWorkerIdList();
			for (String wId: wList) {
				WTWorker w = FindWTWorker(workers, wId);
				Label label = mnls.getNoisyLabelByWorkerId(wId);
				int valD = label.getValue();
				jointKZ[k].z1.put(wId, pk*w.pi[k][valD]);
				jointKZ[k].z2.put(wId, pk*psi[k][valD]);
			}
		}
		computeEstimatedY();
		computeEstimatedZ();
	}
	
	public void printConfusionMatrix() {
		System.out.print("[");
		for (int i = 0; i < numCate; i++) {
			for (int j = 0; j < numCate; j++) {
				System.out.print(psi[i][j] + "  ");
			}
		}
		System.out.print("]\n");
	}
	
	private void initializePsi() {
		psi = new double[numCate][];
		for (int i = 0; i < numCate; i++)
			psi[i] = new double [numCate];
		double diagonal = 0.8;
		for (int i = 0; i < numCate; i++)
			for (int j = 0; j < numCate; j++) {
				if (i == j)
					psi[i][j] = diagonal;
				else
					psi[i][j] = (1-diagonal)/ (numCate - 1);
			}
	}
	
	private WTWorker FindWTWorker(ArrayList<WTWorker> workers, String wId) {
		for (int i = 0; i < workers.size(); i++)
			if (workers.get(i).getId().equals(wId))
				return workers.get(i);
		return null;
	}
	
	private void computeEstimatedY() {
		for (int i = 0; i < jointKZ.length; i++)
			jointKZ[i].computeLikeY();
		double [] listY = new double[jointKZ.length];
		for (int i = 0; i < jointKZ.length; i++)
			listY[jointKZ[i].category] = jointKZ[i].likeY;
		estimatedY = Misc.findMaxPositionRand(listY);
		Label integratedL = new Label(null, new Integer(estimatedY).toString(), example.getId(), "WTCMM");
		example.setIntegratedLabel(integratedL);
	}
	
	private void computeEstimatedZ() {
		for (Map.Entry<String, Integer> entry : estimatedZ.entrySet()) {
			double z1prob = 0.0;
			double z2prob = 0.0;
			for (int k = 0; k < jointKZ.length; k++) {
				z1prob += jointKZ[k].z1.get(entry.getKey());
				z2prob += jointKZ[k].z2.get(entry.getKey());
			}
			if (z1prob >= z2prob)
				estimatedZ.put(entry.getKey(), 1);
			else
				estimatedZ.put(entry.getKey(), 2);
		}
		// for debug
		// printeEstimatedZ();
	}
	
	public void  printeEstimatedZ() {
		System.out.print("Z of ["+getId()+"]={");
		for (Map.Entry<String, Integer> entry : estimatedZ.entrySet()) {
			System.out.print("(" + entry.getKey() + "," + entry.getValue()+"), ");
		}
		System.out.println("}");
	}
	
	public void m_step() {
		int totalZ2 = 0;
		for (Map.Entry<String, Integer> entry : estimatedZ.entrySet()) {
			if (entry.getValue() == 2)
				totalZ2 += 1;
		}
		if (totalZ2 != 0) {
			for (int d = 0; d < numCate; d++) {
				double totalD = 0;
				for (int i = 0;  i < mnls.getLabelSetSize(); i++) {
					if (mnls.getLabel(i).getValue() == d)
						totalD +=1;
				}
				psi[estimatedY][d] = totalD / (double)totalZ2;
			}
		}
		w1 = 1- (double)totalZ2/(double)estimatedZ.size();
	}
	
	private int numCate = 2;
	private Example example = null;
	private double [][] psi = null;
	private HashMap<String, Integer> estimatedZ = new HashMap<String, Integer>();
	private MultiNoisyLabelSet mnls = null;
	private double likelihood = 0;
	private JointKZ [] jointKZ = null;
	public int estimatedY = -1;
	private double w1 = 0.5;
	
	class JointKZ {
		int category = 0;
		HashMap<String, Double> z1 = new HashMap<String, Double>();
		HashMap<String, Double> z2 = new HashMap<String, Double>();
		double likeY = 0.0;
		void computeLikeY() {
			for (Map.Entry<String, Double> entry : z1.entrySet())
				likeY += entry.getValue();
			for (Map.Entry<String, Double> entry : z2.entrySet())
				likeY += entry.getValue();
		}
	}
}

package ceka.utils;

import java.util.ArrayList;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.core.Worker;

public class WorkerAnalyzer {

	public class WorkerInfo implements IdDecorated{
		
		public WorkerInfo(String idStr) {
			id = new String(idStr);
		}
		public String getId() {
			return id;
		}
		public void setSpam(boolean flag) {
			isSpam = flag;
		}
		public boolean getSpam() {
			return isSpam;
		}
		
		public void setAcc(double acc) {
			accuracy = acc;
		}
		
		public double getAcc() {
			return accuracy;
		}
		private String id;
		private boolean isSpam = false;
		private double accuracy = 0.0;
	}
	
	public void initialize(Dataset dataset) {
		data = dataset;
		int numWorker = dataset.getWorkerSize();
		for (int i = 0; i < numWorker; i++) {
			infos.add(new WorkerInfo(dataset.getWorkerByIndex(i).getId()));
		}
	}
	
	public ArrayList<WorkerInfo> getWorkerInfo() {
		return infos;
	}
	
	public WorkerInfo getWorkerInfoById (String id) {
		return Misc.getElementById(infos, id);
	}
	
	public void analyzeSpammer () {
		int numWorker = data.getWorkerSize();
		for (int i = 0; i < numWorker; i++) {
			Worker worker = data.getWorkerByIndex(i);
			MultiNoisyLabelSet mnls = worker.getMultipleNoisyLabelSet(0);
			int c = isConsistent(mnls);
			if ((c != 0 ) && (mnls.getLabelSetSize() >= spamThresholdNum)) {
				// spam
				WorkerInfo wi = Misc.getElementById(infos, worker.getId());
				wi.setSpam(true);
				System.out.println("Woker Id = " + wi.getId() + " is spam.");
			}
		}
	}
	
	public void analyzeAccuracy () {
		int numWorker = data.getWorkerSize();
		for (int i = 0; i < numWorker; i++) {
			Worker worker = data.getWorkerByIndex(i);
			MultiNoisyLabelSet mnls = worker.getMultipleNoisyLabelSet(0);
			double total = mnls.getLabelSetSize();
			double correct = 0;
			for (int j = 0; j < mnls.getLabelSetSize(); j++) {
				Label label = mnls.getLabel(j);
				Example example = data.getExampleById(label.getExampleId());
				if (example.getTrueLabel().getValue() == label.getValue())
					correct += 1;
			}
			WorkerInfo wi = Misc.getElementById(infos, worker.getId());
			assert (wi != null);
			wi.setAcc(correct/total);
		}
	}
	
	private int isConsistent(MultiNoisyLabelSet mnls) {
		int ret = -1;
		int numLabel = mnls.getLabelSetSize();
		int preLabel = mnls.getLabel(0).getValue();
		for (int i = 1; i < numLabel; i++) {
			int currLabel = mnls.getLabel(i).getValue();
			if (currLabel != preLabel) {
				ret = 0;
				break;
			}
		}
		if (ret != 0) {
			if (preLabel == 1)
				ret = 1;
		}
		return ret;
	}
	
	private ArrayList<WorkerInfo> infos = new ArrayList<WorkerInfo>();
	private Dataset data = null;
	private int spamThresholdNum = 5;
}

package ceka.noise.avnc;

import java.util.ArrayList;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.core.Worker;

public class WorkerStat {

	class WorkerInfo {
		
		WorkerInfo (Worker w) {
			worker = w;
		}
		
		double calculateEstimatedAcc(Dataset data) {	
			MultiNoisyLabelSet mnls = worker.getMultipleNoisyLabelSet(0);
			int numLabel = mnls.getLabelSetSize();
			int numCorrect = 0;
			for (int i = 0; i < numLabel; i++) {
				Label l = mnls.getLabel(i);
				String exampleId = l.getExampleId();
				Example e = data.getExampleById(exampleId);
				Label trueL = e.getTrueLabel();
				if (trueL.getValue() ==  l.getValue())
					numCorrect++;
			}
			estimatedAcc = (double)numCorrect / (double)numLabel;
			return estimatedAcc;
		}
		
		Worker worker;
		double estimatedAcc = 0.0;
	}
		
	public WorkerStat() {
		
	}
	
	public double calculateEstimatedMeanAcc(Dataset data) {
		double r = 0;
		int numWorker = data.getWorkerSize();
		for (int k = 0; k < numWorker; k++) {
			Worker w = data.getWorkerByIndex(k);
			WorkerInfo wInfo = new WorkerInfo(w);
			r += wInfo.calculateEstimatedAcc(data);
			workerInfos.add(wInfo);		
		}
		r = r/(double)(numWorker);
		return r;
	}
	
	private ArrayList<WorkerInfo> workerInfos = new ArrayList<WorkerInfo>();
}

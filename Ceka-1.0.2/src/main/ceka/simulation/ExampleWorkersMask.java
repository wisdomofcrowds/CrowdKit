/**
 * 
 */
package ceka.simulation;

import java.util.ArrayList;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.MultiNoisyLabelSet;
import ceka.utils.IdDecorated;
import ceka.utils.Misc;

/**
 * This class is used for selecting noisy labels of each example.
 * After noisy labels have been selected, use FileSaver to dump the 
 * dataset to files. Then, use FileLoader to generate new dataset.
 * @author Zhang
 *
 */
public class ExampleWorkersMask {

	class Mask implements IdDecorated {
		
		class MultiNoisyLabelSetMask implements IdDecorated {

			MultiNoisyLabelSetMask (String labelName) {
				this.labelName = labelName;
			}
			
			/* (non-Javadoc)
			 * @see ceka.utils.IdDecorated#getId()
			 */
			@Override
			public String getId() {
				// TODO Auto-generated method stub
				return labelName;
			}
			
			public void initialize(MultiNoisyLabelSet mnls) {
				for (int i = 0; i < mnls.getLabelSetSize(); i++)
					workerIdList.add(new String (mnls.getLabel(i).getWorkerId()));
			}
			
			public ArrayList<String> getWorkerMask() {
				return workerIdList;
			}
			
			/**
			 * sequentially select number N of noisy labels
			 * @param numberOfLabels
			 */
			public void sequentialSelect(int numberOfLabels) {
				int count = 0;
				for (int i = 0; i < workerIdList.size(); i++) {
					if (count >= numberOfLabels)
						workerIdList.set(i, "");
				}
			}
			
			/**
			 * randomly select number N of noisy labels
			 * @param numberOfLabels
			 */
			public void randSelect(int numberOfLabels) {
				int numWorkers = workerIdList.size();
				ArrayList<Integer> sel = Misc.randSelect(numberOfLabels, 0, numWorkers - 1);
				for (int i = 0; i < workerIdList.size(); i++) {
					Integer index = new Integer(i);
					if (Misc.getElementEquals(sel, index) == null)
						workerIdList.set(index, "");
				}
			}
			
			String labelName = null;
			ArrayList<String> workerIdList = new ArrayList<String> ();
		}

		Mask (String exampleId) {
			this.exampleId = exampleId;
		}
		
		/* (non-Javadoc)
		 * @see ceka.utils.IdDecorated#getId()
		 */
		@Override
		public String getId() {
			return exampleId;
		}
		
		public void initialize (Example e) {
			int numMultiNoisyLabelSet = e.getMultipleNoisyLabelSetSize();
			for (int i = 0; i < numMultiNoisyLabelSet; i++) {
				MultiNoisyLabelSet mnls = e.getMultipleNoisyLabelSet(i);
				MultiNoisyLabelSetMask mask = new MultiNoisyLabelSetMask(mnls.getId());
				mask.initialize(mnls);
				maskList.add(mask);
			}
		}
		
		public ArrayList<String> getWorkerMask(int mnlsIndex) {
			if (mnlsIndex >= maskList.size())
				return null;
			return maskList.get(mnlsIndex).getWorkerMask();
		}
		
		/**
		 * sequentially select number N of noisy labels
		 * @param numberOfLabels
		 * @param mnlsIndex the index of multiple noisy label set
		 */
		public void sequentialSelect(int numberOfLabels, int mnlsIndex) {
			maskList.get(mnlsIndex).sequentialSelect(numberOfLabels);
		}
		
		/**
		 * randomly select number N of noisy labels
		 * @param numberOfLabels
		 * @param mnlsIndex the index of multiple noisy label set
		 */
		public void randSelect(int numberOfLabels, int mnlsIndex) {
			maskList.get(mnlsIndex).randSelect(numberOfLabels);
		}
		
		ArrayList<MultiNoisyLabelSetMask> maskList = new ArrayList<MultiNoisyLabelSetMask>();
		String exampleId;
	}
	
	public ExampleWorkersMask() {
		
	}
	
	/**
	 * initialize Example Mask with a dataset 
	 * @param data
	 */
	public void initialize(Dataset data) {
		int numExample = data.getExampleSize();
		for (int i = 0; i  < numExample; i++) {
			Example e = data.getExampleByIndex(i);
			Mask mask = new Mask(e.getId());
			mask.initialize(e);
			exampleMaskList.add(mask);
		}
	}
	
	/**
	 * get worker id mask of first multiple noisy label set
	 * @param exampleId
	 * @return
	 */
	public ArrayList<String> getWorkerMask(String exampleId) {
		Mask mask = Misc.getElementById(exampleMaskList, exampleId);
		if (mask == null)
			return null;
		return mask.getWorkerMask(0);
	}
	
	/**
	 * sequentially select number N of noisy labels
	 * @param numberOfLabels
	 */
	public void sequentialSelect(int numberOfLabels) {
		for (int i = 0; i < exampleMaskList.size(); i++)
			exampleMaskList.get(i).sequentialSelect(numberOfLabels, 0);
	}
	
	/**
	 * randomly select number N of noisy labels
	 * @param numberOfLabels
	 */
	public void randSelect(int numberOfLabels) {
		for (int i = 0; i < exampleMaskList.size(); i++)
			exampleMaskList.get(i).randSelect(numberOfLabels, 0);
	}
	
	private ArrayList<Mask> exampleMaskList = new  ArrayList<Mask>();
}

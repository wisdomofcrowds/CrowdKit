package ceka.core;

import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Random;

import ceka.utils.Misc;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.SparseInstance;

/**
 * Dataset is a sub class of weka.core.Instances.
 */
public class Dataset extends weka.core.Instances {

	/**
	 * for serialization
	 */
	private static final long serialVersionUID = -6124363789274455770L;
	
	public Dataset(Instances dataset) {
		super(dataset);
		categorySets.add(new ArrayList<Category>());
	}

	public Dataset(Instances dataset, int capacity) {
		super(dataset, capacity);
		categorySets.add(new ArrayList<Category>());
	}
	
	/**
	 * create an empty dataset
	 */
	public Dataset(String name, FastVector attInfo, int capacity) {
		super(name, attInfo, capacity);
		categorySets.add(new ArrayList<Category>());
	}
	
	/**
	 * Create a data set for a arff file reader
	 * @param reader
	 * @throws IOException
	 */
	public Dataset(Reader reader) throws IOException {
		super(reader);
		categorySets.add(new ArrayList<Category>());
	}
	
	/**
	 * this is for creating an new empty dataset based on current one
	 * @return
	 */
	public Dataset generateEmpty() {
		return new Dataset(this, 0);
	}
	
	/**
	 * subclass override this to release some thing manually
	 */
	public void release() {
		
	}

	/**
	 * find example index by provided id
	 * @param id
	 * @return index or -1 (not found)
	 */
	public int getExampleIndexById(String id) {
		int index = -1;
		for (int i = 0; i < this.numInstances(); i++)
			if (((Example)this.instance(i)).getId().equals(id)) {
				index = i;
				break;
			}
		return index;
	}
	
	/**
	 * get example by provided Id
	 * @param id
	 * @return an Example object or null if not found
	 */
	public Example getExampleById(String id) {
		return Misc.getElementById(examples, id);
	}
	
	/**
	 * get number of workers
	 * @return number of workers
	 */
	public int getWorkerSize() {
		return workers.size();
	}
	
	/**
	 * get a worker by provided Id
	 * @param id
	 * @return a Worker object or null if not found
	 */
	public Worker getWorkerById(String id) {
		return Misc.getElementById(workers, id);
	}
	
	/**
	 * get a worker by index
	 * @param index
	 * @return a Worker object or null if not found
	 */
	public Worker getWorkerByIndex(int index) {
		return workers.get(index);
	}
	
	/**
	 * add an example to dataset
	 * @param e example cannot be null
	 */
	public void addExample(Example e) {
		if (getExampleById(e.getId()) == null) {
			// when we add an instance to weka Instances, it actually
			// add a copy of this instance. So, we must add this instance
			// to Weka Instance first, than using the copied instance.
			super.add(e);
			int lastInstanceIndex = this.numInstances() - 1;
			Example addE = (Example)this.instance(lastInstanceIndex);
			assert addE.getId().equals(e.getId());
			examples.add(addE);
		}
	}
	
	/**
	 * remove last example of the dataset
	 */
	public void removeLastExample() {
		super.delete(super.numInstances() - 1);
		examples.remove(examples.size() - 1);
	}
	
	/**
	 * add a worker to dataset
	 * @param w worker cannot be null
	 */
	public void addWorker(Worker w) {
		if (getWorkerById(w.getId()) == null)
			workers.add(w);
	}
	
	/**
	 * add a category to dataset
	 * @param c category cannot be null
	 */
	public void addCategory(Category c) {
		if (Misc.getElementEquals(categorySets.get(0), c) == null)
			categorySets.get(0).add(c);
	}
	
	/**
	 * get number of examples
	 * @return number of examples
	 */
	public int getExampleSize() {
		return examples.size();
	}
	
	/**
	 * get all examples of the dataset
	 * @return example list
	 */
	public ArrayList<Example> getExamples() {
		return examples;
	}
	
	/**
	 * get example at specific position
	 * @param index index must be in the range
	 * @return example stored in position $index
	 */
	public Example getExampleByIndex(int index) {
		return examples.get(index);
	}
	
	/**
	 * get Category Set size. For single label, the result is 1.
	 * @return the size of Category Set
	 */
	public int getCategorySetSize() {
		return categorySets.size();
	}
	
	/**
	 * get a category set
	 * @param index
	 * @return category list
	 */
	public ArrayList<Category> getCategorySet(int index) {
		return categorySets.get(index);
	}
	
	/**
	 * get the first categorySet size. This is for convenience of single label. 
	 * @return the first category set size.
	 */
	public int getCategorySize() {
		return getCategorySizeML(0);
	}
	
	/**
	 * get the index-th category in first category set. This is for convenience of single label.
	 * @param index
	 * @return the retrieved category object.
	 */
	public Category getCategory(int index) {
		return getCategoryML(0, index);
	}
	
	/**
	 * get the size of index-th category set. For both single and multiple label.
	 * @param index the index-th category set.
	 */
	public int getCategorySizeML(int index) {
		return categorySets.get(index).size();
	}
	
	/**
	 * get a specific category object in a specific category set. For both single and multiple label.
	 * @param cateSetIndex the index of category set.
	 * @param posIndex the position index of the retrieving object.
	 * @return
	 */
	public Category getCategoryML (int cateSetIndex, int posIndex) {
		return categorySets.get(cateSetIndex).get(posIndex);
	}
	
	/**
	 * make Weka Instance Class Value equals to the example's Integrated Label
	 */
	public void assignIntegeratedLabel2WekaInstanceClassValue() {
		for (Example example : examples) {
			example.assignIntegeratedLabel2WekaInstanceClassValue();
		}
	}
	
	/**
	 * make Weka Instance Class Value equals to the example's True Label
	 */
	public void assignTrueLabel2WekaInstanceClassValue() {
		for (Example example : examples) {
			example.assignTrueLabel2WekaInstanceClassValue();
		}
	}
	
	/**
	 * randomize the examples in the dataset 
	 */
	 public void randomize(Random random){
	    for (int j = examples.size() - 1; j > 0; j--){
	    	int second = random.nextInt(j + 1);
	    	super.swap(j, second);
	    	swapExample(j, second);
	    }
	}
	
	/**
	 * simply remove an example by its index
	 * @param index
	 */
	public void simpleRemoveExampleByIndex (int index) {
		this.delete(index);
		examples.remove(index);
	}
	
	/**
	 * get a worker's accuracy of labeling on a multiple noisy label set by index
	 * @param wId worker's id
	 * @param multipleNosiyLabelSetIndex index of a multiple noisy label set
	 * @return
	 */
	public double getWorkerAccuracy(String wId, int multipleNosiyLabelSetIndex) {
		Worker w = this.getWorkerById(wId);
		MultiNoisyLabelSet mnls = w.getMultipleNoisyLabelSet(multipleNosiyLabelSetIndex);
		double total = (double) (mnls.getLabelSetSize());
		double correct = 0;
		for (int i = 0; i < mnls.getLabelSetSize(); i++) {
			Label l = mnls.getLabel(i);
			Label trueL = this.getExampleById(l.getExampleId()).getTrueLabel();
			if (trueL == null) return -1;
			if (l.getValue() == trueL.getValue())
				correct += 1;
		}
		return correct/total;
	}
	
	/**
	 * delete a worker with wId. All related labels given by this worker will be removed
	 * @param wId
	 */
	public void delWorker(String wId) {
		Worker w = Misc.getElementById(workers, wId);
		if (w == null)
			return;
		// find all related examples
		MultiNoisyLabelSet mnls = w.getMultipleNoisyLabelSet(0);
		for (int i = 0; i < mnls.getLabelSetSize() ; i++) {
			Label label = mnls.getLabel(i);
			String exampleId = label.getExampleId();
			for (int j = 0; j < examples.size(); j++) {
				Example e = examples.get(j);
				if (e.getId().equals(exampleId)) {
					// delete related label in the example
					e.delNoisyLabelByWorkerId(wId);
					// check if this example has no noisy labels
					if (e.getMultipleNoisyLabelSet(0).getLabelSetSize() == 0) {
						// delete this examples
						simpleRemoveExampleByIndex(j);
					}
				}
			}
		}
		// delete this wId
		Misc.delElementById(workers, wId);
	}
	
	/**
	 * get count number by true categories
	 * @return
	 */
	public int [] getCountersByTrueCategories()
	{
		int cateNum = getCategorySize();
		int [] list = new int[cateNum];
		for (int i = 0 ; i < cateNum; i++) {
			list[i] = 0;
		}
		for (Example example : examples) {
			list[example.getTrueLabel().getValue()]++;
		}
		return list;
	}
	
	public ArrayList<Worker> getWorkerListByExampleId(String examId) {
		Example e = getExampleById(examId);
		ArrayList<Worker> workerList = new ArrayList<Worker>();
		ArrayList<String> workerIdList = e.getWorkerIdList();
		for (int i = 0; i < workerIdList.size(); i++)
			workerList.add(getWorkerById(workerIdList.get(i)));
		return workerList;
	}
	
	/*
	 * List of category set used both multi-label and signal label
	 */
	protected ArrayList<ArrayList<Category>> categorySets = new ArrayList<ArrayList<Category>>();
	private ArrayList<Example> examples = new ArrayList<Example>();
	private ArrayList<Worker>  workers = new ArrayList<Worker>();
	
	private final void swapExample(int first, int second) {

		Example help = examples.get(first);

	    examples.set(first, examples.get(second));
	    examples.set(second, help);
	  }
}

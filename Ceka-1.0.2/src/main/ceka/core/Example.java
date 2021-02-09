package ceka.core;

import java.util.ArrayList;

import ceka.utils.IdDecorated;
import ceka.utils.Misc;

/**
 * An Example is an weka Instance
 *
 */
public class Example extends weka.core.SparseInstance implements IdDecorated {

	/**
	 * for serialization
	 */
	private static final long serialVersionUID = -6312687588459714005L;

	/**
	 * Create Example from weka Instance
	 * @param instance
	 */
	public Example(weka.core.Instance instance) {
		super(instance);
		id = new Integer(this.hashCode()).toString() + "-" + new Integer(instance.hashCode()).toString();
		// create a MultipleNoisyLabelSet with default name
		multiNoisyLabelSets.add(new MultiNoisyLabelSet(null));
		categorySets.add(new ArrayList<Category>());
	}
	
	/**
	 * Create Example from weka Instance
	 * @param instance
	 */
	public Example(weka.core.Instance instance, String idStr) {
		super(instance);
		if (idStr == null)
			id = new Integer(this.hashCode()).toString() + "-" + new Integer(instance.hashCode()).toString();
		else
			id = idStr;
		// create a MultipleNoisyLabelSet with default name
		multiNoisyLabelSets.add(new MultiNoisyLabelSet(null));
		categorySets.add(new ArrayList<Category>());
	}
	
	/**
	 * Create Example at least with one attribute that is the true label
	 * @param numAttributes must greater than 0
	 * @param id example id
	 */
	public Example(int numAttributes, String id) {
		super(numAttributes);
		this.id = new String(id);
		// create a MultipleNoisyLabelSet with default name
		multiNoisyLabelSets.add(new MultiNoisyLabelSet(null));
		categorySets.add(new ArrayList<Category>());
	}
	
	/**
	 * @return the shallow copy
	 */
	public Object copy() {
		Example e = new Example(this);
		return e;
	}
	
	public Object deepCopy () {
		int m_NumAttributes = this.m_NumAttributes;
		double []  m_AttValues = this.m_AttValues;
		int []    m_Indices = this.m_Indices;
		double m_Weight = this.m_Weight;
		
		Example e = new Example(m_Weight, m_AttValues, m_Indices, m_NumAttributes);
		e.setId(this.getId() + e.hashCode());
		e.setDataset(this.dataset());
		
		// copy categorySets
		for (ArrayList<Category> categroySet : this.categorySets) {
			ArrayList<Category> cateSet = new ArrayList<Category>();
			for (Category cate: categroySet) {
				cateSet.add(cate.copy());
			}
			e.categorySets.add(cateSet);
		}
		// copy true labels
		for (Label trueLabel : this.trueLabelSet) {
			e.trueLabelSet.add(trueLabel.copy());
		}
		// copy all multiple noisy label sets
		for (MultiNoisyLabelSet multiNoisyLabelSet : this.multiNoisyLabelSets) {
			e.multiNoisyLabelSets.add(multiNoisyLabelSet.copy());
		}
		return e;
	}
	
	/**
	 * obtain identifier
	 * @return identifier
	 */
	public String getId() {
		return id;
	}
	
	/**
	 * set the identifier
	 * @param id identifier
	 */
    public void setId(String id) {
        this.id = id;
    }
    
	/**
	 * set the true label of this example
	 */
	public void setTrueLabel(Label label) {
		if (trueLabelSet.isEmpty()) {
			trueLabelSet.add(label);
		}
		else {
			trueLabelSet.remove(0);
			trueLabelSet.add(0, label);
		}
	}
	
	/**
	 * get true label of this example
	 * @return Label
	 */
	public Label getTrueLabel() {
		return trueLabelSet.get(0);
	}
	
	public void addNoisyLabel(Label label) {
		multiNoisyLabelSets.get(0).addNoisyLabel(label);
	}
	
	/**
	 * get all workers' Ids associated with this example
	 * @return workers' Ids
	 */
	public ArrayList<String> getWorkerIdList() {
		ArrayList<String> workerIdList = new ArrayList<String>();
		int noisyLabelSize = multiNoisyLabelSets.get(0).getLabelSetSize();
		for (int i  = 0; i < noisyLabelSize; i++) {
			Label label =  multiNoisyLabelSets.get(0).getLabel(i);
			String wId = label.getWorkerId();
			if (Misc.getElementEquals(workerIdList, wId) == null)
				workerIdList.add(wId);
		}
		return workerIdList;
	}
	
	/**
	 * get noisy label assigned by is specific worker
	 * @param wId worker id
	 * @return Label
	 */
	public Label getNoisyLabelByWorkerId(String wId)
	{
		int noisyLabelSize = multiNoisyLabelSets.get(0).getLabelSetSize();
		for (int i  = 0; i < noisyLabelSize; i++) {
			Label label =  multiNoisyLabelSets.get(0).getLabel(i);
			if (label.getWorkerId().equals(wId))
				return label;
		}
		return null;
	}
	
	/**
	 * return the number of Multiple Noisy Label Sets 
	 * @return number of Multiple Noisy Label Sets
	 */
	public int getMultipleNoisyLabelSetSize() {
		return multiNoisyLabelSets.size();
	}
	
	
	/**
	 * return the index-th Multiple Noisy Label Set
	 * @param index
	 * @return
	 */
	public MultiNoisyLabelSet getMultipleNoisyLabelSet(int index) {
		return multiNoisyLabelSets.get(index);
	}
	
	/**
	 * resets the first multi-noisy label set
	 */
	public void resetMultiNoisyLabelSet() {
        multiNoisyLabelSets.set(0, new MultiNoisyLabelSet(null));
    }
	
	/**
	 * set an integrated label to the first multiple noisy label set
	 * @param label
	 */
	public void setIntegratedLabel(Label label) {
		multiNoisyLabelSets.get(0).setIntegratedLabel(label);
	}
	
	/**
	 * get an integrated label from the first multiple noisy label set
	 * @return
	 */
	public Label getIntegratedLabel() {
		return multiNoisyLabelSets.get(0).getIntegratedLabel();
	}
	
	/**
	 * make Weka Instance Class Value equals to the example's Integrated Label
	 */
	public void assignIntegeratedLabel2WekaInstanceClassValue() {
		Label label = multiNoisyLabelSets.get(0).getIntegratedLabel();
		if ((label != null) && (classIndex() >= 0)) {
			this.setClassValue(new Integer(label.getValue()).toString());
		}
	}
	
	public void assignTrueLabel2WekaInstanceClassValue() {
		Label label = trueLabelSet.get(0);
		if ((label != null) && (classIndex() >= 0)) {
			this.setClassValue(new Integer(label.getValue()).toString());
		}
	}
	
	public int getTrainingLabel() {
		return (int) this.value(classIndex());
	}
	
	public void setTrainingLabel(int value) {
		this.setClassValue(value);
	}
	
	public Category getCategory(int classValue) {
		ArrayList<Category> list = categorySets.get(0);
		for (Category c: list) {
			if (c.getValue() == classValue)
				return c;
		}
		return null;
	}
	
	public void addCategory(Category c) {
		if (Misc.getElementEquals(categorySets.get(0), c) == null)
			categorySets.get(0).add(c);
	}
	
	/**
	 * delete a noisy label by worker Id
	 * @param wId
	 */
	public void delNoisyLabelByWorkerId(String wId) {
		MultiNoisyLabelSet mnls = multiNoisyLabelSets.get(0);
		mnls.delNoisyLabelByWorkerId(wId);
	}
	
	/**
	 * this copy constructive function is very import. When add an example to a
	 * data set, this function will be called. 
	 * @param example
	 */
	private Example(Example example) {
		super(example);
		this.setDataset(example.dataset());
		this.id = new String(example.getId());
		// copy categorySets
		for (ArrayList<Category> categroySet : example.categorySets) {
			ArrayList<Category> cateSet = new ArrayList<Category>();
			this.categorySets.add(cateSet);
			for (Category cate: categroySet) {
				cateSet.add(cate.copy());
			}
		}
		// copy true labels
		for (Label trueLabel : example.trueLabelSet) {
			this.trueLabelSet.add(trueLabel.copy());
		}
		// copy all multiple noisy label sets
		for (MultiNoisyLabelSet multiNoisyLabelSet : example.multiNoisyLabelSets) {
			multiNoisyLabelSets.add(multiNoisyLabelSet.copy());
		}
	}
	
	/**
	 * this is only for deep copy
	 * @param numAttributes
	 */
	private Example(double weight, double[] attValues, int[] indices,
		    int maxNumValues) {
		super(weight, attValues, indices, maxNumValues);
	}
	
	private String id = null;
	
	/*
	 * List of category set used both multi-label and signal label
	 */
	private ArrayList<ArrayList<Category>> categorySets = new ArrayList<ArrayList<Category>>();
	
	/*
	 * True Label Set used both multi-label and single label
	 */
	private ArrayList<Label> trueLabelSet = new ArrayList<Label>();
	
	/*
	 * used both multi-label and single label
	 */
	private ArrayList<MultiNoisyLabelSet> multiNoisyLabelSets = new ArrayList<MultiNoisyLabelSet>();
}

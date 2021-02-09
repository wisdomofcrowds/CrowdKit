package ceka.core;

import java.util.ArrayList;

import ceka.utils.IdDecorated;
import ceka.utils.Misc;

/**
 * Worker class presents workers, users, annotators, or
 * others that provide content (labels) in crowdsourcing systems  
 * 
 */
public class Worker implements IdDecorated {
	
	/**
	 * used for the Label that is the ground truth
	 */
	public static final String WORKERID_GOLD = "Gold";
	/**
	 * used for the Label that is integrated
	 */
	public static final String WORKERID_INT  = "Integrated";
	
	/**
	 * constructor
	 * @param id the unique identifier of a worker
	 */
	public Worker(String id) {
		this.id = id;
		// create a MultipleNoisyLabelSet with default name
		multiNoisyLabelSets.add(new MultiNoisyLabelSet(null));
	}
	
	/**
	 * get ID of an object of class Worker
	 * @return ID of the object
	 */
	public String getId() {
		return id;
	}
	
	/**
	 * set id of a worker
	 * @param wID
	 */
	public void setId(String wID) {
		id = new String(wID);
	}
	
	/**
	 * get a copy of worker with the same Id
	 * @return
	 */
	public Worker copy() {
		Worker cpW = new Worker(id);
		cpW.setMultiNoisyLabelSet(multiNoisyLabelSets.get(0).copy());
		return cpW;
	}
	
	/**
	 * add a noisy label to this worker
	 * @param label
	 */
	public void addNoisyLabel(Label label) {
		MultiNoisyLabelSet mnls = Misc.getElementById(multiNoisyLabelSets, label.getName());
		mnls.addNoisyLabel(label);
	}
	
	/**
	 * get multiple noisy label set in index-th position.
	 * for single label scenario, index is 0
	 * @param index
	 * @return multiple noisy label set
	 */
	public MultiNoisyLabelSet getMultipleNoisyLabelSet(int index) {
		return multiNoisyLabelSets.get(index);
	}
	
	private void setMultiNoisyLabelSet(MultiNoisyLabelSet mnls) {
		multiNoisyLabelSets.clear();
		multiNoisyLabelSets.add(mnls);
	}
	
	private String id;
	/*
	 * Designed for multi-label and single label usages. The first is for single label
	 */
	private ArrayList<MultiNoisyLabelSet> multiNoisyLabelSets = new ArrayList<MultiNoisyLabelSet>();
}

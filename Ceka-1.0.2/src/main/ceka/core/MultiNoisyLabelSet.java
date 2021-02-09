/**
 * 
 */
package ceka.core;

import java.util.ArrayList;

import ceka.utils.IdDecorated;
import ceka.utils.Misc;

/**
 * The Multiple Noisy Label Set contains labels provided by workers
 *
 */
public class MultiNoisyLabelSet implements IdDecorated {

	/**
	 * create a MultiNoisyLabelSet with id. If id is null or empty,
	 * then use the value {@link Label.DEFAULT_LABEL_NAME}
	 * @param id
	 */
	public MultiNoisyLabelSet(String id) {
		if ((id == null) || (id.isEmpty()))
			this.id = new String(Label.DEFAULT_LABEL_NAME);
		else
			this.id = new String(id);
	}
	
	/** (non-Javadoc)
	 * @see ceka.utils.IdDecorated#getId()
	 */
	@Override
	public String getId() {
		return id;
	}
	
	/**
	 * Add noisy label if label exists then change its value
	 * @param label
	 */
	public void addNoisyLabel(Label label) {
		Label exist = Misc.getElementEquals(labels, label);
		if (exist != null) {
			exist.setValue(label.getValue());
		} else {
			labels.add(label);
		}
	}
	
	/**
	 * @return size of this label set
	 */
	public int getLabelSetSize() {
		return labels.size();
	}
	
	/**
	 * get label by index
	 * @param index
	 * @return Label
	 */
	public Label getLabel(int index) {
		return labels.get(index);
	}
	
	/**
	 * copy current MultiNoisyLabelSet
	 * @return new MultiNoisyLabelSet
	 */
	public MultiNoisyLabelSet copy() {
		MultiNoisyLabelSet mnls = new MultiNoisyLabelSet(this.id);
		for (Label label : this.labels) {
			mnls.labels.add(label.copy());
		}
		if (integratedLabel != null)
			mnls.integratedLabel = this.integratedLabel.copy();
		return mnls;
	}
	
	/**
	 * set integrated label of the multiple noisy label set
	 * @param label
	 */
	public void setIntegratedLabel(Label label) {
		integratedLabel = label;
	}

	/**
	 * get integrated label of the multiple noisy label set
	 * @return
	 */
	public Label getIntegratedLabel() {
		return integratedLabel;
	}
	
	/**
	 * get the noisy label given by worker Id
	 * @param wId
	 * @return
	 */
	public Label getNoisyLabelByWorkerId(String wId) {
		int numLabel = labels.size();
		for (int i = 0; i < numLabel; i++) {
			Label label = labels.get(i);
			if (label.getWorkerId().equals(wId))
				return label;
		}
		return null;
	}
	
	/**
	 * get the noisy label given by example Id
	 * @param wId
	 * @return
	 */
	public Label getNoisyLabelByExampleId(String eId) {
		int numLabel = labels.size();
		for (int i = 0; i < numLabel; i++) {
			Label label = labels.get(i);
			if (label.getExampleId().equals(eId))
				return label;
		}
		return null;
	}

	/**
	 * delete noisy label given by a worker
	 * @param wId
	 */
	public void delNoisyLabelByWorkerId(String wId) {
		ArrayList<Label> toRemove = new ArrayList<Label>();
		int numLabel = labels.size();
		for (int i = 0; i < numLabel; i++) {
			Label label = labels.get(i);
			if (label.getWorkerId().equals(wId)) {
				toRemove.add(label);
			}
		}
		for (int i = 0; i < toRemove.size(); i++) {
			Label label = toRemove.get(i);
			labels.remove(label);
		}
	}
	
	private String id;
	private ArrayList<Label> labels = new ArrayList<Label>();
	private Label integratedLabel = null;
}

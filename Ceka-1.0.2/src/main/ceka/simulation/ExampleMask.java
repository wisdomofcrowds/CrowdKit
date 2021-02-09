/**
 * 
 */
package ceka.simulation;

import java.util.ArrayList;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.simulation.ExampleWorkersMask.Mask;
import ceka.utils.IdDecorated;
import ceka.utils.Misc;

/**
 * This class is used for selecting the examples to be saved to file
 * @author Zhang
 *
 */
public class ExampleMask {
	
	class Mask implements IdDecorated{
		int status = 1;
		String exampleId = null;
		
		public Mask(String id) {
			exampleId = id;
		}
		/* (non-Javadoc)
		 * @see ceka.utils.IdDecorated#getId()
		 */
		@Override
		public String getId() {
			return exampleId;
		}
	}

	public ExampleMask() {
		
	}
	
	public void intialize(Dataset data) {
		int numExample = data.getExampleSize();
		for (int i = 0; i  < numExample; i++) {
			Example e = data.getExampleByIndex(i);
			Mask mask = new Mask(e.getId());
			masks.add(mask);
		}
	}
	
	public void disableExample (String exampleId) {
		Mask mask = Misc.getElementById(masks, exampleId);
		if (mask != null)
			mask.status = 0;
	}
	
	public void enableExample (String exampleId) {
		Mask mask = Misc.getElementById(masks, exampleId);
		if (mask != null)
			mask.status = 1;
	}
	
	public boolean isActiveExample(String exampleId) {
		boolean ret = true;
		Mask mask = Misc.getElementById(masks, exampleId);
		if ((mask != null) && (mask.status == 0))
			ret = false;
		return ret;
	}
	
	private ArrayList<Mask> masks = new ArrayList<Mask>();
}

/**
 * 
 */
package ceka.simulation;

import java.util.ArrayList;
import java.util.List;

import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.Worker;
import ceka.utils.Misc;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author Jing
 *
 */
public class SimExampleFactory {
	/**
	 * create an empty dataset, a number of categories must be specified
	 * @param name name of the dataset
	 * @param numCate  number of categories, must >=2
	 * @return empty dataset
	 * @throws IllegalArgumentException
	 */
	public static Dataset createEmptyDataset(String name, int numCate) throws IllegalArgumentException {
		if (numCate < 2)
			throw new IllegalArgumentException();
		ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
		// we need at least one attribute "class"
		List<String> nominalValues = new ArrayList<String>(numCate);
		for (int i = 0; i < numCate; i++) {
			nominalValues.add(new Integer(i).toString());
		}
		Attribute classAttrib = new Attribute("class", nominalValues);
		attInfo.add(classAttrib);
		Instances wekaInstances = new Instances(name, attInfo, 0);
		wekaInstances.setClassIndex(0);
		Dataset dataset = new Dataset(wekaInstances);
		for (int i = 0; i < numCate; i++) {
			dataset.addCategory(new Category(null, new Integer(i).toString()));
		}
		return dataset;
	}
	
	/**
	 * A convenient method for adding an example into a dataset
	 * @param dataset
	 * @param exampleId
	 * @param trueClass
	 */
	public static void addExampleToDataset(Dataset dataset, String exampleId, String trueClass) {
		// check the category
		Integer cateInteger = Integer.parseInt(trueClass);
		List<Integer> cateList = new ArrayList<Integer>();
		for (int i = 0; i < dataset.getCategorySize(); i++)
			cateList.add(dataset.getCategory(i).getValue());
		if (Misc.getElementEquals(cateList, cateInteger) == null)
			return;
		// create an example
		Instance wekaInstance = new DenseInstance(1);
		wekaInstance.setValue(0, cateInteger.intValue());
		Example example = new Example(wekaInstance, exampleId);
		Label trueLabel = new Label(null, cateInteger.toString(), exampleId, Worker.WORKERID_GOLD);
		example.setTrueLabel(trueLabel);
		dataset.addExample(example);
	}
	
}

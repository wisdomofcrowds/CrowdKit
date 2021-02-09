package ceka.core;

/**
 * Category of a label
 *
 */
public class Category {

	/**
	 * create a category
	 * @param name the name of category, if the name is empty then
	 * use the value {@link Label.DEFAULT_LABEL_NAME}
	 * @param value the integer code of this category
	 */
	public Category(String name, String value){
		if (name == null || name.isEmpty())
			this.labelName = Label.DEFAULT_LABEL_NAME;
		else
			this.labelName = new String(name);
		this.value = Integer.parseInt(value);
	}
	
	/**
	 * set the integer code of the Category
	 * @param v integer code
	 */
	public void setValue(int v) {
		value = v;
	}
	
	/**
	 * get the name of this Category
	 * @return name of the Category
	 */
	public String getLabelName(){
		return labelName;
	}
	
	/**
	 * get integer code of this Category
	 * @return the integer code of the category
	 */
	public int getValue(){
		return value;
	}
	
	/**
	 * set probability of the Category
	 * @param p probability in [0, 1]
	 */
	public void setProbability(double p) {
		prob = p;
	}
	
	/**
	 * get probability of the Category
	 * @return the probability in [0, 1]
	 */
	public double getProbability() {
		return prob;
	}
	
	public boolean equals(Object obj) {
		if ((obj == null) || (! (obj instanceof Category)))
			return false;
		if (obj != null)
		{
			if (this == obj)
				return true;
			if (((Category)obj).labelName.equals(this.labelName) 
				&& ((Category)obj).value == (this.value))
				return true;
		}
		return false;
	}
	
	/**
	 * copy a same category 
	 * @return the newly created same Category
	 */
	public Category copy()
	{
		Category newCate = new Category(this.labelName, new Integer(this.value).toString());
		newCate.prob = this.prob;
		return newCate;
	}
	
	// for single label, label name is always Label.DEFAULT_LABEL_NAME;
	private String labelName = null;
	private int    value = -1;
	private double prob  = 0.0;
}

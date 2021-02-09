package ceka.core;

/**
 * Label. A Label is associated with a worker and an example.
 */
public class Label {

	public static final String DEFAULT_LABEL_NAME = "label0";
	/**
	 * create Label object
	 * @param name Label name. For single label the name always be "label0",
	 * and for multi-label the names could be label0, label1, ..., lableN. 
	 * @param value The value of the label must be an integer
	 * @param exampleId The example associates this label
	 * @param workerId The worker associates this label
	 */
	public Label(String name, String value, String exampleId, String workerId) {
		if (name == null || name.isEmpty())
			this.name = DEFAULT_LABEL_NAME;
		else
			this.name = new String(name);
		this.value = Integer.parseInt(value);
		this.exampleId = new String(exampleId);
		this.workerId = new String(workerId);
	}
	
	public void setValue(int v) {
		value = v;
	}
	
	public String getName(){
		return name;
	}
	
	public int getValue(){
		return value;
	}
	
	public void setExampleId(String eID) {
		exampleId = new String(eID);
	}
	
	public String getExampleId(){
		return exampleId;
	}
	
	public void setWorkerId(String wID) {
		workerId = new String(wID);
	}
	
	public String getWorkerId() {
		return workerId;
	}
	
	/**
	 * if two labels are same if and only if they associate with
	 * the same example and worker (also label names are same)
	 */
	public boolean equals(Object obj) {
		if ((obj == null) || (! (obj instanceof Label)))
			return false;
		if (obj != null)
		{
			if (this == obj)
				return true;
			if (((Label)obj).name.equals(this.name) 
				&& ((Label)obj).exampleId.equals(this.exampleId)
				&& ((Label)obj).workerId.equals(this.workerId))
				return true;
		}
		return false;
	}
	
	/**
	 * copy a same Label
	 * @return the newly created same Label
	 */
	public Label copy() {
		Label newLabel = new Label(this.name, new Integer(this.value).toString(),
				this.exampleId, this.workerId);
		return newLabel;
	}
	
	private String name = null;
	private int value = -1;
	private String exampleId = null;
	private String workerId = null;
}

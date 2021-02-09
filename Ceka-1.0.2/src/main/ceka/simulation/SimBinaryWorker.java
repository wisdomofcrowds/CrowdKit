package ceka.simulation;

public class SimBinaryWorker extends SimWorker {

	public SimBinaryWorker(String id)  {
		super(id);
	}

	/**
	 * get sensitivity (recall, true positive rate)
	 * @return
	 */
	public double getSensitivity() {
		return sensitivity;
	}
	
	/**
	 * get specificity (true negative rate)
	 * @return
	 */
	public double getSpecificity() {
		return specificity;
	}
	
	public void setQuality(double v) {
		q = v;
		sensitivity = specificity = q;
	}
	
	/**
	 * set true positive rate and true negative rate
	 * @param sens sensitivity (recall, true positive rate)
	 * @param spec specificity (true negative rate)
	 */
	public void setParameters(double sens, double spec) {
		sensitivity = sens;
		specificity = spec;
	}
	
	protected double sensitivity = 0.5;
	protected double specificity = 0.5;
}

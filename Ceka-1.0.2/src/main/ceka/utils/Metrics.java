/**
 * 
 */
package ceka.utils;

/**
 * A simple class to record metrics
 *
 */
public class Metrics {
	
	public double accuracy = 0.0;
	public double auc = 0.0;
	public double aucConvex = 0.0;
	public double f1Score = 0.0;
	
	public Metrics() {
		
	}
	
	public Metrics(Metrics mtc) {
		accuracy = mtc.accuracy;
		auc = mtc.auc;
		aucConvex = mtc.aucConvex;
		f1Score = mtc.f1Score;
	}
	
	/**
	 * add mtc values to this object
	 * @param mtc
	 * @return this object
	 */
	public Metrics addValues(Metrics mtc) {
		accuracy += mtc.accuracy;
		auc += mtc.auc;
		aucConvex += mtc.aucConvex;
		f1Score += mtc.f1Score;
		return this;
	}
	
	/**
	 * minus other Metrics obj
	 * @param mtc
	 * @return this object
	 */
	public Metrics minusValues(Metrics mtc) {
		accuracy -= mtc.accuracy;
		auc -= mtc.auc;
		aucConvex -= mtc.aucConvex;
		f1Score -= mtc.f1Score;
		return this;
	}
	
	/**
	 * Divided by a scaler value
	 * @param denominator
	 * @return this object
	 */
	public Metrics dividedBy(double denominator) {
		if (denominator == 0) {
			accuracy = auc = aucConvex = f1Score = 0;
		} else {
			accuracy /= denominator;
			auc /= denominator;
			aucConvex /= denominator;
			f1Score /= denominator;
		}
		return this;
	}
	
	/**
	 * powered by a factor
	 * @param factor
	 * @return this object
	 */
	public Metrics pow(double factor) {
		accuracy = Math.pow(accuracy, factor);
		auc = Math.pow(auc, factor);
		aucConvex = Math.pow(aucConvex, factor);
		f1Score =  Math.pow(f1Score, factor);
		return this;
	}
	
}

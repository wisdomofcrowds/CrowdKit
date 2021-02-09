package ceka.utils;

public class AscendingElement <T> implements Comparable <AscendingElement <T> > {
	
	public void setKey(double k) {
		key = k;
	}
	
	public void setData(T d) {
		data = d;
	}
	
	public T getData() {
		return data;
	}
	
	public double getKey () {
		return key;
	}
	
	public int compareTo(AscendingElement<T> elem) {
		if (Misc.isDoubleSame(key, elem.key, MIN_E))
			return 0;
		else if (key > elem.key)
			return 1;
		else
			return -1;
	}
	
	private T data;
	private double key = 0.0;
	
	private static final double MIN_E = 1.0E-6;
}

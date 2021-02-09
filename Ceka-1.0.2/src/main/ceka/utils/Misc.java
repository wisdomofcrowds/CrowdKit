package ceka.utils;

import java.io.File;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class Misc {
	
	/**
	 * whether two double values are same
	 * @param d1 value1
	 * @param d2 value 2
	 * @param e  the minimum difference if two values are not same
	 * @return true or false
	 */
	public static boolean isDoubleSame(double d1, double d2, double e) {
		double diff = d1 - d2;
		if (Math.abs(diff) <= e)
			return true;
		return false;
	}
	
	/**
	 * round
	 * @param d
	 * @param decimalPlace
	 * @return
	 */
	public static Double round(double d, int decimalPlace) {
		// see the Javadoc about why we use a String in the constructor
		// http://java.sun.com/j2se/1.5.0/docs/api/java/math/BigDecimal.html#BigDecimal(double)
		BigDecimal bd = new BigDecimal(Double.toString(d));
		bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP);
		return bd.doubleValue();
	}
	
	/**
	 * extract file name with/without suffix
	 * @param path
	 * @param suffix whether need suffix
	 * @return file name
	 */
	public static String exstractFileName(String path, boolean suffix) {
		int begin = path.lastIndexOf('/');
		if (begin == -1)
			begin = path.lastIndexOf('\\');
		if (begin == -1)
			begin = 0;
		else
			begin += 1;
		int lastdot = path.lastIndexOf('.');
		if ((lastdot == -1) || (suffix))
			lastdot = path.length();
		return path.substring(begin, lastdot);
	}
	
	/**
	 * extract extension name of a path
	 * @param path
	 * @return extension name
	 */
	public static String extractFileSuffix(String path) {
		String filename = exstractFileName(path, true);
		int lastdot = filename.lastIndexOf('.');
		return filename.substring(lastdot + 1, filename.length());
	}
	
	/**
	 * extract directory from a path
	 * @param path
	 * @return the directory
	 */
	public static String extractDir(String path) {
		int lastSlash = path.lastIndexOf('/');
		if (lastSlash == -1)
			lastSlash = path.lastIndexOf('\\');
		if (lastSlash == -1)
			return path;
		return path.substring(0, lastSlash + 1);
	}
	
	/**
	 * get an object from a List by provided id
	 * @param list
	 * @param key
	 * @return object or null if not found
	 */
	public static <E extends IdDecorated, K> E getElementById(List<E> list, K key) {
		E retValue = null;
		for (E e: list) {
			if (e.getId().equals(key)) {
				retValue = e;
				break;
			}
		}
		return retValue;
	}
	
	/**
	 *  remove an object in a List 
	 * @param list
	 * @param elem
	 */
	public static <E extends IdDecorated, K> void delElementById(List<E> list, K key) {
		E obj = getElementById(list, key);
		if (obj != null)
			list.remove(obj);
	}
	
	/**
	 * add an object to a List if not existed checking by provided id
	 * @param list
	 * @param elem
	 * @return 
	 */
	public static <E extends IdDecorated> void addElementById(List<E> list, E elem) {
		E obj = getElementById(list, elem.getId());
		if (obj == null)
			list.add(elem);
	}
	
	/**
	 * get an object from a List by equals function
	 * @param list
	 * @param elem
	 * @return object or null if not found
	 */
	public static <E extends Object> E getElementEquals(List<E> list, E elem) {
		E retValue = null;
		for (E e: list) {
			if (e.equals(elem)) {
				retValue = e;
				break;
			}
		}
		return retValue;
	}
	
	/**
	 * ad an object to a List if not existed checking by equals function
	 * @param list
	 * @param elem
	 * @return
	 */
	public static <E extends Object> void addElementIfNotExistedEquals(List<E> list, E elem) {
		E obj = getElementEquals(list, elem);
		if (obj == null)
			list.add(elem);
	}
	
	/**
	 * randomly split a list into two lists based on the element number of first list.
	 * @param originalList
	 * @param numFirstList the element number of first list
	 * @return
	 */
	public static <E extends Object> ArrayList<List<E>> splitRandom(List<E> originalList, int numFirstList) {
		ArrayList<List<E>> lists = new ArrayList<List<E>>();
		List<E> first = null;
		List<E> second = null;
		if (originalList instanceof LinkedList) {
			first = new LinkedList<E>();
			second = new LinkedList<E>();
		} else {
			first = new ArrayList<E>();
			second = new ArrayList<E>();
		}
		lists.add(first);
		lists.add(second);
		ArrayList<Integer> firstIndices = new ArrayList<Integer>();
		long seed = System.nanoTime() + (long)(Math.random() * MAGIC);
		Random rand = new Random(seed);
		while (firstIndices.size() < numFirstList) {
			Integer nextInt = new Integer(rand.nextInt(originalList.size()));
			boolean inFirst = false;
			for (Integer elem: firstIndices) {
				if (elem.intValue() == nextInt.intValue()) {
					inFirst = true;
					break;
				}
			}
			if (!inFirst)
				firstIndices.add(nextInt);
		}
		int index = 0;
		for (E e : originalList) {
			boolean inFirst = false;
			for (Integer elem: firstIndices) {
				if (elem.intValue() == index) {
					inFirst = true;
					break;
				}
			}
			if (inFirst)
				first.add(e);
			else
				second.add(e);
			index++;
		}
		return lists;
	}
	
	/**
	 * randomly split a list into two lists with the cut point.
	 * @param originalList
	 * @param cut the proportion of the first list
	 * @return
	 */
	public static <E extends Object> ArrayList<List<E>> splitRandom(List<E> originalList, double cut) {
		int numFirst = (int) (originalList.size() * cut);
		return splitRandom(originalList, numFirst);
	}
	
	/**
	 * create a directory
	 * @param dirPath
	 */
	public static void createDirectory(String dirPath) {
		File tempDatasetDir  = new File(dirPath);
		if (!tempDatasetDir.exists())
			tempDatasetDir.mkdirs();
	}
	
	/**
	 * randomly select {num} number of non-repeated integers between interval [min, max]  
	 * @param num number of selected integers
	 * @param min minimum
	 * @param max maximum
	 * @return a list of the selected integers
	 */
	public static ArrayList<Integer> randSelect(int num, int min, int max) {
		int diff = max - min;
		Random rand = new Random();
		ArrayList<Integer> list = new ArrayList<Integer> ();
		
		int count = 0;
		while (count < num) {
			Integer next = new Integer(rand.nextInt(diff + 1));
			if (getElementEquals(list, next) == null) {
				list.add(next);
				count++;
			}
		}
		
		for (int i = 0; i < list.size(); i++) {
			Integer rst = new Integer(list.get(i) + min);
			list.set(i, rst);
		}
		
		return list;
	}
	
	/**
	 * f(n) = n*f(n-1)
	 * @param n
	 * @return
	 */
	public static double fact(long n) {
		double x = 1;
		for (int i = 1; i <= n; i++)
			x *= i;
		return x;
	}
	
	/**
	 * f(n) = n*f(n-1) using big integer
	 * @param n
	 * @return
	 */
	public static BigInteger factBig(BigInteger n) {
		BigInteger x = new BigInteger("1");
		BigInteger i = new BigInteger("1");
		BigInteger one = new BigInteger("1");
		for (; !(i.compareTo(n) == 1); i=i.add(one))
			x = x.multiply(i);
		return x;
	}
	
	/**
	 * f(n) = n*f(n-1) using big integer
	 * @param n
	 * @return
	 */
	public static BigInteger factBig(long pn){
		Long l = new Long(pn);
		BigInteger n = new BigInteger(l.toString());
		return factBig(n);
	}
	
	/**
	 * tail of beta distribution
	 * @param a
	 * @param b
	 * @param v
	 * @return
	 */
	public static double betaI(int a, int b, double v){
		double r = 0.0;
		for (int j=a; j<=(a+b-1); j++) {
			double numerator = fact(a+b-1);
			double denominator = fact(j) * fact(a+b-1-j);
			r += ((double)numerator * Math.pow(v, j) * Math.pow(1-v, a+b-1-j) / (double)denominator);
		}
		return r;
	}
	
	/**
	 * tail of beta distribution, using BigInteger and BigDecimal
	 * @param a
	 * @param b
	 * @param v
	 * @return
	 */
	public static BigDecimal betaIBig(BigInteger a, BigInteger b, BigDecimal v) {
		int scale = 8;
		
		BigDecimal r = new BigDecimal("0.0");
		BigInteger j = new BigInteger(a.toString());
		BigInteger one =  new BigInteger("1");
		
		BigInteger up = a.add(b).subtract(one);
		
		for (; !(j.compareTo(up) == 1); j=j.add(one)) {
			BigInteger ab1 = a.add(b).subtract(one);
			BigInteger ab1j = ab1.subtract(j);
			BigDecimal numerator = new BigDecimal(factBig(ab1));
			BigDecimal denominator = new BigDecimal(factBig(j).multiply(factBig(ab1j)));
			BigDecimal  pvj = v.pow(j.intValue());
			BigDecimal  pvj2 = new BigDecimal(1.0 - v.doubleValue());
			pvj2 = pvj2.pow(ab1j.intValue());
			r = r.add(numerator.multiply(pvj).multiply(pvj2).divide(denominator, scale, BigDecimal.ROUND_HALF_EVEN));
		}
		return r;
	}
	
	/**
	 * tail of beta distribution, using BigInteger internally
	 * @param a
	 * @param b
	 * @param v
	 * @return
	 */
	public static double betaIBig(int a, int b, double v){
		Integer pa = new Integer(a);
		Integer pb = new Integer(b);
		Double pv = new Double(v);
		BigInteger A = new BigInteger(pa.toString());
		BigInteger B = new BigInteger(pb.toString());
		BigDecimal V = new BigDecimal(pv.toString());
		
		return betaIBig(A, B, V).doubleValue();
	}
	
	public static int findMaxPositionRand(double [] vals) {
		ArrayList<DescendingElement<Integer>> list = new ArrayList<DescendingElement<Integer>>();
		for (int i =0; i < vals.length; i++) {
			DescendingElement<Integer> e = new DescendingElement<Integer>();
			e.setData(i);
			e.setKey(vals[i]);
			list.add(e);
		}
		Collections.sort(list);
		ArrayList<DescendingElement <Integer>> maxCountList = new ArrayList<DescendingElement <Integer>>();
		maxCountList.add(list.get(0));
		for (int j = 1; j < list.size(); j++){
			if (isDoubleSame(list.get(j).getKey(), list.get(0).getKey(), 0.000000000000000000001))
				maxCountList.add(list.get(j));
			else
				break;
		}
		int s = 0;
		if (maxCountList.size() > 1) {
			double r = Math.random();
			double grain = 1.0 / (double)maxCountList.size();
			s =  (int)(r / grain);
		}
		return maxCountList.get(s).getData().intValue();
	}
	
	private static final int MAGIC = 0xCE66D;
}

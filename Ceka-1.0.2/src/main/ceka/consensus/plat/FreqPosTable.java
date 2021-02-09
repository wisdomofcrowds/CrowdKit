package ceka.consensus.plat;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.ListIterator;

import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.utils.Misc;

public class FreqPosTable {
	
	private final double minE=0.00001;
	
	public class FreqPos implements Comparable<FreqPos> {
		public int    category = 0; // 0 negative, 1 positive
		public double freq = 0.0;
		public double proportion = 0.0;
		public ArrayList<Integer> items;
		
		public FreqPos() {
			items = new ArrayList<Integer>();
		}
		
		public int compareTo(FreqPos fp) {
			if (Misc.isDoubleSame(freq, fp.freq, minE))
				return 0;
			else if (freq > fp.freq)
				return 1;
			else
				return -1;
		}
		
		public int itemSize() {
			return items.size();
		}
	}

	
	public FreqPosTable() {
		entries = new ArrayList<FreqPos>();
	}
	
	public int entriesSize() {
		return entries.size();
	}
	
	public int totalItemCount() {
		int count = 0;
		for (int i = 0; i < entries.size(); i++) {
			count += entries.get(i).itemSize();
		}
		return count;
	}
	
	public FreqPos getFreqPos(int i) {
		if (i >= entries.size())
			i = entries.size() - 1;
		return entries.get(i);
	}
	
	public void sort() {
		Collections.sort(entries);
	}
	
	public void buildTable(ceka.core.Dataset dataset) {
		int count = 0;
		int exampleSize = dataset.getExampleSize();
		for (int i = 0; i < exampleSize; i++) {
			ceka.core.Example example = dataset.getExampleByIndex(i);
			if (example.getMultipleNoisyLabelSet(0).getLabelSetSize() != 0) {
				double d = getPositiveProportion(example.getMultipleNoisyLabelSet(0));
				insertTable(d, i);
				count++;
			}
		}
		for (int i = 0; i < entries.size(); i++) 
			entries.get(i).proportion = (double)entries.get(i).items.size() / (double)count;
	}
	
	public void printTable(FileWriter file)  {
		try {
			String pstr = null;
			pstr = String.format("Positive Frequence distribution:\n");
			file.write(pstr);
			file.write("Freq:  ");
			ListIterator<FreqPos> fsIter = entries.listIterator();
			for (int i = 0; i < entries.size(); i++) {
				FreqPos fs = fsIter.next();
				pstr = String.format("%.3f  ", fs.freq);
				file.write(pstr);
			}
			file.write("\n");
			file.write("Count: ");
			fsIter = entries.listIterator();
			for (int i = 0; i < entries.size(); i++) {
				FreqPos fs = fsIter.next();
				pstr = String.format("%d  ", fs.items.size());
				file.write(pstr);
			}
			file.write("\n");
			file.write("Prop: ");
			fsIter = entries.listIterator();
			for (int i = 0; i < entries.size(); i++) {
				FreqPos fs = fsIter.next();
				pstr = String.format("%.3f  ", fs.proportion);
				file.write(pstr);
			}
			file.write("\n");
			file.write("Category: ");
			fsIter = entries.listIterator();
			for (int i = 0; i < entries.size(); i++) {
				FreqPos fs = fsIter.next();
				pstr = String.format("%d  ", fs.category);
				file.write(pstr);
			}
			file.write("\n");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private void insertTable(double freq, int index) {
		ListIterator<FreqPos> fsIter = entries.listIterator();
		for (int i = 0; i < entries.size(); i++) {
			FreqPos fs = fsIter.next();
			if (Misc.isDoubleSame(freq, fs.freq, minE)) {
				fs.items.add(index);
				return;
			}
		}
		
		// not found
		FreqPos nFP = new FreqPos();
		nFP.freq = freq;
		nFP.items.add(index);
		entries.add(nFP);
	}
	
	private double getPositiveProportion(MultiNoisyLabelSet mnls) {
		int labelSize = mnls.getLabelSetSize();
		int positiveCount = 0;
		for (int i = 0; i < labelSize; i++) {
			Label label = mnls.getLabel(i);
			if (label.getValue() == 1) {
				positiveCount++;
			}
		}
		return (double) positiveCount / (double) labelSize;
	}
	
	private ArrayList<FreqPos> entries;
}


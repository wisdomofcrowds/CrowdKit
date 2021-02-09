package ceka.noise.clustering;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

import weka.core.Attribute;
import weka.core.Instances;
import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;

class DatasetConverter 
{
    public static File removeClassAttribute(File original) throws Exception
    {
        File newFile = new File(original.getAbsolutePath() + ".minusClasses.arff");
        newFile.delete();
        Scanner scanner = new Scanner(original);
        BufferedWriter bw = new BufferedWriter(new FileWriter(newFile));
        boolean dataFlag = false;
        while(scanner.hasNextLine())
        {
           String line = scanner.nextLine();
           if(line.contains(" class ") || line.contains(" CLASS ") || line.contains(" Class "))
           {
               
           }
           else
           {
               if(dataFlag)
               {
                   String[] data = line.split(",");
                   for(int i = 0; i < data.length - 1; i++)
                   {
                       bw.write(data[i]);
                       if(i != data.length - 2)
                       {
                           bw.write(",");
                       }
                   }
                   bw.write("\n");
               }
               else if(line.contains("@DATA") || line.contains("@data") || line.contains("@Data"))
               {
                   dataFlag = true;
                   bw.write(line + "\n");
               }
               else
               {
                   bw.write(line + "\n");
               }
           }
        }
        
        bw.close();
        return newFile;
    }
    
    public static Dataset loadFileNoClasses(String arffPath) throws Exception {
		FileReader reader = new FileReader(arffPath);
		BufferedReader readerBuffer = new BufferedReader(reader);
		Instances instSet = new Instances(readerBuffer);
		// find class attribute
		ArrayList<Integer> categories = new ArrayList<Integer>();
		int numAttrib = instSet.numAttributes();
		for (int i = 0; i < numAttrib; i++) {
			Attribute attr  = instSet.attribute(i);
			String attribName = attr.name();
			if (attribName.equalsIgnoreCase("class")) {
				int numV = attr.numValues();
				for (int j = 0; j < numV; j++) {
					String vStr = attr.value(j);
					Integer vInt = Integer.parseInt(vStr);
					categories.add(vInt);
				}
				//instSet.setClassIndex(i);
				break;
			}
		}
		// after weka Instance Set has been created, we can create ourself's data set
		Dataset  dataset = new Dataset(instSet, 0);
		Collections.sort(categories);
		boolean correct = true;
		for (int k = 0; k < categories.size(); k++) {
			if (categories.get(k).intValue() != k) {
				correct = false;
				break;
			}
			Category category = new Category(null, categories.get(k).toString());
			dataset.addCategory(category);
		}
		if (!correct)
			throw new Exception("Invalid cateories, categories must be consecutive integers staring from 0");
		// Create Examples
		
		int numInst =  instSet.numInstances();
		for (int i = 0; i < numInst; i++) {
			Example example = new Example(instSet.instance(i), new Integer(i).toString());
			//Integer classValue = new Integer((int)instSet.instance(i).classValue());
			//Label trueLabel = new Label(null, classValue.toString(), example.getId(),  Worker.WORKERID_GOLD);
			///example.setTrueLabel(trueLabel);
			dataset.addExample(example);
		}
		
		reader.close();
		readerBuffer.close();
		return dataset;
	}
}

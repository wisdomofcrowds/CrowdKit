package ceka.noise;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.filters.unsupervised.attribute.Normalize;
import ceka.consensus.ds.DSWorker;
import ceka.consensus.ds.DawidSkene;
import ceka.converters.FileLoader;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.Worker;
import ceka.utils.Misc;
import ceka.utils.PerformanceStatistic;

/*
 * Dynamic Classification Filtering (DCF) (Bryce Nicholson, Victor S. Sheng, Jing Zhang, Zhiheng Wang, and Xuefeng Xian. 
 * "Improving Label Accuracy by Filtering Low-Quality Workers in Crowdsourcing." In Proceedings of the 2015 Mexican
 * International Conference on Artificial Intelligence (forthcoming).) is an algorithm for filtering spammers 
 * from a crowdsourced data set. DCF can use any number of characteristics of workers as attributes for learning.
 * It builds a model from a set of workers from other crowdsourced data sets and classifies the workers in the data
 * set to filter. It uses a specific proportion of labels to remove, then classifies known workers in the "training"
 * set as spammer or non-spammer accordingly. In theory, DCF can be trained to remove any proportion
 * of the lowest-quality workers. DCF is compared with filtering algorithms by Raykar and Yu (RY) and by Ipeirotis
 * et al. (IPW), and is demonstrated to outperform them overall.
 * 
 * Constructor takes a string, representing the path to the directory to store the created data sets
 * necessary for learning.
 */
public class DynamicClassificationFilter 
{
	private String fileGenerationPath;
	
	public DynamicClassificationFilter()
	{
		this("");
	}
	
	/*
	 * Constructor takes a string, representing the path to the directory to store the created data sets
	 */
	public DynamicClassificationFilter(String fileGenerationPath)
	{
		this.fileGenerationPath = fileGenerationPath;
	}
	
	/**
	 * The method that filters spammers.
	 * 
	 * @param dataset The data set from which to remove spammers
	 * @param name The name of the data set
	 * @param names The array list of all data sets' names to use (the data set that is filtered, and the others to use as training data)
	 * @param datasets The array list of all data sets to use (the data set that is filtered, and the others to use as training data)
	 * @param filterLevel (Approximately) the proportion of labels to remove
	 * @param tolerance The acceptable margin of error to accept with respect to the filtering level
	 * @param classifier The classifier to use for learning
	 * @param attributeSet The list of worker attributes to use for learning
	 * 
	 * Currently, the list of worker attributes that are supported are:
	 * distanceFromAverageEvenness
	 * logSimilarity
	 * EMAccuracy
	 * spammerScore
	 * workerCost
	 * proportion
	 * 
	 * You must include EMAccuracy in the list of attributes, because it is used in determining
	 * which workers to label as spammers for the training set. The rules are not case sensitive.
	 */
	public HashMap<String, Double> filterSpammers(Dataset dataset, String name, ArrayList<String> names, 
			ArrayList<Dataset> datasets, double filterLevel, double tolerance, Classifier classifier, 
			ArrayList<String> attributeSet) throws Exception
    {
        for(int i = 0; i < datasets.size(); i++)
        {
            new DawidSkene(30).doInference(dataset);
        }
        ArrayList<Dataset> dat = new ArrayList();
        ArrayList<AnalyzedWorker> spammers = new ArrayList();
        dat.add(dataset);
        String filename = createSpammersArff(dat, attributeSet);
        DawidSkene mv = new DawidSkene(30);
        //MainProc.clearAllIntegratedLabels(dataset);
        mv.doInference(dataset);
        //AdaptiveWeightedMajorityVote mv = new AdaptiveWeightedMajorityVote();
        //MajorityVote mv = new MajorityVote();
        Dataset workers = FileLoader.loadFile(filename);
        Dataset workersCopy = FileLoader.loadFile(filename);

        ArrayList<AnalyzedWorker> possibleSpammers = getWorkersForDataset(dataset);
        ArrayList<AnalyzedTask> tasks = getTasksForDataset(dataset);
        WorkerTaskGraph graph = new WorkerTaskGraph(dataset, possibleSpammers, tasks);
        
        //printCorrelationCode(dataset);
        //workersCopy.setClassIndex(-1);
        Normalize normalize = new Normalize();
        normalize.setInputFormat(workersCopy);
        normalize.useFilter(workersCopy, normalize);
        double[] categorySizes = new double[workersCopy.getExampleSize()];
        for(int i = 0; i < workersCopy.getExampleSize(); i++)
        {
            Example e = workersCopy.getExampleByIndex(i);
            //e.setValue(4, 0);
            categorySizes[i] = e.value(2);
        }
     
        boolean ready = false;
        double threshold = .5;
        double adder = .25;
        Dataset others = null;
        int[] classifications = new int[0];
        double spamEst = 0;
        double spamPropEst = 0;
        int totalLabels = graph.getTotalNumLabels();
        int iterations = 0;
        double desiredProp = filterLevel;
        int desiredIterations = -1;
        double[] iterationVals = new double[10];
        //flag for starting over and stopping at detected iteration
        boolean flag = false;
        while(!ready)
        {
            if(iterations == 10)
            {
                System.out.println("Starting Over");
                iterations = 0;
                flag = true;
                threshold = .5;
                adder = .25;
            }
            if(flag)
            {
                double min = Double.POSITIVE_INFINITY;
                int minIndex = 0;
                for(int i = 0; i < iterationVals.length; i++)
                {
                    if(Math.abs(desiredProp - iterationVals[i]) < min)
                    {
                        min = Math.abs(desiredProp - iterationVals[i]);
                        minIndex = i;
                    }
                }
                desiredIterations = minIndex;
            }
            others = integratedDataset(others, names, datasets, name, threshold, attributeSet);
            classifier.buildClassifier(others);
            spamEst = 0;
            spamPropEst = 0;
            classifications = new int[workersCopy.getExampleSize()];
            //workersCopy.setClassIndex(3);
            //workersCopy.setClassIndex(5);
            workersCopy.setClassIndex(workersCopy.numAttributes() - 1);
            for(int i = 0; i < workersCopy.getExampleSize(); i++)
            {
                Example e = workersCopy.getExampleByIndex(i);
                //e.setValue(2,categorySizes[i]);
                int classification = (int)classifier.classifyInstance(e);
                classifications[i] = classification;
                //classifications[i] = 5;
                if(classifications[i] == 1)
                {
                    //try
                    //{
                        spamPropEst += (double)possibleSpammers.get(i).getMultipleNoisyLabelSet(0).getLabelSetSize() /
                                (double)totalLabels;
                        spamEst++;
                   // }
                    ///catch(IndexOutOfBoundsException ex)
                    //{}
                }
            }
            spamEst /= (double)workersCopy.getExampleSize();
            iterationVals[iterations] = spamPropEst;
            //if(spamEst < .45 && iterations < 20)
            if(iterations == desiredIterations)
            {
                break;
            }
            if(spamPropEst < desiredProp - tolerance)
            {
                threshold += adder;
                adder /= 2.0;
            }
            else if(spamPropEst > desiredProp + tolerance)
            {
                threshold -= adder;
                adder /= 2.0;
            }
            else
                ready = true;
            iterations++;
        }
        for(int i = 0; i < classifications.length; i++)
        {
            //try
            ///{
            if(classifications[i] == 1)
                spammers.add(possibleSpammers.get(i));
            //}
            ///catch(Exception e)
            //{}
        }
        

        double formerAccuracy = 0;
        double formerAUC = 0;
        
        formerAccuracy = accuracy(dataset);
        formerAUC = auc(dataset);

        
        //System.out.println("Label Quality of " + name + ": " + graph.getLabelQuality());
        Dataset newDataset = graph.removeSpammers(spammers);
        mv = new DawidSkene(30);
        mv.doInference(newDataset);
        ArrayList<AnalyzedWorker> workers2 = getWorkersForDataset(newDataset);
        ArrayList<AnalyzedTask> tasks2 = getTasksForDataset(newDataset);
        WorkerTaskGraph graph2 = new WorkerTaskGraph(newDataset, workers2, tasks2);
        //System.out.println("Now accuracy of " + name + ": " + accuracy(newDataset));
        //System.out.println("Percent increase: " + (accuracy(newDataset) - formerAccuracy) + "\n");
        //System.out.println("Now auc of " + name + ": " + MainProc.auc(newDataset));
        //System.out.println("Percent increase: " + (MainProc.auc(newDataset) - formerAUC) + "\n");
        //System.out.println("Now Label Quality of " + name + ": " + graph2.getLabelQuality() + "\n");
        HashMap<String, Double> result = new HashMap();
        result.put("BeforeAccuracy", Double.parseDouble("" + formerAccuracy));
        result.put("AfterAccuracy", Double.parseDouble("" + accuracy(newDataset)));
        result.put("BeforeAUC", Double.parseDouble("" + formerAUC));
        result.put("AfterAUC", Double.parseDouble("" + auc(newDataset)));
        return result;
    }
	
	private static ArrayList<AnalyzedWorker> getWorkersForDataset(Dataset dataset)
	{
		ArrayList<AnalyzedWorker> workers = new ArrayList();
		int numWorkers = dataset.getWorkerSize();
		int numCategories = dataset.getCategorySize();
		for(int i = 0; i < numWorkers; i++)
		{
			workers.add(new AnalyzedWorker(dataset.getWorkerByIndex(i), numCategories, true));
		}
		return workers;
	}
	
	private static ArrayList<AnalyzedTask> getTasksForDataset(Dataset dataset)
	{
		ArrayList<AnalyzedTask> tasks = new ArrayList();
		int numCategories = dataset.getCategorySize();
		for(int i = 0; i < dataset.getExampleSize(); i++)
		{
                    AnalyzedTask t = new AnalyzedTask(dataset.getExampleByIndex(i), numCategories);
                    t.setIntegratedLabel(dataset.getExampleByIndex(i).getIntegratedLabel());
                    tasks.add(t);
		}
		return tasks;
	}
	
	private Dataset integratedDataset(Dataset others, ArrayList<String> names, ArrayList<Dataset> datasets, String name, double threshold, ArrayList<String> attributeSet)
            throws Exception
    {
        ArrayList<Double> accs = new ArrayList();
        for(int i = 0; i < datasets.size(); i++)
        {
            if(name.equals(names.get(i)))
                continue;
            Dataset dataset = datasets.get(i);
            ArrayList<AnalyzedWorker> workers = getWorkersForDataset(dataset);
            ArrayList<AnalyzedTask> tasks = getTasksForDataset(dataset);
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
            for(int j = 0; j < workers.size(); j++)
            {
                accs.add(graph.getWorkerAccuracy(workers.get(j)));
            }
        }
        Collections.sort(accs);
        double threshVal = accs.get((int)((double)accs.size() * threshold));
        for(int i = 0; i < datasets.size(); i++)
        {
            if(name.equals(names.get(i)))
                continue;
            Dataset dataset = datasets.get(i);
            Dataset copy = makeCopy(dataset);
            new DawidSkene(30).doInference(copy);
            ArrayList<AnalyzedWorker> workers = getWorkersForDataset(copy);
            ArrayList<AnalyzedTask> tasks = getTasksForDataset(copy);
            WorkerTaskGraph graph = new WorkerTaskGraph(copy, workers, tasks);
            String filename = createIndividualSpamArff(graph, names.get(i), threshVal, attributeSet);
            Dataset d = FileLoader.loadFile(filename);
            if(i == 0 || (name.equals(names.get(0)) && i == 1))
            {
            	others = FileLoader.loadFile(filename);
            	others = others.generateEmpty();
            }
            Normalize n = new Normalize();
            n.setInputFormat(d);
            n.useFilter(d, n);
            others = combineDatasets(others, d);
        }
        others.setClassIndex(6);
        return others;
    }
	
    private String createIndividualSpamArff(WorkerTaskGraph graph, String name, double proportion, ArrayList<String> attSet) throws Exception
    {
        DecimalFormat df = new DecimalFormat("#.#####");
        String fileDirStr = fileGenerationPath + "\\Spammers\\Datasets";
        File fileDir = new File(fileDirStr);
        if(!fileDir.exists())
            fileDir.mkdirs();
        String filename = fileDir + "\\" + name + ".arff";
        File f = new File(filename);
        f.delete();
        BufferedWriter bw = new BufferedWriter(new FileWriter(f));
        bw.write("@RELATION " + name + "Spam\n\n");
        for(int i = 0; i < attSet.size(); i++)
        {
        	bw.write("@ATTRIBUTE\tatt" + (i + 1) + "\tNUMERIC\n");
        }
        bw.write("@ATTRIBUTE\tclass\t{0,1}\n\n");
        bw.write("@DATA\n\n");
        double[][] data = new double[graph.workers.size()][attSet.size()];
        int emAccIndex = -1;
        for(int i = 0; i < graph.workers.size(); i++)
        {
        	for(int j = 0; j < attSet.size(); j++)
        	{
        		data[i][j] = graph.getCharacteristicValueForWorker(attSet.get(j), graph.workers.get(i));
        		if(attSet.get(j).toLowerCase().equals("EMAccuracy".toLowerCase()))
        			emAccIndex = j;
        	}
        }
        if(emAccIndex == -1)
        	throw new Exception("It appears that you have not selected \"EMAccuracy\" as a characteristic. It is required.");
        double prop = 0;
        double threshold = 0;
        double[] nums = new double[data.length];
        for(int i = 0; i < data.length; i++)
        {
            nums[i] = data[i][emAccIndex];
        }
        if(proportion == -1)
        {
            prop = .80;
            threshold = .80;
        }
        else
        {
            //threshold = StatCalc.findPercentile(nums, proportion);
            threshold = proportion;
        }
        for(int i = 0; i < data.length; i++)
        {
            boolean record = false;
            for(int j = 0; j < data[i].length; j++)
            {
                if(data[i][j] == 0)
                    continue;
                else
                {
                    record = true;
                    break;
                }
            }
            if(record)
            {
                for(int j = 0; j < data[i].length; j++)
                {
                    if(j != emAccIndex)
                    {
                        bw.write(df.format(data[i][j]) + ",");
                    }
                    else
                        bw.write("" + 0 + ",");
                }
                
                
                if(data[i][emAccIndex] > threshold)
                    bw.write("0\n");
                else
                    bw.write("1\n");
            }
        }
        bw.close();
        return filename;
    }
	
	public static Dataset makeCopy(Dataset dataset)
	{
		Dataset result = dataset.generateEmpty();
		for(int i = 0; i < dataset.getExampleSize(); i++)
		{
			result.addExample(dataset.getExampleByIndex(i));
		}
		for(int i = 0; i < dataset.getWorkerSize(); i++)
		{
			result.addWorker(dataset.getWorkerByIndex(i));
		}
		for(int i = 0; i < dataset.getCategorySize(); i++)
		{
			result.addCategory(dataset.getCategory(i));
		}
		return result;
	}
	
	private static Dataset combineDatasets(Dataset dataset1, Dataset dataset2)
	{
		Dataset result = dataset1.generateEmpty();
		for(int i = 0; i < dataset1.getCategorySize(); i++)
		{
			result.addCategory(dataset1.getCategory(i));
		}
		for(int i = 0; i < dataset1.getExampleSize(); i++)
		{
			result.addExample(dataset1.getExampleByIndex(i));
		}
                int offset = result.getExampleSize();
		for(int i = 0; i < dataset2.getExampleSize(); i++)
		{
                    Example e = dataset2.getExampleByIndex(i);
                    e.setId("" + (Integer.parseInt(e.getId()) + offset));
                    result.addExample(e);
		}
		return result;
	}
	
	private static double accuracy(Dataset dataset)
    {
        int correct = 0;
        for(int i = 0; i < dataset.getExampleSize(); i++)
        {
            Example e = dataset.getExampleByIndex(i);
            if(e.getIntegratedLabel().getValue() == e.getTrueLabel().getValue())
                correct++;
        }
        return (double)correct / (double)dataset.getExampleSize();
    }
	
	private static double auc(Dataset dataset)
    {
        PerformanceStatistic ps = new PerformanceStatistic();
        ps.stat(dataset);
        return ps.getAUC();
    }
	
    private String createSpammersArff(ArrayList<Dataset> datasets, ArrayList<String> characteristics) throws Exception
    {
    	String filename = fileGenerationPath + "/spammers.arff";
        File f = new File(filename);
        f.delete();
        DecimalFormat df = new DecimalFormat("#.####");
        BufferedWriter bw = new BufferedWriter(new FileWriter(f));
        bw.write("@RELATION\tspammers\n");
        for(int i = 0; i < characteristics.size(); i++)
        {
        	bw.write("@ATTRIBUTE\tatt" + (i + 1) + "\treal\n");
        }
        bw.write("@ATTRIBUTE\tclass\t{0,1}\n");
        bw.write("\n@DATA\n");
        for(int i = 0; i < datasets.size(); i++)
        {
            Dataset dataset = datasets.get(i);
            new DawidSkene(30).doInference(dataset);
            ArrayList<AnalyzedWorker> workers = getWorkersForDataset(dataset);
            ArrayList<AnalyzedTask> tasks = getTasksForDataset(dataset);
            WorkerTaskGraph graph = new WorkerTaskGraph(dataset, workers, tasks);
            for(int j = 0; j < workers.size(); j++)
            {
            	AnalyzedWorker w = workers.get(j);
	            for(int k = 0; k < characteristics.size(); k++)
	            {
	                String c = characteristics.get(k);
	                bw.write("" + graph.getCharacteristicValueForWorker(c, w) + ",");
	            }
	            bw.write("0\n");
            }
        }
        bw.close();
        return filename;
    }
}

class AnalyzedWorker extends Worker
{
	protected double[][] tendencies;
	protected String id;
        public Worker worker;
        public double sim = 0;
	int numToLabel = 0;

	public AnalyzedWorker(Worker w, int numClasses, boolean includeLabels)
	{
		super("");
                worker = w;
		tendencies = new double[numClasses][numClasses];
		id = w.getId();
		if(includeLabels)
		{
			int numLabels = w.getMultipleNoisyLabelSet(0).getLabelSetSize();
			for(int i = 0; i < numLabels; i++)
			{
				addNoisyLabel(w.getMultipleNoisyLabelSet(0).getLabel(i));
			}
		}
	}
	
	public String getId()
	{
		return id;
	}
	
	public String toString()
	{
		String result = "Worker " + id + "\n";
		int dimension = tendencies.length;
		for(int i = 0; i < dimension; i++)
		{
			result += "|";
			for(int j = 0; j < dimension; j++)
			{
				result += tendencies[i][j];
				if(j != dimension - 1)
					result += "\t";
			}
			result += "|\n";
		}
		return result;
	}
	
	public void setNumToLabel(int num)
	{
		numToLabel = num;
	}
	
	public int getNumToLabel()
	{
		return numToLabel;
	}
        
    public Worker getWorker()
    {
        return worker;
    }
    
    public double getSim()
    {
        return sim;
    }
    
    public void setSim(double sim)
    {
        this.sim = sim;
    }
    
    public boolean equals(AnalyzedWorker other)
    {
        if(other.getId().equals(this.getId()))
            return true;
        else
            return false;
    }
}

class AnalyzedTask extends Example implements Comparable<AnalyzedTask>
{
	private String id;
	protected double[] tendencies;
	private int numLabels;
	public AnalyzedTask(Example e, int numClasses)
	{
		super((Instance)e);
		this.setTrueLabel(e.getTrueLabel());
		id = e.getId();
		tendencies = new double[numClasses];
		numLabels = 0;
	}
	
	public String getId()
	{
		return id;
	}
	
	public String toString()
	{
		String result = "Task " + id + "\n" + getTrueLabel().getValue() + " |";
		int dimension = tendencies.length;
		for(int i = 0; i < dimension; i++)
		{
			result += tendencies[i];
			if(i != dimension - 1)
				result += "\t";
		}
		result += "|\n";
		return result;
	}
	
	public void updateNumLabels()
	{
		numLabels = this.getMultipleNoisyLabelSet(0).getLabelSetSize();
		return;
	}
	
	public int compareTo(AnalyzedTask other)
	{
		if(this.numLabels > other.numLabels)
			return 1;
		else if(this.numLabels < other.numLabels)
			return -1;
		else
			return 0;
	}
}

class WorkerTaskGraph 
{
	protected ArrayList<AnalyzedWorker> workers = new ArrayList();
	protected ArrayList<AnalyzedTask> tasks = new ArrayList();
	protected ArrayList<String[]> edges = new ArrayList();
	protected Dataset dataset;
        protected double[][][] workerCMs = null;
	
	public WorkerTaskGraph(Dataset dataset, ArrayList<AnalyzedWorker> workers, 
			ArrayList<AnalyzedTask> tasks)
	{
		this.dataset = dataset;
		this.workers = workers;
		this.tasks = tasks;
		for(int i = 0; i < workers.size(); i++)
		{
			int numLabels = workers.get(i).getMultipleNoisyLabelSet(0).getLabelSetSize();
			for(int j = 0; j < numLabels; j++)
			this.addEdge(workers.get(i).getMultipleNoisyLabelSet(0).getLabel(j).getWorkerId(),
					workers.get(i).getMultipleNoisyLabelSet(0).getLabel(j).getExampleId());
		}
	}
	
	public void addEdge(String workerId, String taskId)
	{
		String[] edge = new String[2];
		edge[0] = workerId;
		edge[1] = taskId;
		edges.add(edge);
	}
	
	public ArrayList<AnalyzedTask> allTasksForWorker(AnalyzedWorker worker)
	{
		ArrayList<AnalyzedTask> tasks = new ArrayList();
		String workerId = worker.getId();
		for(String[] edge : edges)
		{
			if(edge[0].equals(workerId))
			{
				for(AnalyzedTask at : this.tasks)
				{
					if(at.getId().equals(edge[1]))
					{
						tasks.add(at);
					}
				}
			}
		}
		return tasks;
	}
	
	public ArrayList<AnalyzedWorker> allWorkersForTask(AnalyzedTask task)
	{
		ArrayList<AnalyzedWorker> workers = new ArrayList();
		String taskId = task.getId();
		for(String[] edge : edges)
		{
			if(edge[1].equals(taskId))
			{
				for(AnalyzedWorker w : this.workers)
				{
					if(w.getId().equals(edge[0]))
					{
						workers.add(w);
					}
				}
			}
		}
		return workers;
	}
	
	public int labelFor(AnalyzedWorker worker, AnalyzedTask task)
	{
		return LabelFor(worker, task).getValue();
	}
        
    public Label LabelFor(AnalyzedWorker worker, AnalyzedTask task)
	{
		int numLabels = worker.getMultipleNoisyLabelSet(0).getLabelSetSize();
		for(int i = 0; i < numLabels; i++)
		{
			Label l = worker.getMultipleNoisyLabelSet(0).getLabel(i);
			if(l.getExampleId().equals(task.getId()))
			{
				return l;
			}
		}
		return null;
	}
	
	
	public void inferWorkerAndTaskTendencies(int numCategories, int maxGuesses)
	{
		Random rand = new Random();

		for(int i = 0; i < 5; i++)
		{
			updateTaskTendencies(numCategories, maxGuesses);
			updateWorkerTendencies(numCategories, maxGuesses);
		}
		
	}
	
	public void updateWorkerTendencies(int numCategories, int maxGuesses)
	{
		//Update each worker's tendencies
		//Worker tendencies have the form of a square matrix.
		//Assume there are two classes, so...
		
		//      (worker labels it...)
		//              0 1
		//(class is) 0 |0 1|   this means that for class 0, worker labels it 1 
		//(class is) 1 |1 0|   100% of the time, and vice versa.
		
		//int randNum = rand.nextInt() % workers.size();
		//int randNum = 52;
		
		Random rand = new Random();
		for(int i = 0; i < workers.size(); i++)
		{
			//System.out.println("" + i + "/" + (workers.size() - 1));
			AnalyzedWorker thisWorker = workers.get(i);
			ArrayList<AnalyzedTask> theseTasks = allTasksForWorker(thisWorker);
			//if(randNum == i)
				//System.out.println(thisWorker + " has labeled " + theseTasks.size() + " tasks.");
			//For each class:
			for(int c = 0; c < numCategories; c++)
			{
				//if(randNum == i)
					//System.out.println("For category " + c + ":");
				//Define maxProb for now to be the probability generated
				//by the current tendencies
				double maxProb = 1;
				{
					for(int j = 0; j < theseTasks.size(); j++)
					{
						AnalyzedTask thisTask = theseTasks.get(j);
						if(thisTask.getTrueLabel().getValue() == c)
						{
							/*
							if(randNum == i)
							{
								System.out.println("The worker has done " + thisTask
										+ "and has labeled it " + graph.labelFor(thisWorker, thisTask) + "\n(" + thisWorker.tendencies[c][graph.labelFor(thisWorker, thisTask)] +
										" x " + thisTask.tendencies[graph.labelFor(thisWorker, thisTask)] + ") / (" +
										thisWorker.tendencies[c][graph.labelFor(thisWorker, thisTask)] +
										" x " + thisTask.tendencies[graph.labelFor(thisWorker, thisTask)] 
										+ " + " + (1 - thisWorker.tendencies[c][graph.labelFor(thisWorker, thisTask)])
										+ " x " + thisTask.tendencies[(graph.labelFor(thisWorker, thisTask) + 1) % 2]
										+ ")");
							}
							*/
							maxProb = maxProb * (thisWorker.tendencies[c][labelFor(thisWorker, thisTask)]
									* thisTask.tendencies[labelFor(thisWorker, thisTask)]);
							maxProb = maxProb / (thisWorker.tendencies[c][labelFor(thisWorker, thisTask)]
									* thisTask.tendencies[labelFor(thisWorker, thisTask)]
									+ (1 - thisWorker.tendencies[c][labelFor(thisWorker, thisTask)]) * 
									thisTask.tendencies[(labelFor(thisWorker, thisTask) + 1) % 2]);
							//if(randNum == i)
								//System.out.println("MaxProb is now " + maxProb);
						}
					}
				}						
				
				double maxGuess = thisWorker.tendencies[c][0];
				
				//Generate 1,000 guesses as to the maximally likely
				//set of tendencies and test them
				for(int g = 0; g < maxGuesses; g++)
				{
					double prob = 1;
					double guess = rand.nextDouble();
					double[] guesses = new double[2];
					guesses[0] = guess;
					guesses[1] = 1 - guess;

					for(int j = 0; j < theseTasks.size(); j++)
					{
						AnalyzedTask thisTask = theseTasks.get(j);
						if(thisTask.getTrueLabel().getValue() == c)
						{
							int label = labelFor(thisWorker, thisTask);
							prob = prob * (guesses[label] * 
									thisTask.tendencies[label]);
							prob = prob / (guesses[label] * 
									thisTask.tendencies[label]
									+ guesses[(label + 1) % 2] * 
									thisTask.tendencies[(label + 1) % 2]);
						}
					}
					
					if(prob > maxProb)
					{
						maxProb = prob;
						maxGuess = guess;
					}
				}
				
				//Finally update worker's tendencies
				thisWorker.tendencies[c][0] = maxGuess;
				thisWorker.tendencies[c][1] = 1 - maxGuess;
				
				//if(i == randNum)
					//System.out.println("Now the worker looks like this: " + thisWorker);
			}
			//System.out.println(thisWorker);
		}
	}
	
	public void updateTaskTendencies(int numCategories, int maxGuesses)
	{
		//Update each task's tendencies.
		//Task tendencies have the form of a vector or array.
		//Suppose there are 2 classes. Task tendencies will 
		//have the form:
		//|0 1|
		//Meaning a 0% tendency to be labeled as class 0 and a 100% 
		//tendency to be labeled as class 1.
		
		Random rand = new Random();
		for(int i = 0; i < tasks.size(); i++)
		{
			//System.out.println("" + i + "/" + (tasks.size() - 1));
			AnalyzedTask thisTask = tasks.get(i);
			int category = thisTask.getTrueLabel().getValue();
			ArrayList<AnalyzedWorker> theseWorkers = allWorkersForTask(thisTask);
			double prob = 1;
			double[] guesses = new double[2];
			guesses[0] = thisTask.tendencies[0];
			guesses[1] = 1 - guesses[0];
			for(int j = 0; j < theseWorkers.size(); j++)
			{
				AnalyzedWorker thisWorker = theseWorkers.get(j);
				int label = labelFor(thisWorker, thisTask);
				prob = prob * (thisWorker.tendencies[category][label] * guesses[label]);
				prob = prob / (thisWorker.tendencies[category][label] * guesses[label]
						+ thisWorker.tendencies[category][(label + 1) % 2] *
						guesses[(label + 1) % 2]);
			}
			double maxGuess = guesses[0];
			double maxProb = prob;
			for(int g = 0; g < maxGuesses; g++)
			{
				double guess = rand.nextDouble();
				prob = 1;
				guesses = new double[2];
				guesses[0] = guess;
				guesses[1] = 1 - guess;
				for(int j = 0; j < theseWorkers.size(); j++)
				{
					AnalyzedWorker thisWorker = theseWorkers.get(j);
					int label = labelFor(thisWorker, thisTask);
					prob = prob * (thisWorker.tendencies[category][label] * guesses[label]);
					prob = prob / (thisWorker.tendencies[category][label] * guesses[label]
							+ thisWorker.tendencies[category][(label + 1) % 2] *
							guesses[(label + 1) % 2]);
				}
				
				if(prob > maxProb)
				{
					maxProb = prob;
					maxGuess = guess;
				}
			}
			thisTask.tendencies[0] = maxGuess;
			thisTask.tendencies[1] = 1 - maxGuess;
			//System.out.println(thisTask);
		}
	}
	
	public void initializeTendencies()
	{
		//Initialize each task's tendencies
		for(int i = 0; i < tasks.size(); i++)
		{
			ArrayList<AnalyzedWorker> theseWorkers = allWorkersForTask(tasks.get(i));
			for(int j = 0; j < theseWorkers.size(); j++)
			{
				int label = labelFor(theseWorkers.get(j), tasks.get(i));
				tasks.get(i).tendencies[label]++;
			}
			double sum = 0;
			for(int j = 0; j < tasks.get(i).tendencies.length; j++)
			{
				sum += tasks.get(i).tendencies[j];
			}
			for(int j = 0; j < tasks.get(i).tendencies.length; j++)
			{
				tasks.get(i).tendencies[j] = (double)tasks.get(i).tendencies[j] / (double)sum;
			}
			//System.out.println(tasks.get(i).tendencies[0] + " " + tasks.get(i).tendencies[1]);
		}
		
		//Initialize each worker's tendencies
		for(int i = 0; i < workers.size(); i++)
		{
			AnalyzedWorker thisWorker = workers.get(i);
			ArrayList<AnalyzedTask> theseTasks = allTasksForWorker(thisWorker);
			for(int j = 0; j < theseTasks.size(); j++)
			{
				AnalyzedTask thisTask = theseTasks.get(j);
				int trueLabel = thisTask.getTrueLabel().getValue();
				int guessedLabel = labelFor(thisWorker, thisTask);
				thisWorker.tendencies[trueLabel][guessedLabel]++;
			}
			for(int j = 0; j < thisWorker.tendencies.length; j++)
			{
				int sum = 0;
				for(int k = 0; k < thisWorker.tendencies[j].length; k++)
				{
					sum += thisWorker.tendencies[j][k];
				}
				for(int k = 0; k < thisWorker.tendencies[j].length; k++)
				{
					thisWorker.tendencies[j][k] = (double)thisWorker.tendencies[j][k] / (double)sum;
				}
			}
		}	
	}
	
	public double getWorkerAccuracy(AnalyzedWorker worker)
	{
            ArrayList<AnalyzedTask> tasks = allTasksForWorker(worker);
            int counter = 0;
            int correct = 0;
            for(AnalyzedTask t : tasks)
            {
                if(labelFor(worker, t) == t.getTrueLabel().getValue())
                        correct++;
                counter++;
            }
            return (double)correct / (double)counter;
	}
        
        public double getWorkerEMAccuracy(AnalyzedWorker worker)
        {
            ArrayList<AnalyzedTask> tasks = allTasksForWorker(worker);
            int counter = 0;
            int correct = 0;
            for(AnalyzedTask t : tasks)
            {
                if(labelFor(worker, t) == t.getIntegratedLabel().getValue())
                        correct++;
                counter++;
            }
            return (double)correct / (double)counter;
        }
        
        public double getWorkerEMAUC(AnalyzedWorker worker)
        {
            Dataset aDataset = DynamicClassificationFilter.makeCopy(dataset);
            for(int i = 0; i < aDataset.getExampleSize(); i++)
            {
                aDataset.getExampleByIndex(i).setTrueLabel(dataset.getExampleByIndex(i).getIntegratedLabel());
            }
            ArrayList<AnalyzedWorker> notThisWorker = new ArrayList();
            for(int i = 0; i < workers.size(); i++)
            {
                boolean a = workers.get(i).equals(worker);
                if(!a)
                {
                    notThisWorker.add(workers.get(i));
                }
            }
            WorkerTaskGraph g = new WorkerTaskGraph(aDataset, workers, tasks);
            aDataset = g.removeSpammers(notThisWorker);
            ArrayList<Integer> indices = new ArrayList();
            for(int i = 0; i < aDataset.getExampleSize(); i++)
            {
                Example e = aDataset.getExampleByIndex(i);
                if(e.getMultipleNoisyLabelSet(0).getLabelSetSize() == 0)
                    indices.add(i);
            }
            for(int i = aDataset.getExampleSize() - 1; i >= 0; i--)
            {
                if(indices.size() > 0 && i == indices.get(indices.size() - 1))
                {
                    aDataset.simpleRemoveExampleByIndex(i);
                    indices.remove(indices.size() - 1);
                }
            }

            for(int i  = 0; i < aDataset.getExampleSize(); i++)
            {
                Example e = aDataset.getExampleByIndex(i);
                String lab = "" + e.getMultipleNoisyLabelSet(0).getLabel(0).getValue();
                e.setIntegratedLabel(new Label("", lab, "", ""));
            }
            int previousValue = 0;
            boolean allAreTheSame = true;
            for(int i = 0; i < aDataset.getExampleSize(); i++)
            {
                if(i == 0)
                    previousValue = aDataset.getExampleByIndex(i).getTrueLabel().getValue();
                else
                {
                    if(previousValue != aDataset.getExampleByIndex(i).getTrueLabel().getValue())
                    {
                        allAreTheSame = false;
                        break;
                    }
                    previousValue = aDataset.getExampleByIndex(i).getTrueLabel().getValue();
                }
            }
            PerformanceStatistic ps = new PerformanceStatistic();
            ps.stat(aDataset);
            if(allAreTheSame)
                return ps.getAccuracy();
            else
                return ps.getAUC();
            
        }
	
        public double getWorkerAUC(AnalyzedWorker worker)
        {
            Dataset aDataset = DynamicClassificationFilter.makeCopy(dataset);
            for(int i = 0; i < aDataset.getExampleSize(); i++)
            {
                aDataset.getExampleByIndex(i).setTrueLabel(dataset.getExampleByIndex(i).getTrueLabel());
            }
            ArrayList<AnalyzedWorker> notThisWorker = new ArrayList();
            for(int i = 0; i < workers.size(); i++)
            {
                boolean a = workers.get(i).equals(worker);
                if(!a)
                {
                    notThisWorker.add(workers.get(i));
                }
            }
            WorkerTaskGraph g = new WorkerTaskGraph(aDataset, workers, tasks);
            aDataset = g.removeSpammers(notThisWorker);
            ArrayList<Integer> indices = new ArrayList();
            for(int i = 0; i < aDataset.getExampleSize(); i++)
            {
                Example e = aDataset.getExampleByIndex(i);
                if(e.getMultipleNoisyLabelSet(0).getLabelSetSize() == 0)
                    indices.add(i);
            }
            for(int i = aDataset.getExampleSize() - 1; i >= 0; i--)
            {
                if(indices.size() > 0 && i == indices.get(indices.size() - 1))
                {
                    aDataset.simpleRemoveExampleByIndex(i);
                    indices.remove(indices.size() - 1);
                }
            }

            for(int i  = 0; i < aDataset.getExampleSize(); i++)
            {
                Example e = aDataset.getExampleByIndex(i);
                String lab = "" + e.getMultipleNoisyLabelSet(0).getLabel(0).getValue();
                e.setIntegratedLabel(new Label("", lab, "", ""));
            }
            int previousValue = 0;
            boolean allAreTheSame = true;
            for(int i = 0; i < aDataset.getExampleSize(); i++)
            {
                if(i == 0)
                    previousValue = aDataset.getExampleByIndex(i).getTrueLabel().getValue();
                else
                {
                    if(previousValue != aDataset.getExampleByIndex(i).getTrueLabel().getValue())
                    {
                        allAreTheSame = false;
                        break;
                    }
                    previousValue = aDataset.getExampleByIndex(i).getTrueLabel().getValue();
                }
            }
            PerformanceStatistic ps = new PerformanceStatistic();
            ps.stat(aDataset);
            if(allAreTheSame)
                return ps.getAccuracy();
            else
                return ps.getAUC();
            
        }
                
	public double getWorkerLabelEvenness(AnalyzedWorker worker)
	{
		int numClasses = dataset.getCategorySize();
		ArrayList<AnalyzedTask> tasks = allTasksForWorker(worker);
		double[] counters = new double[numClasses];
		for(AnalyzedTask t : tasks)
		{
			counters[labelFor(worker, t)]++;
		}
		counters = StatCalc.normalize(counters);
		double product = 1.0 / (double)StatCalc.choose(numClasses, 2);
		//System.out.println(StatCalc.choose(numClasses, 2));
		double sum = 0;
		int count = 0;
		for(int i = 0; i < counters.length; i++)
		{
			for(int j = i + 1; j < counters.length; j++)
			{
				double temp = sum;
				sum += (1.0 - Math.abs(counters[i] - counters[j])) * Math.min(counters[i], counters[j]);
				//System.out.println(++count + "\t" + (sum - temp));
			}
		}
		return product * sum * numClasses;
	}
	
	public static double getTrueLabelEvenness(ArrayList<AnalyzedTask> tasks, int numClasses)
	{
		double[] counters = new double[numClasses];
		for(AnalyzedTask t : tasks)
		{
			counters[t.getTrueLabel().getValue()]++;
		}
		counters = StatCalc.normalize(counters);
		double product = 1.0 / (double)StatCalc.choose(numClasses, 2);
		//System.out.println(StatCalc.choose(numClasses, 2));
		double sum = 0;
		int count = 0;
		for(int i = 0; i < counters.length; i++)
		{
			for(int j = i + 1; j < counters.length; j++)
			{
				double temp = sum;
				sum += (1.0 - Math.abs(counters[i] - counters[j])) * Math.min(counters[i], counters[j]);
				//System.out.println(++count + "\t" + (sum - temp));
			}
		}
		return product * sum * numClasses;
	}
        
        public double getWorkerRelativeEvenness(AnalyzedWorker worker)
        {
            int numClasses = dataset.getCategorySize();
            ArrayList<AnalyzedTask> tasks = allTasksForWorker(worker);
            double[] counters = new double[numClasses];
            double[] datasetCounters = new double[numClasses];
            for(AnalyzedTask t : tasks)
            {
                counters[labelFor(worker, t)]++;
            }
            
            for(int i = 0; i < dataset.getExampleSize(); i++)
            {
                datasetCounters[dataset.getExampleByIndex(i).getTrueLabel().getValue()]++;
            }
            
            counters = StatCalc.normalize(counters);
            datasetCounters = StatCalc.normalize(datasetCounters);
            double[] result = new double[numClasses];
            for(int i = 0; i < result.length; i++)
            {
                result[i] = counters[i] - datasetCounters[i];
            }
            double sum = 0;
            for(int i = 0; i < result.length; i++)
            {
                sum += Math.abs(result[i]);
            }
            sum /= 2.0;
            return 1 - sum;
        }
	
	public double getWorkerSimilarity(AnalyzedWorker worker)
	{
		ArrayList<AnalyzedTask> tasks = allTasksForWorker(worker);
		double total = 0;
		double same = 0;
		for(int i = 0; i < tasks.size(); i++)
		{
			AnalyzedTask task = tasks.get(i);
			int givenLabel = labelFor(worker, task);
                        ArrayList<AnalyzedWorker> workers = allWorkersForTask(task);
			int numLabels = workers.size();
			total += numLabels;
			for(int j = 0; j < numLabels; j++)
			{
				if(labelFor(workers.get(j), task) == givenLabel)
					same++;
			}
		}
		return same / total;
	}
        
        public double getWorkerLogSimilarity(AnalyzedWorker worker)
        {
            ArrayList<AnalyzedTask> tasks = allTasksForWorker(worker);
            double total = 0;
            double same = 0;
            double logSum = 0;
            for(int i = 0; i < tasks.size(); i++)
            {
                AnalyzedTask task = tasks.get(i);
                int givenLabel = labelFor(worker, task);
                ArrayList<AnalyzedWorker> workers = allWorkersForTask(task);
                for(int j = 0; j < workers.size(); j++)
                {
                    AnalyzedWorker w = workers.get(j);
                    if(labelFor(w, task) == givenLabel)
                    {
                        same++;
                    }
                    total++;
                }
                double sameProp = same / total;
                logSum += Math.log(sameProp);
            }
            return -1 * logSum / (double)tasks.size();
        }
        public double getWorkerDifference(AnalyzedWorker worker, AnalyzedTask task)
        {
            ArrayList<AnalyzedWorker> workers = allWorkersForTask(task);
            int same = 0;
            int total = 0;
            for(int i = 0; i < workers.size(); i++)
            {
                if(labelFor(workers.get(i),task) == labelFor(worker,task))
                    same++;
                total++;
            }
            double sim = (double)same / (double) total;
            return 1.0 - sim;
        }
        
        public Dataset removeSpammers(ArrayList<AnalyzedWorker> spammers)
        {
            Dataset result = dataset.generateEmpty();
            for(int i = 0; i < dataset.getCategorySize(); i++)
            {
                result.addCategory(dataset.getCategory(i));
            }
            for(int i = 0; i < dataset.getWorkerSize(); i++)
            {
                boolean add = true;
                for(int j = 0; j < spammers.size(); j++)
                {
                    if(dataset.getWorkerByIndex(i).equals(spammers.get(j).getWorker()))
                    {
                        add = false;
                        break;
                    }
                }
                if(add)
                    result.addWorker(dataset.getWorkerByIndex(i));
            }
            for(int i = 0; i < dataset.getExampleSize(); i++)
            {
                Example newExample = (Example)dataset.getExampleByIndex(i).copy();
                newExample.resetMultiNoisyLabelSet();
                result.addExample(newExample);
            }
            for(int i = 0; i < result.getWorkerSize(); i++)
            {
                for(int j = 0; j < result.getWorkerByIndex(i).getMultipleNoisyLabelSet(0).getLabelSetSize(); j++)
                {
                    Label l = result.getWorkerByIndex(i).getMultipleNoisyLabelSet(0).getLabel(j);
                    result.getExampleById(l.getExampleId()).addNoisyLabel(l);
                }
            }
            return result;
        }
        
        public void correctLabels(AnalyzedWorker w)
        {
            ArrayList<AnalyzedTask> tasks = allTasksForWorker(w);
            for(int i = 0; i < tasks.size(); i++)
            {
                AnalyzedTask task = tasks.get(i);
                ArrayList<AnalyzedWorker> otherWorkers = allWorkersForTask(task);
                double max = Double.NEGATIVE_INFINITY;
                AnalyzedWorker maxWorker = null;
                for(int j = 0; j < otherWorkers.size(); j++)
                {
                    double sim = otherWorkers.get(j).getSim();
                    if(sim > max)
                    {
                        max = sim;
                        maxWorker = otherWorkers.get(j);
                    }
                }
                LabelFor(w, task).setValue(labelFor(maxWorker, task));
            }
        }
        
        public void updateWorkerSims()
        {
            for(int i = 0; i < workers.size(); i++)
            {
                workers.get(i).setSim(getWorkerSimilarity(workers.get(i)));
            }
        }
        
        public int getTotalNumLabels()
        {
            int sum = 0;
            for(int i = 0; i < workers.size(); i++)
            {
                sum += workers.get(i).getMultipleNoisyLabelSet(0).getLabelSetSize();
            }
            return sum;
        }
        
        public double getLabelQuality()
        {
            double corr = 0;
            double total = 0;
            for(int i = 0; i < workers.size(); i++)
            {
                AnalyzedWorker w = workers.get(i);
                ArrayList<AnalyzedTask> tasks = allTasksForWorker(w);
                for(int j = 0; j < tasks.size(); j++)
                {
                    AnalyzedTask t = tasks.get(j);
                    if(labelFor(w,t) == t.getTrueLabel().getValue())
                    {
                        corr++;
                    }
                    total++;
                }
            }
            return corr / total;
        }
        
        public double averageEvenness()
        {
            double sum = 0;
            for(int i = 0; i < workers.size(); i++)
            {
                sum += this.getWorkerLabelEvenness(workers.get(i));
            }
            return sum / (double)workers.size();
        }
        
        public double[] maxAndMinLogSims()
        {
            double max = Double.NEGATIVE_INFINITY;
            double min = Double.POSITIVE_INFINITY;
            for(int i = 0; i < workers.size(); i++)
            {
                AnalyzedWorker w = workers.get(i);
                double sim = this.getWorkerLogSimilarity(w);
                if(sim > max)
                    max = sim;
                if(sim < min)
                    min = sim;
            }
            double[] result = {max, min};
            return result;
        }
        
        public double[][][] getWorkerCMs()
        {
            if(workerCMs == null)
            {
                DawidSkene ds = new DawidSkene(30);
                ds.doInference(dataset);
                int numCategories = dataset.getCategorySize();
                ArrayList<DSWorker> workers = ds.getWorkers();
                int numWorkers = workers.size();
                double[][][] result = new double[numWorkers][numCategories][numCategories];
                for(int i = 0; i < workers.size(); i++)
                {
                    result[i] = workers.get(i).getCM();
                }
                workerCMs = result;
                return result;
            }
            else
                return workerCMs;
        }
        
        //Creates a dataset identical to the dataset of this graph,
        //except that all workers chose the majority class as all their labels.
        //This is a function because of calculating the threshold for some
        //spam-elimination filters.
        public Dataset generateMajoritySpammerDataset()
        {
            Dataset newDataset = dataset.generateEmpty();
            double[] labelCounts = new double[dataset.getCategorySize()];
            for(int i = 0; i < dataset.getExampleSize(); i++)
            {
                labelCounts[dataset.getExampleByIndex(i).getTrueLabel().getValue()]++;
            }
            int majorityLabel = StatCalc.maxIndex(labelCounts);
            for(int i = 0; i < dataset.getCategorySize(); i++)
            {
                newDataset.addCategory(dataset.getCategory(i));
            }
            

            for(int i = 0; i < dataset.getExampleSize(); i++)
            {
                newDataset.addExample((Example)dataset.getExampleByIndex(i).copy());
                newDataset.getExampleByIndex(i).setTrueLabel(dataset.getExampleByIndex(i).getTrueLabel());
                int size = dataset.getExampleByIndex(i).getMultipleNoisyLabelSet(0).getLabelSetSize();
                for(int j = 0; j < size; j++)
                {
                    Label copy = dataset.getExampleByIndex(i).getMultipleNoisyLabelSet(0).getLabel(j).copy();
                    copy.setValue(majorityLabel);
                    newDataset.getExampleByIndex(i).addNoisyLabel(copy);
                }
            }
            for(int i = 0; i < workers.size(); i++)
            {
                Worker w = new Worker(workers.get(i).getId());
                for(int j = 0; j < workers.get(i).getMultipleNoisyLabelSet(0).getLabelSetSize(); j++)
                {
                    Label copy = workers.get(i).getMultipleNoisyLabelSet(0).getLabel(j).copy();
                    copy.setValue(majorityLabel);
                    w.addNoisyLabel(copy);
                }
                newDataset.addWorker(w);
            }
            return newDataset;
        }
        
        //Returns true if w has labeled any tasks of true class c, false otherwise
        public boolean anyLabelsForTrueClass(AnalyzedWorker w, int c)
        {
            ArrayList<AnalyzedTask> tasks = allTasksForWorker(w);
            for(int i = 0; i < tasks.size(); i++)
            {
                if(tasks.get(i).getTrueLabel().getValue() == c)
                    return true;
            }
            return false;
        }
        
        //Returns the prior probability that the worker w labels a task as c
        public double workerPrior(AnalyzedWorker w, int c)
        {
            ArrayList<AnalyzedTask> tasks = allTasksForWorker(w);
            int num = 0;
            int total = 0;
            for(int i = 0; i < tasks.size(); i++)
            {
                if(tasks.get(i).getTrueLabel().getValue() == c)
                    num++;
                total++;
            }
            return (double)num / (double)total;
        }
        
        public int getNumTasksForWorker(AnalyzedWorker w)
        {
            return allTasksForWorker(w).size();
        }
        
        public double[][] getWorkerData()
        {
            double[][] workerData2 = new double[workers.size()][7];
            double totalLabels = this.getTotalNumLabels();
            double averageEvenness = this.averageEvenness();
            for(int j = 0; j < workers.size(); j++)
            {
                //if(((double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize()
                //                                / (double)totalLabels < .001) || (double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize()
                 //                       < 9)
                 //                   continue;
                workerData2[j][0] = (double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize() /
                                (double)totalLabels;
                workerData2[j][1] = Math.abs(averageEvenness - this.getWorkerLabelEvenness(workers.get(j)));
                //workerData2[localWorkerNum][2] = dataset.getCategorySize();
                workerData2[j][3] = this.getWorkerLogSimilarity(workers.get(j));
                workerData2[j][4] = this.getWorkerEMAccuracy(workers.get(j));
                //workerData2[j][5] = this.getWorkerAccuracy(workers.get(j));
                workerData2[j][5] = this.spammerScore(workers.get(j), null)[0];
                workerData2[j][6] = this.workerCost(workers.get(j), null)[0];
            }
            return workerData2;
        }
        
        public double[][] getWorkerDataAUC()
        {
            double[][] workerData2 = new double[workers.size()][7];
            double totalLabels = this.getTotalNumLabels();
            double averageEvenness = this.averageEvenness();
            for(int j = 0; j < workers.size(); j++)
            {
                //if(((double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize()
                //                                / (double)totalLabels < .001) || (double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize()
                 //                       < 9)
                 //                   continue;
                workerData2[j][0] = (double)workers.get(j).getMultipleNoisyLabelSet(0).getLabelSetSize() /
                                (double)totalLabels;
                workerData2[j][1] = Math.abs(averageEvenness - this.getWorkerLabelEvenness(workers.get(j)));
                //workerData2[localWorkerNum][2] = dataset.getCategorySize();
                workerData2[j][3] = this.getWorkerLogSimilarity(workers.get(j));
                workerData2[j][4] = this.getWorkerEMAUC(workers.get(j));
                //workerData2[j][5] = this.getWorkerAccuracy(workers.get(j));
                workerData2[j][5] = this.spammerScore(workers.get(j), null)[0];
                workerData2[j][6] = this.workerCost(workers.get(j), null)[0];
            }
            return workerData2;
        }
        
        //The first element is the spammer score for the worker, the second element
        //is the expected value for spam score if the worker were to label all tasks as belonging to majority class
        public double[] spammerScore(AnalyzedWorker w, double[][][] confusionMatrices)
        {
            if(confusionMatrices == null)
            {
                confusionMatrices = this.getWorkerCMs();
            }
            double score = 0;
            double spamScore = 0;
            int i = this.getWorkerIndex(w);
            int numClasses = dataset.getCategorySize();
            double[] labelCounts = new double[dataset.getCategorySize()];
            for(int j = 0; j < dataset.getExampleSize(); j++)
            {
                labelCounts[dataset.getExampleByIndex(j).getTrueLabel().getValue()]++;
            }
            int majorityLabel = StatCalc.maxIndex(labelCounts);
            for(int j = 0; j < numClasses; j++)
            {
                for(int k = 0; k < j; k++)
                {
                    for(int l = 0; l < numClasses; l++)
                    {
                        double scoreAdd = 1.0 / (numClasses * (numClasses - 1)) * Math.pow(confusionMatrices[i][k][l] - confusionMatrices[i][j][l], 2.0);
                        //double spamScoreAdd = 1.0 / (numClasses * (numClasses - 1)) * Math.pow(confusionMatrices2[i][k][l] - confusionMatrices2[i][j][l], 2.0);
                        if(scoreAdd < Double.POSITIVE_INFINITY && scoreAdd > Double.NEGATIVE_INFINITY)
                            score += scoreAdd;
                        //if(spamScoreAdd < Double.POSITIVE_INFINITY && scoreAdd > Double.NEGATIVE_INFINITY)
                            //spamScore += spamScoreAdd;
                        if(l == majorityLabel)
                        {
                            boolean a = this.anyLabelsForTrueClass(w,k);
                            boolean b = this.anyLabelsForTrueClass(w,j);
                            if((a && !b) || (!a && b))
                                spamScore += 1.0 / (numClasses * (numClasses - 1));
                        }
                    }
                }
            }
            double[] result = new double[2];
            result[0] = score;
            result[1] = spamScore;
            return result;
        }
        
        public int getWorkerIndex(AnalyzedWorker w)
        {
            for(int i = 0; i < workers.size(); i++)
            {
                if(w.equals(workers.get(i)))
                    return i;
            }
            return -1;
        }
        
        public double[] workerCost(AnalyzedWorker w, double[][][] confusionMatrices)
        {
            if(confusionMatrices == null)
            {
                confusionMatrices = this.getWorkerCMs();
            }
            ArrayList<ArrayList<ArrayList<Double>>> softLabels = new ArrayList();
            double[] labelCounts = new double[dataset.getCategorySize()];
            for(int i = 0; i < dataset.getExampleSize(); i++)
            {
                labelCounts[dataset.getExampleByIndex(i).getTrueLabel().getValue()]++;
            }
            labelCounts = StatCalc.normalize(labelCounts);
            for(int i = 0; i < workers.size(); i++)
            {
                softLabels.add(new ArrayList());
                ArrayList<AnalyzedTask> t = this.allTasksForWorker(workers.get(i));
                for(int j = 0; j < t.size(); j++)
                {
                    softLabels.get(i).add(new ArrayList());
                }
            }
            ArrayList<AnalyzedTask> ts = this.allTasksForWorker(w);
            double workerCost = 0;
            double spammerCost = 0;
            int numClasses = dataset.getCategorySize();
            for(int j = 0; j < ts.size(); j++)
            {
                AnalyzedTask t = ts.get(j);
                //generate soft label (real and spammer)
                int label = this.labelFor(w, t);
                double[] softLabel = new double[numClasses];
                double[] spamSoftLabel = new double[numClasses];
                for(int k = 0; k < softLabel.length; k++)
                {
                    softLabel[k] = confusionMatrices[getWorkerIndex(w)][k][label] * labelCounts[k];
                    spamSoftLabel[k] = labelCounts[k];
                    softLabels.get(getWorkerIndex(w)).get(j).add(softLabel[k]);
                }
                softLabel = StatCalc.normalize(softLabel);
                spamSoftLabel = StatCalc.normalize(spamSoftLabel);
                //calculate cost of each label
                double cost = 0;
                double spamCost = 0;
                for(int k = 0; k < numClasses; k++)
                {
                    for(int l = 0; l < numClasses; l++)
                    {
                        if(k != l)
                        {
                            double val1 = softLabel[k] * softLabel[l];
                            if(val1 > Double.NEGATIVE_INFINITY && val1 < Double.POSITIVE_INFINITY)
                                cost += val1;
                            double val2 = spamSoftLabel[k] * spamSoftLabel[l];
                            if(val2 > Double.NEGATIVE_INFINITY && val2 < Double.POSITIVE_INFINITY)
                                spamCost += val2;
                        }
                    }
                }
                double val1 = cost * this.workerPrior(w, label);
                if(val1 > Double.NEGATIVE_INFINITY && val1 < Double.POSITIVE_INFINITY)
                      workerCost += val1;  
                double val2 = spamCost;
                if(val2 > Double.NEGATIVE_INFINITY && val2 < Double.POSITIVE_INFINITY)
                    spammerCost += val2;
            }
                double[] result = new double[2];
                result[0] = workerCost;
                result[1] = spammerCost;
                return result;
        }
        
         public Double getCharacteristicValueForWorker(String characteristic, AnalyzedWorker w)
        	throws Exception
        {
        	if(characteristic.toLowerCase().equals("distanceFromAverageEvenness".toLowerCase()))
        		return Math.abs(this.getWorkerLabelEvenness(w) - this.averageEvenness());
        	else if(characteristic.toLowerCase().equals("logSimilarity".toLowerCase()))
        		return this.getWorkerLogSimilarity(w);
        	else if(characteristic.toLowerCase().equals("EMAccuracy".toLowerCase()))
        		return this.getWorkerEMAccuracy(w);
        	else if(characteristic.toLowerCase().equals("EMAccuracy".toLowerCase()))
        		return this.getWorkerAccuracy(w);
        	else if(characteristic.toLowerCase().equals("spammerScore".toLowerCase()))
        		return this.spammerScore(w, null)[0];
        	else if(characteristic.toLowerCase().equals("workerCost".toLowerCase()))
        		return this.workerCost(w,  null)[0];
        	else if(characteristic.toLowerCase().equals("proportion".toLowerCase()))
        		return (double)(w.getMultipleNoisyLabelSet(0).getLabelSetSize()) /
                        (double)this.getTotalNumLabels();
        	else
        		throw new Exception("The characteristic \"" + characteristic +  
        				"\" is not recognized.");
        }
}

class StatCalc 
{
	public static double mean(double[] values)
    {
        double sum = 0;
        int numUndefined = 0;
        for(int i = 0; i < values.length; i++)
        {
        	if(values[i] > Double.NEGATIVE_INFINITY && values[i] < Double.POSITIVE_INFINITY)
        		sum += values[i];
        	else
        		numUndefined++;
        }
        sum /= (double)(values.length - numUndefined);
        return sum;
    }
	
	public static double[] mean(double[][] values)
	{
		double[] result = new double[values[0].length];
		for(int i = 0; i < values[0].length; i++)
		{
			double[] array = new double[values.length];
			for(int j = 0; j < values.length; j++)
			{
				array[j] = values[j][i];
			}
			result[i] = mean(array);
		}
		return result;
	}
    
    public static double variance(double[] values)
    {
        double mean = mean(values);
        double[] squareDifferences = new double[values.length];
        for(int i = 0; i < values.length; i++)
        {
            squareDifferences[i] = Math.pow(values[i] - mean, 2.0);
        }
        return mean(squareDifferences);
    }  
    
    public static double[] variance(double[][] values)
    {
    	double[] result = new double[values[0].length];
		for(int i = 0; i < values[0].length; i++)
		{
			double[] array = new double[values.length];
			for(int j = 0; j < values.length; j++)
			{
				array[j] = values[j][i];
			}
			result[i] = variance(array);
		}
		return result;
    }
    
    
    //Plots the density of the data points given
    public static String generateRCode(double[] data, String title)
    {
    	String result = "png('C:\\\\Users\\\\Bryce\\\\Desktop\\\\GMMPlots\\\\" +
    			title + ".png')\n";
    	result += "z <- c(";
    	for(int i = 0; i < data.length; i++)
    	{
    		result += "" + data[i];
    		if(i != data.length - 1)
    			result += ",";
    	}
    	result += ")\n";
    	result += "plot(density(z), lwd=3, col='red')\n";
		result += "dev.off()\n";
		return result;
    }
    
    public static double[][] calculateCovarianceMatrix(double[][] values)
    {
    	int dim = values[0].length;
    	double[][] result = new double[dim][dim];
    	double[] means = new double[dim];
    	for(int i = 0; i < dim; i++)
    	{
    		double[] array = new double[values.length];
    		for(int j = 0; j < values.length; j++)
    		{
    			array[j] = values[j][i];
    		}
    		means[i] = mean(array);
    		//System.out.println("mean " + i + ": " + means[i]);
    	}
    	for(int i = 0; i < dim; i++)
    	{
    		for(int j = 0; j < dim; j++)
    		{
    			double sum = 0;
    			for(int k = 0; k < values.length; k++)
    			{
    				//System.out.println("i = " + i + "\nj = " + j + "\nk = "
    						//+ k + "\nvalue k,i: " + values[k][i] + "\nvalue k,j: " + values[k][j]);
    				sum += (values[k][i] - means[i]) * (values[k][j] - means[j]);
    			}
    			sum /= (double)values.length;
    			result[i][j] = sum;
    		}
    	}
    	return result;

    }
    
    public static double[] scaleArray(double scale, double[] values)
    {
    	double[] result = new double[values.length];
    	for(int i = 0; i < values.length; i++)
    	{
    		result[i] = values[i] * scale;
    	}
    	return result;
    }
    
    public static double[] subtractArrays(double[] array1, double[] array2)
    {
    	double[] result = new double[array1.length];
    	for(int i = 0; i < result.length; i++)
    	{
    		result[i] = array1[i] - array2[i];
    	}
    	return result;
    }
    
    public static double[] addArrays(double[] array1, double[] array2)
    {
    	double[] result = new double[array1.length];
    	for(int i = 0; i < result.length; i++)
    	{
    		result[i] = array1[i] + array2[i];
    	}
    	return result;
    }
    
    public static double[][] addDimension(double[] array)
    {
    	double[][] result = new double[array.length][1];
    	for(int i = 0; i < array.length; i++)
    	{
    		result[i][0] = array[i];
    	}
    	return result;
    }
    
    
    public static int choose(int n, int k)
    {
    	int num = 1;
    	int denom = 1;
    	for(int i = 0; i < k; i++)
    	{
    		num *= n--;
    		denom *= (i + 1);
    	}
    	return num / denom;
    }
    
    public static int factorial(int n)
    {
    	int prod = 1;
    	for(int i = n; i > 0; i--)
    	{
    		prod = prod * n;
    	}
    	return prod;
    }
    
    public static double[] normalize(double[] values)
    {
    	double sum = 0;
    	for(int i = 0; i < values.length; i++)
    	{
    		sum += values[i];
    	}
    	for(int i = 0; i < values.length; i++)
    	{
    		values[i] /= sum;
    	}
    	return values;
    }
    
    public static double[] normalize(int[] values)
    {
    	double[] vals = new double[values.length];
    	for(int i = 0; i < vals.length; i++)
    	{
    		vals[i] = (double)values[i];
    	}
    	return normalize(vals);
    }
    
    public static double[] removeZeros(double[] values)
    {
        int counter = 0;
        for(int i = 0; i < values.length; i++)
        {
            if(values[i] != 0)
                counter++;
        }
        double[] result = new double[counter];
        counter = 0;
        for(int i = 0; i < values.length; i++)
        {
            if(values[i] != 0)
            {
                result[counter] = values[i];
                counter++;
            }
        }
        return result;
    }
    
    public static double[][] removeZeros(double[][] values)
    {
    	int numKeepRows = 0;
    	for(int i = 0; i < values.length; i++)
    	{
    		boolean removeRow = true;
    		for(int j = 0; j < values[i].length; j++)
    		{
    			if(values[i][j] != 0)
    			{
    				removeRow = false;
    				break;
    			}	
    		}
    		if(removeRow == false)
    			numKeepRows++;
    	}
    	double[][] result = new double[numKeepRows][values[0].length];
    	int rowCounter = 0;
    	for(int i = 0; i < values.length; i++)
    	{
    		boolean removeRow = true;
    		for(int j = 0; j < values[i].length; j++)
    		{
    			if(values[i][j] != 0)
    			{
    				removeRow = false;
    				break;
    			}	
    		}
    		if(removeRow == false)
    		{
    			for(int j = 0; j < values[i].length; j++)
    			{
    				result[rowCounter][j] = values[i][j];
    			}
    			rowCounter++;
    		}
    	}
    	
    	return result;
    }
    public static double[] quartiles(double[] data)
    {
        Arrays.sort(data);
        double length = data.length;
        double n1 = 1.0 / 4.0 * (length + 1);
        double q1 = data[(int)n1]; //* (1.0 + (n1 - (int)n1));
        double n2 = 2.0 / 4.0 * (length + 1);
        double q2 = data[(int)n2];// * (1.0 + (n2 - (int)n2));
        double n3 = 3.0 / 4.0 * (length + 1);
        double q3 = data[(int)n3];// * (1 + (n3 - (int)n3));
        double n4 = 4.0 / 4.0 * (length + 1);
        double q4 = data[data.length - 1];
        double[] quartiles = {q1, q2, q3, q4};
        return quartiles;
    }
    
    public static void printArray(double[] arr)
    {
        for(int i = 0; i < arr.length; i++)
        {
            System.out.print(arr[i]);
            if(i != arr.length - 1)
                System.out.print(",");
        }
        System.out.println();
    }
    
    public static int maxIndex(double[] arr)
    {
        double max = Double.NEGATIVE_INFINITY;
        int maxIndex = -1;
        for(int i = 0; i < arr.length; i++)
        {
            if(arr[i] > max)
            {
                max = arr[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    public static double findPercentile(double[] data, double percentile)
    {
        Arrays.sort(data);
        int index = (int)(percentile * (double)data.length);
        return data[index];
    }
    
    public static double[] arrayCopy(double[] array)
    {
        double[] copy = new double[array.length];
        for(int i = 0; i < array.length; i++)
        {
            copy[i] = array[i];
        }
        return copy;
    }
    
    public static double min(double[] array)
    {
        double min = Double.POSITIVE_INFINITY;
        for(int i = 0; i < array.length; i++)
        {
            if(array[i] < min)
                min = array[i];
        }
        return min;
    }
    
    public static double max(double[] array)
    {
        double max = Double.NEGATIVE_INFINITY;
        for(int i = 0; i < array.length; i++)
        {
            if(array[i] > max)
                max = array[i];
        }
        return max;
    }
    
    public static double pearsonsR(double[] vals1, double[] vals2)
    {
        double mean1 = mean(vals1);
        double mean2 = mean(vals2);
        
        double covSum = 0;
        for(int i = 0; i < vals1.length; i++)
        {
            covSum += (vals1[i] - mean1)*(vals2[i] - mean2);
        }
        covSum /= ((double)vals1.length - 1.0);
        covSum /= Math.sqrt(variance(vals1));
        covSum /= Math.sqrt(variance(vals2));
        return covSum;
    }
    
    public static double calculateAUC(int c1, int c2, int [] predictedLabels,  int [] realLabels, boolean convex) {
		
		double auc = 0;
		// find all example with realLabel= C1 && (predictedLabels = C1 or C2);
		ArrayList<Integer> realC1 = new ArrayList<Integer>();
		ArrayList<Double>  predC12 = new ArrayList<Double>();
		
		for (int i = 0; i < realLabels.length; i++) {
			if ((realLabels[i] == c1) && ((predictedLabels[i] == c1) ||  (predictedLabels[i] == c2))) {
				realC1.add(realLabels[i]);
				predC12.add(new Double(predictedLabels[i]));
			}
			if ((realLabels[i] == c2) && ((predictedLabels[i] == c1) ||  (predictedLabels[i] == c2))) {
				realC1.add(realLabels[i]);
				predC12.add(new Double(predictedLabels[i]));
			}
		}
		
		if (realC1.size() > 0) {
			int [] real = new int[realC1.size()];
			double [] pred = new double [realC1.size()];
			for (int i = 0; i < realC1.size(); i++) {
				if (realC1.get(i).intValue() == c2)
					real[i] = 1;
				else
					real[i] = 0;
				if (Misc.isDoubleSame(predC12.get(i).doubleValue(), (double)c2, 0.0000001))
					pred[i] = 1.0;
				else
					pred[i] = 0.0;
			}
			mloss.roc.Curve rocAnalysis = new mloss.roc.Curve.PrimitivesBuilder().predicteds(pred).actuals(real).build();
			if (convex) {
				// Get the convex hull
			    mloss.roc.Curve convexHull = rocAnalysis.convexHull();
			    auc = convexHull.rocArea();
			    if (Double.isNaN(auc))
			    	auc = 0;
			    //log.debug("AUC_Convex ("+c1 + "," + c2 +")=" + auc);
			} else {
				auc = rocAnalysis.rocArea();
				if (Double.isNaN(auc))
				    auc = 0;
				//log.debug("AUC ("+c1 + "," + c2 +")=" + auc + "    ");
			}
		}
		
		return auc;
	}
    
    public static int sumOfElements(double[][] matrix)
    {
        int total = 0;
        for(int i = 0; i < matrix.length; i++)
        {
            for(int j = 0; j < matrix[i].length; j++)
            {
                total += (int)matrix[i][j];
            }
        }
        return total;
    }
    
    public static double getAUC(double[][] confusionMatrix)
    {
        int dim = confusionMatrix.length;
        int total = (int)sumOfElements(confusionMatrix);
        int[] predLabels = new int[total];
        int[] trueLabels = new int[total];
        int index = 0;
        double auc = 0;
        int numCategory = confusionMatrix.length;
        for(int i = 0; i < confusionMatrix.length; i++)
        {
            for(int j = 0; j < confusionMatrix.length; j++)
            {
                int elem = (int)confusionMatrix[i][j];
                for(int k = 0; k < elem; k++)
                {
                    predLabels[index] = j;
                    trueLabels[index] = i;
                    index++;
                }
            }
        }
        for (int i = 0; i < numCategory - 1; i++)	
        {
                for (int j = i + 1; j < numCategory; j++)
                {
                        auc += calculateAUC(i, j, predLabels, trueLabels, false);
                }
        }
       auc = (2 * auc) / (double) (numCategory * (numCategory - 1));
       return auc;
    }
    
    public static double[] scaleArray(double[] array, double scalar)
    {
        double[] result = new double[array.length];
        for(int i = 0; i < array.length; i++)
        {
            result[i] = scalar * array[i];
        }
        return result;
    }
    
    public static ArrayList<Double> getFirstHalfArrayList(double[] array)
    {
        ArrayList<Double> result = new ArrayList();
        for(int i = 0; i < 9; i++)
        {
            result.add(array[i]);
        }
        return result;
    }
    
    public static ArrayList<Double> getSecondHalfArrayList(double[] array)
    {
        ArrayList<Double> result = new ArrayList();
        for(int i = 9; i < 18; i++)
        {
            result.add(array[i]);
        }
        return result;
    }
}

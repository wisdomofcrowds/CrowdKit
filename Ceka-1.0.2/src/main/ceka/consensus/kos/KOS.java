
package ceka.consensus.kos;

import ceka.consensus.mv.MajorityVote;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.core.Worker;
import java.util.ArrayList;
import java.util.Random;

/**
 * Article:<br>
 * David R Karger, Sewoong Oh, and Devavrat Shah. Iterative learning for reliable crowd-
sourcing systems. In Advances in neural information processing systems, pages 1953-1961,
2011.
 * @author Bryce
 */
public class KOS 
{
    private int maxIterations = 5;
    
    public static final String NAME = "KOS";
    public KOS(int maxIterations)
    {
        this.maxIterations = maxIterations;
    }
    
    /**
     * Performs the KOS Inference technique, using message vectors to calculate
     * iteratively the likelihood of the true label being a certain class.
     * @param dataset 
     */
    public void doInference(Dataset dataset)
    {
        Random rand = new Random();
        double[][] inferenceArray = new double[dataset.getCategorySize()][dataset.getExampleSize()];
        ArrayList<Worker> workers = new ArrayList();
        for(int i = 0; i < dataset.getExampleSize(); i++)
        {
            Example e = dataset.getExampleByIndex(i);
            for(int j = 0; j < e.getMultipleNoisyLabelSet(0).getLabelSetSize(); j++)
            {
                Worker w = dataset.getWorkerById(e.getMultipleNoisyLabelSet(0).getLabel(j).getWorkerId());
                if(!existsIn(workers, w))
                    workers.add(w);
            }
        }
        Graph graph = createGraph(dataset, workers);
        
        for(int c = 0; c < dataset.getCategorySize(); c++)
        {
            int positiveClass = c;
            int[][] A = new int[dataset.getExampleSize()][workers.size()];

            for(int i = 0; i < dataset.getExampleSize(); i++)
            {
                Example e = dataset.getExampleByIndex(i);
                for(int j = 0; j < workers.size(); j++)
                {
                    Label lab = null;
                    for(int k = 0; k < e.getMultipleNoisyLabelSet(0).getLabelSetSize(); k++)
                    {
                        Label l = e.getMultipleNoisyLabelSet(0).getLabel(k);
                        if(workerMatch(l, workers.get(j)))
                        {
                            lab = l;
                            break;
                        }
                    }
                    if(lab == null)
                        A[i][j] = 0;
                    else if(lab.getValue() == positiveClass)
                        A[i][j] = 1;
                    else
                        A[i][j] = -1;
                }
            }

            double[][] y = new double[dataset.getExampleSize()][workers.size()];
            double[][] x = new double[dataset.getExampleSize()][workers.size()];

            for(Node n : graph.getNodes())
            {
                if(n.getElement() instanceof Worker)
                {
                    ArrayList<Example> examples = new ArrayList<Example>(n.getNeighbors());
                    for(int i = 0; i < examples.size(); i++)
                    {
                        y[i][n.getIndex()] = rand.nextGaussian() + 1;
                    }
                }
            }

            for(int iteration = 0; iteration < maxIterations; iteration++)
            {
                for(Node n : graph.getNodes())
                {
                    if(n.getElement() instanceof Example)
                    {
                        ArrayList<Worker> ws = new ArrayList(n.getNeighbors());
                        for(int i = 0; i < ws.size(); i++)
                        {
                            double sum = 0;
                            for(int j = 0; j < ws.size(); j++)
                            {
                                if(i != j)
                                {
                                    sum += A[n.getIndex()][j] * y[n.getIndex()][j];
                                }
                            }
                            x[n.getIndex()][i] = sum;
                        }
                    }
                }
                
                for(Node n : graph.getNodes())
                {
                    if(n.getElement() instanceof Worker)
                    {
                        ArrayList<Example> examples = new ArrayList(n.getNeighbors());
                        for(int i = 0; i < examples.size(); i++)
                        {
                            double sum = 0;
                            for(int j = 0; j < examples.size(); j++)
                            {
                                if(i != j)
                                {
                                    sum += A[j][n.getIndex()] * x[j][n.getIndex()];
                                }
                            }
                            y[i][n.getIndex()] = sum;
                        }
                    }
                }
            }

            double[] x2 = new double[dataset.getExampleSize()];
            for(Node n : graph.getNodes())
            {
                if(n.getElement() instanceof Example)
                {
                    double sum = 0;
                    ArrayList<Worker> ws = new ArrayList(n.getNeighbors());
                    for(int j = 0; j < ws.size(); j++)
                    {
                        sum += A[n.getIndex()][j] * y[n.getIndex()][j];
                    }
                    x2[n.getIndex()] = sum;
                }
            }
            inferenceArray[c] = x2;
        }
        
        for(int i = 0; i < dataset.getExampleSize(); i++)
        {
            double max = Double.NEGATIVE_INFINITY;
            int maxC = -99;
            for(int c = 0; c < dataset.getCategorySize(); c++)
            {
                if(inferenceArray[c][i] > max)
                {
                    max = inferenceArray[c][i];
                    maxC = c;
                }
            }
            Example e = dataset.getExampleByIndex(i);
            e.setIntegratedLabel(new Label(null, "" + maxC, e.getId(), NAME));
        }
        // this is important
     	dataset.assignIntegeratedLabel2WekaInstanceClassValue();
    }
    
    private boolean existsIn(ArrayList<Worker> workers, Worker w)
    {
        for(Worker worker : workers)
        {
            if(w.equals(worker))
            {
                return true;
            }
        }
        return false;
    }
    
    private boolean workerMatch(Label l, Worker w)
    {
        return(l.getWorkerId().equals(w.getId()));
    }
    
    private Worker labelWorker(Label l, ArrayList<Worker> workers)
    {
        for(Worker w : workers)
        {
            if(w.getId().equals(l.getWorkerId()))
                return w;
        }
        return null;
    }
    
    private int indexOfWorker(Label l, ArrayList<Worker> workers)
    {
        for(int i = 0; i < workers.size(); i++)
        {
            if(workers.get(i).getId().equals(l.getWorkerId()))
                return i;
        }
        
        return -1;
    }
    
    private Graph createGraph(Dataset dataset, ArrayList<Worker> workers)
    {
        Graph g = new Graph();
        for(int i = 0; i < dataset.getExampleSize(); i++)
        {
            Example e = dataset.getExampleByIndex(i);
            g.addNode(new Node<Example, Worker>(e, i));
            for(int j = 0; j < e.getMultipleNoisyLabelSet(0).getLabelSetSize(); j++)
            {
                MultiNoisyLabelSet mnls = e.getMultipleNoisyLabelSet(0);
                Label l = mnls.getLabel(j);
                dataset.getWorkerById(mnls.getLabel(j).getWorkerId());
                g.getNode(i).addNeighbor(labelWorker(l, workers), indexOfWorker(l, workers));
            }
        }
        for(int i = 0; i < workers.size(); i++)
        {
            g.addNode(new Node<Worker, Example>(workers.get(i), i));
            for(int j = 0; j < dataset.getExampleSize(); j++)
            {
                Example e = dataset.getExampleByIndex(j);
                for(int k = 0; k < e.getMultipleNoisyLabelSet(0).getLabelSetSize(); k++)
                {
                    Label l = e.getMultipleNoisyLabelSet(0).getLabel(k);
                    if(workerMatch(l,workers.get(i)))
                        g.getNode(g.numNodes() - 1).addNeighbor(e, j);
                }  
            }
        }
        
        return g;
    }
    
    private class Graph
    {
        private ArrayList<Node> nodes;
        
        public Graph() 
        {
            nodes = new ArrayList();
        }
        
        public ArrayList<Node> getEdges()
        {
            return nodes;
        }
        
        public void addNode(Node n)
        {
            nodes.add(n);
        }
        
        public Node getNode(int index)
        {
            return nodes.get(index);
        }
        
        public ArrayList<Node> getNodes()
        {
            return nodes;
        }
        
        public int numNodes()
        {
            return nodes.size();
        }
    }
    
    class Node<T1, T2>
    {
        T1 t1;
        int t1Index;
        ArrayList<T2> t2s;
        ArrayList<Integer> t2Indexes;
        
        public Node(T1 t1, int t1Index)
        {
            this.t1 = t1;
            t2s = new ArrayList();
            this.t1Index = t1Index;
            t2Indexes = new ArrayList();
        }
        
        public void addNeighbor(T2 t2, int index)
        {
            t2s.add(t2);
            t2Indexes.add(index);
        }
        
        public ArrayList<T2> getNeighbors()
        {
            return t2s;
        }
        
        public T1 getElement()
        {
            return t1;
        }
        
        public int getIndex()
        {
            return t1Index;
        }
    }
}

//package prediction;
import java.io.*;
import java.util.*;
import java.text.DecimalFormat;
import java.util.concurrent.ThreadLocalRandom;

public class DecisionTree{
	
	private static void predictClassDecisionTree(boolean partitionTrainSet, String trainSetPath, String validationSetPath, boolean headings, int [] columnsToUse, int predictionColumn, int [] splitPoints, String formula) {
		DecimalFormat df = new DecimalFormat("#.####");
		// read train 
		
		ArrayList<ArrayList<Double>> trainSet = CommonMethods.readDatasetFile(trainSetPath, headings);
		ArrayList<ArrayList<Double>> validationSet;
		//split train set into two if needed
		if(partitionTrainSet){
			Pair<ArrayList<ArrayList<Double>>,ArrayList<ArrayList<Double>>> splits = CommonMethods.randomlyHalfSet(trainSet);
			trainSet = splits.getFirst();
			validationSet = splits.getSecond();
		}
		else{
			validationSet = CommonMethods.readDatasetFile(validationSetPath, headings);
		}
		
		//create column structure
		ArrayList<ArrayList<Double>> trainColumns = CommonMethods.createColumnStructure(trainSet);
		ArrayList<ArrayList<Double>> validationColumns = CommonMethods.createColumnStructure(validationSet);
		
		//trim columns which wont be used
		trainColumns = CommonMethods.trimColumns(trainColumns, columnsToUse, predictionColumn);
		validationColumns = CommonMethods.trimColumns(validationColumns, columnsToUse, predictionColumn);
		
		//update row structure
		trainSet = CommonMethods.createRowStructure(trainColumns);
		validationSet = CommonMethods.createRowStructure(validationColumns);
		
		//split classification column into sets if needed, for continuous variables
		trainSet = CommonMethods.classifyDataset(trainSet, splitPoints);
		validationSet = CommonMethods.classifyDataset(validationSet, splitPoints);
		
		//update column structure
		trainColumns = CommonMethods.createColumnStructure(trainSet);
		validationColumns = CommonMethods.createColumnStructure(validationSet);
		
		/*
		for(int i =0; i < trainColumns.size() -1; i++){
			//System.out.println(i + ": " + infoGain(trainColumns,i));
			gainRatio(trainColumns,i);
		*/
		
		ArrayList<Integer> usedAtts = new ArrayList<Integer>();
		
		Node root = new Node(trainColumns, usedAtts);
		
		buildTree(root, "Information Gain");
		
		preOrder(root);
		
	}
	
	public static void preOrder(Node root) {  
			if(root == null)
				return;
				
			System.out.println("split on: " + root.getSplitIndex() + ", atts used: " + root.getUsedAttributes());
			
			if(root.isLeaf())
				return;
			
			for(Double x: root.children.keySet()){
				preOrder(root.children.get(x));
			}
			
	}
	
	static double infoGain(ArrayList<ArrayList<Double>> trainColumns, int targetColumnIndex){ //pred column last index
	
		HashMap<Double,ArrayList<Integer>> classIndexMap = CommonMethods.getColumnValueLocationMap(trainColumns, trainColumns.size()-1);  //hash map: key is class, value is arraylist of indexes
		
		double setEntropy = getEntropy(trainColumns, trainColumns.size()-1);
		double setSize = (double)trainColumns.get(0).size();
		
		HashMap<Double,ArrayList<Integer>> columnValueIndexMap = CommonMethods.getColumnValueLocationMap(trainColumns, targetColumnIndex);
		
		double columnEntropy = 0.0;
		for(Double currentColumnValue: columnValueIndexMap.keySet()){
			double valueCount = (double)columnValueIndexMap.get(currentColumnValue).size();
			double valueRatio = (valueCount/setSize);
			
			double accum = 0.0;
			for(Double currentClass: classIndexMap.keySet()){
				double commonCount = 0.0;
				for(Integer index: classIndexMap.get(currentClass)){
					if(columnValueIndexMap.get(currentColumnValue).contains(index)){
						commonCount++;
					}
				}
			
				double ratio = (commonCount/valueCount);
				double calc;
				if (ratio > 0.0)
					calc = (-1 * ratio * (Math.log(ratio)/Math.log(2)));
				else
					calc = 0.0;
				
				accum += calc;
				//System.out.println("columnValue: " + currentColumnValue + ", currentColumnValue size: " + valueCount + ", class: " + currentClass + ", count: " + commonCount + ", calc: " + calc);
			}
			columnEntropy += (valueRatio * accum);
			//System.out.println();
		}
		
		//System.out.println("Set entropy: " +setEntropy + ", Column entropy: " + columnEntropy);
		
		double gain = (setEntropy - columnEntropy);
		
		return gain;
	}
	
	static double gainRatio(ArrayList<ArrayList<Double>> trainColumns, int targetColumnIndex){
		
		double infoGain = infoGain(trainColumns, targetColumnIndex);
		HashMap<Double,ArrayList<Integer>> columnValueIndexMap = CommonMethods.getColumnValueLocationMap(trainColumns, targetColumnIndex);
		double setSize = (double)trainColumns.get(0).size();
		
		double splitInfo = 0.0;
		for(Double currentClass: columnValueIndexMap.keySet()){
			double size = (double)columnValueIndexMap.get(currentClass).size();
			double ratio = (size/setSize);
			double calc;
				if (ratio > 0.0)
					calc = (-1 * ratio * (Math.log(ratio)/Math.log(2)));
				else
					calc = 0.0;
			splitInfo += calc;
		}
		
		double gainRatio = (infoGain/splitInfo);
		//System.out.println("info gain: " + infoGain + ", gainRatio: " + gainRatio);
		return gainRatio;
		
	}
	
	static double getEntropy(ArrayList<ArrayList<Double>> trainColumns, int columnIndex){
		HashMap<Double,ArrayList<Integer>> classIndexMap = CommonMethods.getColumnValueLocationMap(trainColumns, columnIndex);  //hash map: key is class, value is arraylist of indexes
		
		double setEntropy = 0.0;
		double setSize = (double)trainColumns.get(columnIndex).size();
		for(Double currentClass: classIndexMap.keySet()){
			double size = (double)classIndexMap.get(currentClass).size();
			double ratio = (size/setSize);
			double calc;
				if (ratio > 0.0){
					calc = (-1 * ratio * (Math.log(ratio)/Math.log(2)));
					/*
					if(calc == 0.0){
						System.out.println("proper 0 entropy");
					}
					*/
				}
				else
					System.out.println("Problem - should never be in here");
					calc = 0.0;
			setEntropy += calc;
			//System.out.println("class: " + currentClass + ", size: " + size + "/" + setSize + ", ratio: " + ratio + ", Entropy: " + calc);
		}
		return setEntropy;
	}
	

	static int selectAttribute(ArrayList<ArrayList<Double>> trainColumns, String formula, ArrayList<Integer> usedAttributes){ //returns index of attribute to be used for next split
		
			if(formula.equals("Gain Ratio")){
				int bestIndex = -1;
				double bestGR = 0.0;
				for(int i = 0; i < trainColumns.size()-1; i++){ //last element is pred column
					if(!usedAttributes.contains(i)){
						double currentGR = gainRatio(trainColumns,i);
						//System.out.print("row: " + i + ", GR: " + currentGR + " /// ");
						if(currentGR > bestGR){
							bestGR = currentGR;
							bestIndex = i;
						}
					}
					
				}
				//System.out.println();
				return bestIndex;
			}
			else{ //information gain
				int bestIndex = -1;
				double bestIG = 0.0;
				for(int i = 0; i < trainColumns.size()-1; i++){ //last element is pred column
					if(!usedAttributes.contains(i)){
						double currentIG = infoGain(trainColumns,i);
						if(currentIG > bestIG){
							bestIG = currentIG;
							bestIndex = i;
						}
					}
	
				}
				return bestIndex;
			}
	}
	
	private static Double getMaxCount(HashMap<Double,Double> classCounts){
		//create tree map with comparitor which sorts by value instead of key
		TreeSet<Map.Entry<Double, Double>> entriesSet = new TreeSet<>(new Comparator<Map.Entry<Double, Double>>(){
           @Override 
			public int compare(Map.Entry<Double, Double> x, Map.Entry<Double, Double> y) {
				return x.getValue().compareTo(y.getValue());
			}
        });
        entriesSet.addAll(classCounts.entrySet());
		return entriesSet.last().getKey();
	}
	
	public static void main(String [] args) {
	
		//predictClassBayes(true, "student-mat-normalised.csv", null, true, new int[]{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18},32, new int[]{7,14});
		//predictClassBayes(true, "IrisNumerical.csv", null, true, new int[]{1,2,3,4},5, null);
		predictClassDecisionTree(true, "CarDataNumerical.csv", null, true, new int[]{0,1,2,3,4,5},6, null, "Gain Ratio");
		
		
	}
	
	static Node buildTree(Node root, String formula) {
		//check if leaf
			//1 class left - entropy will be 0
			ArrayList<ArrayList<Double>> currentColumns = root.getData();
			Double entropy = getEntropy(currentColumns, (currentColumns.size()-1));
			if(CommonMethods.equalsDouble(entropy,0.0)){
				root.setLeaf(true);
				return root;
			}
		
			//fully split - usedAttributes will be = to columns.size -1
			
			if(root.getUsedAttributes().size() == (currentColumns.size()-1)){
				root.setLeaf(true);
				return root;
			}
		
			
		//if not - split & get children and call on them
		
		int bestAttribute = selectAttribute(currentColumns,formula, root.getUsedAttributes());
		
		HashMap<Double,ArrayList<Integer>> columnValueIndexMap = CommonMethods.getColumnValueLocationMap(currentColumns, bestAttribute);
		
		HashMap<Double,ArrayList<ArrayList<Double>>> classColumnStructuresMap = CommonMethods.getClassColumnStructuresMap(CommonMethods.createRowStructure(currentColumns), columnValueIndexMap);
		
		root.setSplitIndex(bestAttribute);
		
		ArrayList<Integer> atts = root.getUsedAttributes();
		atts.add(bestAttribute);
		
		for (Double a: classColumnStructuresMap.keySet()) {
			Node currentChild = new Node(classColumnStructuresMap.get(a),atts);
			root.children.put(a,currentChild);
		}
		
		for(Double x: root.children.keySet()){
			buildTree(root.children.get(x),formula);
		}
		
		//return
		
		return root;
		
	}
	
}

class Node {
	public HashMap<Double, Node> children;
	private ArrayList<ArrayList<Double>> trainColumns;
	private ArrayList<Integer> usedAttributes;
	private int splitIndex;
	boolean isLeaf;
	
	public Node(ArrayList<ArrayList<Double>> trainColumns, ArrayList<Integer> usedAttributes) {
		this.trainColumns = trainColumns;
		this.usedAttributes = usedAttributes;
		setSplitIndex(-1);
		setLeaf(false);
	}
	
	public ArrayList<Integer> getUsedAttributes(){
		return this.usedAttributes;
	}

	public ArrayList<ArrayList<Double>> getData() {
		return trainColumns;
	}

	public void setLeaf(boolean isLeaf) {
		this.isLeaf = isLeaf;
	}

	public boolean isLeaf() {
		return isLeaf;
	}
	
	public void setSplitIndex(int splitIndex) {
		this.splitIndex = splitIndex;
	}

	public int getSplitIndex() {
		return splitIndex;
	}
}


















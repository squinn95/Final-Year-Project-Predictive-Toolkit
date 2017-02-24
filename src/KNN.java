//package prediction;
import java.io.*;
import java.util.*;
import java.text.DecimalFormat;
import java.util.concurrent.ThreadLocalRandom;

public class KNN{
	
	private static void predictClassKNN(boolean partitionTrainSet, String trainSetPath, String validationSetPath, boolean headings, int [] columnsToUse, int predictionColumn, int [] splitPoints, int k, String formula ) {
		
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
		
		//convert trimmed columns back to row structure
		trainSet = CommonMethods.createRowStructure(trainColumns);
		validationSet = CommonMethods.createRowStructure(validationColumns);
		
		//normalise set
		trainSet = CommonMethods.normaliseDataset(trainSet, trainColumns);
		validationSet = CommonMethods.normaliseDataset(validationSet, validationColumns);
		
		//split classification column into sets if needed, for continuous variables
		trainSet = CommonMethods.classifyDataset(trainSet, splitPoints);
		validationSet = CommonMethods.classifyDataset(validationSet, splitPoints);
		
		Iterator<ArrayList<Double>> iter6 = validationSet.iterator();
		
		double rightCount = 0;
		int g = 0;
		while (iter6.hasNext()) { // classify validation set one by one using knn and add to resultsMap
		
			int currentK = k;
			ArrayList<Double> current = iter6.next(); // current validation row
			
			TreeMap<Double,ArrayList<Double>> distanceMap = genDistanceMap(current, trainSet, formula);
			
			TreeMap<Double,ArrayList<Double>> neighbourMap = getKClosest(distanceMap, currentK);
			
			HashMap<Double,Double> classCounts = getClassCounts(neighbourMap);
			
			while(!isOneMax(classCounts)){ //in event of tie reduce k by 1 and re-run
				currentK--;
				neighbourMap = getKClosest(neighbourMap, currentK);
				classCounts = getClassCounts(neighbourMap);
			}
			
			Double classValue = getMaxClass(classCounts);
			/*
			if(o == 50){
				for(Double a: classCounts.keySet())
					System.out.println(a + ": " + classCounts.get(a));
				System.out.println(classValue);
			}
			*/
			
			if(CommonMethods.equalsDouble(current.get(current.size()-1),classValue))
				rightCount++;

			g++;
			
		}
		
		DecimalFormat df = new DecimalFormat("#.####");
		double success = Double.parseDouble(df.format((rightCount/g)*100)); // percentage success rounded to 4 places
		System.out.println(rightCount + " predictions out of " + g + " correct");
		System.out.println("K nearest neighbours classifier built with " + success + "% accuracy");
		// System.out.println("total " + g);
	
	}
	
	private static HashMap<Double, Double> getClassCounts(TreeMap<Double,ArrayList<Double>> neighbourMap){
		HashMap<Double, Double> classCounts = new HashMap<Double, Double>();
		for(Double x: neighbourMap.keySet()){ //distances
			ArrayList<Double> current = neighbourMap.get(x); // class values associated with distance x
			for(Double a: current){ //for each class value associated with current dist
				if(classCounts.containsKey(a)){
					Double count = classCounts.get(a);
					count++;
					classCounts.put(a, count);
				}
				else{
					classCounts.put(a, 1.0);
				}
			}
		}
		return classCounts;
	}
	
	private static double euclideanDistance(ArrayList<Double> a, ArrayList<Double> b){ 
		DecimalFormat df = new DecimalFormat("#.####");
		double total = 0;
		for(int i = 0; i < a.size() - 1; i++){
			double difference = (a.get(i) - b.get(i));
			total += (difference * difference); //replace 1 with weighting scheme
		}
		return Double.parseDouble(df.format(Math.sqrt(total)));
	}
	
	private static double manhattanDistance(ArrayList<Double> a, ArrayList<Double> b){ 
		DecimalFormat df = new DecimalFormat("#.####");
		double total = 0;
		for(int i = 0; i < a.size() - 1; i++){
			double difference = (a.get(i) - b.get(i));
			total += Math.abs(difference); 
		}
		return total;
	}
	
	private static double calcDistance(ArrayList<Double> a, ArrayList<Double> b, String formula){
			if(formula.equals("manhattan"))
				return manhattanDistance(a, b);
			else
				return euclideanDistance(a, b);
	}
	
	private static ArrayList<Double> getNRandomListDoubles(ArrayList<Double> inputList, int n){
		ArrayList<Double> input = new ArrayList<Double>(inputList); //because arraylists passed by reference
		ArrayList<Double> output = new ArrayList<Double>();
		for(int i = 0; i < n; i++){
			int rnd = ThreadLocalRandom.current().nextInt(0, input.size());
			output.add(input.get(rnd));
			input.remove(rnd);
		}
		return output;
	}
	
	private static TreeMap<Double,ArrayList<Double>> getKClosest(TreeMap<Double,ArrayList<Double>> input, int k){
		TreeMap<Double,ArrayList<Double>> distanceMap = new TreeMap<Double,ArrayList<Double>>(input); //because of pass by reference
		TreeMap<Double,ArrayList<Double>> neighbourMap = new TreeMap<Double,ArrayList<Double>>();
		int j = 0;
		while (j < k) { // k nearest neighbours
			Map.Entry<Double,ArrayList<Double>> a = distanceMap.pollFirstEntry(); // returns and removes smallest key entry in treemap - closest neighbour    (Store these in structure, method to get dominant one)
			int q = a.getValue().size();
					
			if((q + j) > k){
				int n = k - j; //choose n from q;
				ArrayList<Double> newA = getNRandomListDoubles(a.getValue(), n);
				neighbourMap.put(a.getKey(),newA);
				j = j + n;
			}
			else{
				neighbourMap.put(a.getKey(),a.getValue());
				j = j + q;
			}
		}
		return neighbourMap;
	}
	
	private static TreeMap<Double,ArrayList<Double>> genDistanceMap(ArrayList<Double> valRow, ArrayList<ArrayList<Double>> trainSet, String formula) {
		TreeMap<Double,ArrayList<Double>> distanceMap = new TreeMap<Double,ArrayList<Double>>(); // this will hold (dist, classes) pairs. TreeMap keeps them in sorted order - can use pollFirstEntry
		Iterator<ArrayList<Double>> iter7 = trainSet.iterator(); // calculate distance from this to each row in training set
		while (iter7.hasNext()) {
			ArrayList<Double> trainRow = iter7.next();
				
			double dist = calcDistance(valRow, trainRow, formula);
			if(distanceMap.containsKey(dist)){
				ArrayList<Double> t = distanceMap.get(dist);
				t.add(trainRow.get(trainRow.size() -1));
				distanceMap.put(dist, t);
			}
			else{
				ArrayList<Double> b = new ArrayList<Double>();
				b.add(trainRow.get(trainRow.size() -1));
				distanceMap.put(dist, b); //  last index holds class value of training set
			}
		}
		return distanceMap;
	}
	
	
	private static boolean isOneMax(HashMap<Double,Double> classCounts){
		//sort 
		//if next index the same as first then there is at least two max therefore return false
		if(classCounts.keySet().size() == 1){
				return true;
		}
		else{
			ArrayList<Double> a = new ArrayList<Double>();
			for(Double x: classCounts.keySet())
				a.add(classCounts.get(x));
			Collections.sort(a);
			Collections.reverse(a); //now in desc order
			if(a.get(0).equals(a.get(1))){
				return false;
			}
			else{
				return true;
			}
		}
	}
	
	private static Double getMaxClass(HashMap<Double,Double> classCounts){
		//create tree map with comparitor which sorts by value instead of key
		TreeSet<Map.Entry<Double, Double>> entriesSet = new TreeSet<>(new Comparator<Map.Entry<Double, Double>>(){
           @Override 
			public int compare(Map.Entry<Double, Double> me1, Map.Entry<Double, Double> me2) {
				return me1.getValue().compareTo(me2.getValue());
			}
        });
        entriesSet.addAll(classCounts.entrySet());
		return entriesSet.last().getKey();
	}
	

	public static void main(String [] args) {
	
		predictClassKNN(true, "student-mat-normalised.csv", null, true, new int[]{5,7,8,9,10,11,12,13,14,15,16,28},32, new int[]{7,14}, 5, "euclidean");
		//{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}
		
	}
	
}

















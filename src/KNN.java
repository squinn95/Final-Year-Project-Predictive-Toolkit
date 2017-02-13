//package prediction;
import java.io.*;
import java.util.*;
import java.text.DecimalFormat;
import java.util.concurrent.ThreadLocalRandom;

public class KNN{

	/*
	private ArrayList<ArrayList<Double>> trainSet;
	private ArrayList<ArrayList<Double>> trainColumns;	// needed to normalise values & calculate weightings
	//private static double [] euclideanWeighting;
	private static DecimalFormat df; // limits to 4 decimal places
	private String path;        //API input value
	private Boolean headings;//API input value
	private int numColumns;//API input value
	private ArrayList<Integer> columnsToUse;//API input value
	private int predictionColumn;//API input value
	private int [] splitPoints; //API input value - a list of splitPoint values in ascending order, eg, (7,14), for continuous values or just [] for dicrete

	public KNN(String path, boolean partitionTrainSet, Boolean headings, int numColumns, int predictionColumn, int [] splitPoints) {
		this.path = path;/////
		this.headings = headings;
		this.numColumns = numColumns; 
		this.predictionColumn = predictionColumn;
		this.splitPoints = splitPoints;
		trainSet = new ArrayList<>();
		trainColumns = new ArrayList<>();
		df = new DecimalFormat("#.####"); // limits decimals to 4 decimal places
		
		for (int i = 0; i < numColumns; i++ ) { // this just instantiates the blank arraylists in columns so that readDataset works
			ArrayList<Double> x = new ArrayList<>();
			columns.add(x);
		}
		
	}
	*/
	
	//////////////////////////////////////////////////////////////
	
	private static void predictClassKNN(boolean partitionTrainSet, ArrayList<ArrayList<Double>> trainSet, ArrayList<ArrayList<Double>> validationSet, int k ) {
	
		if(partitionTrainSet){
			Pair<ArrayList<ArrayList<Double>>,ArrayList<ArrayList<Double>>> splits = CommonMethods.randomlyHalfSet(trainSet);
			trainSet = splits.getFirst();
			validationSet = splits.getSecond();
		}
		
		HashMap<ArrayList<Double>,Double> resultsMap = new HashMap<>(); // this will hold (row, predictedClass) prediction pairs.
		Iterator<ArrayList<Double>> iter6 = validationSet.iterator();
		
		//int o =0;
		while (iter6.hasNext()) { // classify validation set one by one using knn and add to resultsMap
		
			int currentK = k;
			ArrayList<Double> current = iter6.next(); // current validation row
			
			TreeMap<Double,ArrayList<Double>> distanceMap = genDistanceMap(current, trainSet);
			
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
			resultsMap.put(current, classValue);
			//assign resulting class
			
			//o++;
				
		}
			
		// next step is check success rate, compare prediction with actual
		DecimalFormat df = new DecimalFormat("#.####");
		double rightCount = 0;
		int g = 0;
		Iterator it = resultsMap.entrySet().iterator();
		while (it.hasNext()) {
			Map.Entry prediction = (Map.Entry)it.next();
			// System.out.println("Actual: " + ((ArrayList<Double>)prediction.getKey()).get(33) + " Predicted: " + prediction.getValue());
			ArrayList<Double> t = (ArrayList<Double>)prediction.getKey();
			System.out.print("(" + t.get(t.size()-1) + "," + prediction.getValue() +"): ");
			if ((t.get(t.size()-1)).equals(prediction.getValue()))				// if predicted class = real class
				rightCount++;
			g++;
		}
		double success = Double.parseDouble(df.format((rightCount/g)*100)); // percentage success rounded to 4 places
		System.out.println(rightCount + " predictions out of " + g + " correct");
		System.out.println(success + "% of predictions correct");
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
	
	private static double weightedEuclideanDistance(ArrayList<Double> a, ArrayList<Double> b){ 
		DecimalFormat df = new DecimalFormat("#.####");
		double total = 0;
		for(int i = 0; i < a.size() -2; i++){  /////////////////////////////this needs to be fixed, at what stage in the program will partition take place? not in this method, needs to happen at start
			double difference = (a.get(i) - b.get(i));
			total += (1 * difference * difference); //replace 1 with weighting scheme
		}
		return Double.parseDouble(df.format(Math.sqrt(total)));
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
	
	private static TreeMap<Double,ArrayList<Double>> genDistanceMap(ArrayList<Double> valRow, ArrayList<ArrayList<Double>> trainSet) {
		TreeMap<Double,ArrayList<Double>> distanceMap = new TreeMap<Double,ArrayList<Double>>(); // this will hold (dist, classes) pairs. TreeMap keeps them in sorted order - can use pollFirstEntry
		Iterator<ArrayList<Double>> iter7 = trainSet.iterator(); // calculate distance from this to each row in training set
		while (iter7.hasNext()) {
			ArrayList<Double> trainRow = iter7.next();
				
			double dist = weightedEuclideanDistance(valRow, trainRow);
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
		//KNN z = new KNN("student-mat-normalised.csv", true, true, 33, 32, new int[]{5,10,15});
		ArrayList<ArrayList<Double>> trainSet = CommonMethods.readDatasetFile("student-mat-normalised.csv", true);
		ArrayList<ArrayList<Double>> trainColumns = CommonMethods.createColumnStructure(trainSet);
		trainColumns = CommonMethods.trimColumns(trainColumns, new int[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}, 32);
		trainSet = CommonMethods.createRowStructure(trainColumns);
		trainSet = CommonMethods.normaliseDataset(trainSet, trainColumns);
		trainSet = CommonMethods.classifyDataset(trainSet, new int[]{5,10,15});
		trainColumns = CommonMethods.createColumnStructure(trainSet);
		predictClassKNN(true, trainSet, null, 5);
		
		
	}
	
}

















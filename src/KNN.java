import java.io.*;
import java.util.*;
import java.text.DecimalFormat;

public class KNN{

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

	public KNN(String trainPath, String testPath, boolean partitionTrainSet, Boolean headings, int numColumns, int predictionColumn, int [] splitPoints) {
		this.path = path;/////
		this.headings = headings;
		this.numColumns = numColumns; 
		this.predictionColumn = predictionColumn;
		this.splitPoints = splitPoints;
		trainSet = new ArrayList<>();
		trainColumn = new ArrayList<>();
		df = new DecimalFormat("#.####"); // limits decimals to 4 decimal places
		for (int i = 0; i < numColumns; i++ ) { // this just instantiates the blank arraylists in columns so that readDataset works
			ArrayList<Double> x = new ArrayList<>();
			columns.add(x);
		}
	}

	private void readDataset() {
		String currentRow = "";
		try (BufferedReader br = new BufferedReader(new FileReader(path))) {
			if(headings){
				String headings = br.readLine(); // this takes the column headings out if they are there
			}
			while ((currentRow = br.readLine()) != null) {
				ArrayList<Double> row = toDoubleArrayList(currentRow.split(",")); // splits string into string [] then into ArrayList<double>
				dataset.add(row);
				// add to columns
				for (int i = 0; i < numColumns; i++) {
					columns.get(i).add(row.get(i)); // e.g this will add the school attribute to the arrilist of all the school values for each row
				}
			}
		} 
		catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private void readDataset() {
		
	}
	
	/////////////////////////////////////////////////
	private ArrayList<ArrayList<Double>> readDatasetFile(String filePath) {
		ArrayList<ArrayList<Double>> dataset = new ArrayList<ArrayList<Double>>();
		String currentRow = "";
		try (BufferedReader br = new BufferedReader(new FileReader(filePath))){
			if(headings){
				String headings = br.readLine(); // this takes the column headings out if they are there
			}
			while ((currentRow = br.readLine()) != null) {
				ArrayList<Double> row = toDoubleArrayList(currentRow.split(",")); // splits string into string [] then into ArrayList<double>
				dataset.add(row);
			}
			return dataset;
		} 
		catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private ArrayList<ArrayList<Double>> createColumnStructure(ArrayList<ArrayList<Double>> input){
		//for each row, add element at each index to arraylist at that index in columns
		ArrayList<ArrayList<Double>> columns = new ArrayList<ArrayList<Double>>();
		for (int i = 0; i < numColumns; i++ ) { // this just instantiates the blank arraylists in columns so that readDataset works
			ArrayList<Double> x = new ArrayList<>();
			columns.add(x);
		}
		for(ArrayList<Double> a: input){ //for each row
			for(int i = 0; i < numColumns; i++){
				columns.get(i).add(a.get(i));
			}
		}
	}
	
	
	
	
	
	/////////////////////////
	
	private static ArrayList<Double> toDoubleArrayList(String [] values) {
		ArrayList<Double> result = new ArrayList<>();
		for (String a: values) {
			result.add(Double.parseDouble(a));
		}
		return result;
	}
	
	private void trimDataset(){
		//read idexes of columns to use
		//System.out.println("No. rows before: " + dataset.size());
		//System.out.println("No. columns before: " + columns.size());
		columnsToUse = new ArrayList<Integer>();
		System.out.println("Enter indexs of columns to be used for prediction in range 0 - " + numColumns);
		Scanner s = new Scanner(System.in);
        Scanner numScanner = new Scanner(s.nextLine()); // read indexes in one line
        while (numScanner.hasNextInt()) {
            columnsToUse.add(numScanner.nextInt());
        } //columnsToUse now contains the indexes of columns to be used, predictionColumn holds one to predict
		
		ArrayList<ArrayList<Double>> newColumns = new ArrayList<ArrayList<Double>>();
		for(int a: columnsToUse){
			newColumns.add(columns.get(a));
		} // adds only the columns needed for prediction to newColumns
		newColumns.add(columns.get(predictionColumn)); //add predictionColumn last
		
		ArrayList<ArrayList<Double>> newDataset = new ArrayList<ArrayList<Double>>();
		int columnSize = newColumns.get(0).size();
		
		for(int i = 0; i < columnSize; i++){ //for each row
			ArrayList<Double> currentRow = new ArrayList<Double>();
			for(int j = 0; j < newColumns.size(); j++){ //add each column value
				currentRow.add(newColumns.get(j).get(i));
			}
			newDataset.add(currentRow);
			/*
			for(Double a: currentRow){
				System.out.print(a + " ");
			}
			System.out.println();
			*/
		}
		dataset = newDataset;
		columns = newColumns;
		//System.out.println("No. rows after: " + dataset.size());
		//System.out.println("No. columns after: " + columns.size());
	}
	
	private void normaliseDataset() { // all values in the range 0-1 to stop skewing by attributes such as age
		int size = columns.size() -1; //-1 because we dont want to normalise prediction column
		double [] maxValues = new double [size];
		double [] minValues = new double [size]; // these will hold max and min values for each column
		for (int i = 0; i < size; i++) {
			maxValues[i] = Collections.max(columns.get(i));
			minValues[i] = Collections.min(columns.get(i));
		}
		
		Iterator<ArrayList<Double>> iter = dataset.iterator();
		while (iter.hasNext()) { // loop through rows
			ArrayList<Double> current = iter.next(); // current row
			for (int i = 0; i < size; i++) { // loop through  rows to be normalised
				double oldValue = current.get(i);
				double normValue = Double.parseDouble(df.format((oldValue - minValues[i])/(maxValues[i]- minValues[i]))); // normalisation formula
				current.set(i, normValue);
			}
		}
		/*
		for(ArrayList<Double> a : dataset){ //for each row
			System.out.print("[");
			for(Double x : a){
				System.out.print(x + " ");
			}
			System.out.println("]");
		}
		*/
	}
	
	private void classifyDataset() { // this sets up classes in training set, adds new column, if needed for continuous sets
		if(splitPoints.length > 0){
			Iterator<ArrayList<Double>> iter4 = dataset.iterator();
			while (iter4.hasNext()) { // loop through rows
				ArrayList<Double> current = iter4.next(); // current row
				Double value = current.get(current.size() -1); //last row is predictionColumn since trim
				int classCount = 1;
				boolean assigned = false;
				loop:
				for(int i = 0; i < splitPoints.length; i++){
					if(value < splitPoints[i]){
						//class = classCount
						current.add(new Double(classCount));
						assigned = true;
						break loop;
					}
					classCount++;
				}
				if(assigned == false){
					//class = classCount
					current.add(new Double(classCount));
					assigned = true;
				}
				//System.out.println(value + ": " + classCount );
			}
		}
		/*
		for(ArrayList<Double> a : dataset){ //for each row
			System.out.print("[");
			for(Double x : a){
				System.out.print(x + " ");
			}
			System.out.println("]");
		}
		*/
	}
	
	private void predictClassKNN() { // train on first 200, run on second 195
		HashSet<ArrayList<Double>> trainingSet = new HashSet<>();
		HashSet<ArrayList<Double>> validationSet = new HashSet<>();
		Iterator<ArrayList<Double>> iter5 = dataset.iterator(); // splits into two sets
        int rnd;
		while (iter5.hasNext()) { // loop through rows
			rnd = ThreadLocalRandom.current().nextInt(1, 11);
			if (rnd <= 5)
				trainingSet.add(iter5.next()); // 200 rows
			else
				validationSet.add(iter5.next());// 195
		}
		HashMap<ArrayList<Double>,Double> resultsMap = new HashMap<>(); // this will hold 195 (row, predictedClass) prediction pairs.
		
		Iterator<ArrayList<Double>> iter6 = validationSet.iterator();
		while (iter6.hasNext()) { // classify validation set one by one using knn and add to resultsMap
			TreeMap<Double,Double> distanceMap = new TreeMap<Double,Double>(); // this will hold 200(dist, class) pairs. TreeMap keep them in sorted order - can use pollFirstEntry
			ArrayList<Double> current = iter6.next(); // current validation row
			Iterator<ArrayList<Double>> iter7 = trainingSet.iterator(); // calculate distance from this to each row in training set
			while (iter7.hasNext()) {
				ArrayList<Double> trainRow = iter7.next();
				// System.out.println(trainRow.get(33));
				double dist = weightedEuclideanDistance(current, trainRow);
				// System.out.println(dist);
				distanceMap.put(dist, trainRow.get(33)); //  index 33 holds clasification value
			}

			int [] classCount = new int[4]; // we wont use index 0.
			
			for (int k = 0; k < 5; k++) { // 5 nearest neighbours
				Map.Entry<Double,Double> a = distanceMap.pollFirstEntry(); // returns and removes smallest key entry in treemap - closest neighbour
				classCount[Integer.valueOf(a.getValue().intValue())]++;  // converts double to int and increments index 1,2 or 3 of classCount
				// System.out.print("(" + a.getKey() + "," + a.getValue() + ") ");
			}
			
			if (!isOneMax(classCount)) { // in the event of 2 way tie(2,2,1) - increase to 6 nearest neighbours
				Map.Entry<Double,Double> a = distanceMap.pollFirstEntry();
				classCount[Integer.valueOf(a.getValue().intValue())]++;
				// System.out.print("(" + a.getKey() + "," + a.getValue() + ") ");
				if (!isOneMax(classCount)) { // in the event of 3 way tie(2,2,2) - increase to 7 nearest neighbours
					Map.Entry<Double,Double> b = distanceMap.pollFirstEntry();
					classCount[Integer.valueOf(b.getValue().intValue())]++;
					// System.out.print("(" + b.getKey() + "," + b.getValue() + ") ");
				}
			} // there is now definately only one max
			// System.out.println();
				
			if ((classCount[1] > classCount[2]) && (classCount[1] > classCount[3]))
				resultsMap.put(current, 1.0);
			else if ((classCount[2] > classCount[1]) && (classCount[2] > classCount[3]))
				resultsMap.put(current, 2.0);
			else 
				resultsMap.put(current, 3.0);
		}
		
		// next step is check success rate, compare prediction with actual
		double rightCount = 0;
		int g = 0;
		Iterator it = resultsMap.entrySet().iterator();
		while (it.hasNext()) {
			Map.Entry prediction = (Map.Entry)it.next();
			// System.out.println("Actual: " + ((ArrayList<Double>)prediction.getKey()).get(33) + " Predicted: " + prediction.getValue());
			if ((((ArrayList<Double>)prediction.getKey()).get(33)).equals(prediction.getValue()))				// if predicted class = real class
				rightCount++;
			g++;
		}
		double success = Double.parseDouble(df.format((rightCount/195)*100)); // percentage success rounded to 4 places
		System.out.println(rightCount + " predictions out of 195 correct");
		System.out.println(success + "% of predictions correct");
		// System.out.println("total " + g);
	}
	
	public static void main(String [] args) {
		KNN z = new KNN("student-mat-normalised.csv", true, 33, 32, new int[]{5,10,15});
		z.readDataset();
		z.trimDataset();
		z.normaliseDataset();
		z.classifyDataset();
	}
	
}
















